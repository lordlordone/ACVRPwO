import os
import csv
import argparse
from pulp import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



def read_csv(file_path):
    constants = {
        'dist_cost': None,
        'outsourcing_cost': None,
        'cap_w': None,
        'cap_v': None,
    }
    demand = []
    dist_matrix = []

    with open(file_path, 'r') as file:
        csv_reader = csv.reader(file)

        # Extract and assign capacity values to the dictionary
        constants['dist_cost'] = float(next(csv_reader)[0])
        constants['outsourcing_cost'] = float(next(csv_reader)[0])
        constants['cap_w'] = float(next(csv_reader)[0])
        constants['cap_v'] = float(next(csv_reader)[0])

        # Extract the first row as it contains distance vector of the depot
        dist_matrix.append( [ float(x) for x in next(csv_reader)] )

        # Process demand and distance matrix
        for demand_row, dist_row in zip(csv_reader, csv_reader):
            demand.append(tuple(map(float, demand_row)))
            dist_matrix.append(list(map(float, dist_row)))

    return constants, demand, dist_matrix


def get_vehicle2customer(z, vehicles, nodes):

    vehicle2customer = {}

    for v in vehicles:

        if sum( [z[n][v].varValue for n in nodes ] ) >= 1:
            vehicle2customer[v] = []
            for n in nodes:

                if z[n][v].varValue == 1:
                    vehicle2customer[v].append(n)

    return vehicle2customer


def get_activated_arcs(x, nodes):

    activated_arcs = []

    for i in nodes:
        for j in nodes:

            if x[i][j].varValue == 1:
                activated_arcs.append( (i,j) )

    return activated_arcs


def get_route(vehicle2customer, activated_arcs):

    route = {}
    done = True

    for v in vehicle2customer.keys():

        i = 0
        route[v] = []
        while(done):

            possible_arcs = [ (i,customer) for customer in vehicle2customer[v] ]

            for arc in possible_arcs:

                if arc in activated_arcs:
                    route[v].append(arc)
                    i = arc[1]

                    if arc[1] == 0:
                        done = False

        done = True

    return route


def get_graphs(route):

    for key in route.keys():

        G = nx.DiGraph()
        G.add_edges_from(route[key])
        nx.draw_networkx(G)
        plt.savefig(str(key) + "_route.png")
        plt.clf()
        G.clear()


###################################################### parser CLI #######################################################
parser = argparse.ArgumentParser()


parser.add_argument('filename', type=str, help='Specify the csv file where the data will be taken.')
parser.add_argument('-lp', type=str, help='Following with a name, it provides the lp file of the model.')
parser.add_argument('-show_vars', action='store_true', help='It furnishes the solution values acquired through the solver.')
parser.add_argument('-show_graph', action='store_true', help='It generates graphical representations illustrating the routes that each vehicle is required to take.')
parser.add_argument('-timelimit', type=int, help='It enables the imposition of a time constraint on the solver.')


args = parser.parse_args()


csv_file_path = args.filename
lp_filename = args.lp
timelimit = args.timelimit


if timelimit is None:
    timelimit = 60


################################### defining constants ################################

# Get data from CSV
const, demand, dist_matrix = read_csv(csv_file_path)


n_nodes = range(len(dist_matrix))
n_vehicles = n_customers = range(1, len(demand)+1)

arcs = [(i, j) for i in n_nodes for j in n_nodes if i != j]


# arcs_star is a set of all possible arcs except itself
# and the arcs linked to the depot
arcs_star = [arc for arc in arcs if arc[0] != 0 and arc[1] != 0]


# Initializing dictionaries for frontward and backward arcs
frontward_arcs = {}
backward_arcs = {}

# Defining frontward arcs
for i in n_nodes:
    # For each node, create a list of arcs excluding the self-loop (i, i)
    frontward_arcs[i] = [(i, j) for j in n_nodes if i != j]

# Defining backward arcs
for i in n_nodes:
    # For each node, create a list of arcs excluding the self-loop (i, i)
    backward_arcs[i] = [(j, i) for j in n_nodes if i != j]

##################### adding variables to the models and model definition ###############
print('initializing variables...')


x = LpVariable.dicts('x', (n_nodes, n_nodes), 0,1, cat=LpInteger)      # x_ij are our arc variables
z = LpVariable.dicts("z", (n_nodes, n_vehicles), 0,1, cat=LpInteger)  # z_ik are the vehicle variables
v = LpVariable.dicts("v", (n_nodes, n_nodes), 0, cat=LpContinuous)     # v_ij are the flow volume variables
w = LpVariable.dicts("w", (n_nodes, n_nodes), 0, cat=LpContinuous)     # w_ij are the flow weight variables
l = LpVariable.dicts("l", n_customers, 0, 1, cat=LpInteger)          # l_i is the outsourced customer variables
b = LpVariable.dicts('b', (n_nodes,n_nodes,n_vehicles), 0,1,cat=LpInteger )

model = LpProblem("acvrp", LpMinimize)

######################### defining objective function ###################################
print('building model...')


model += const['dist_cost'] * lpSum([ dist_matrix[i][j] * x[i][j] for (i,j) in arcs ]) + \
         const['outsourcing_cost'] * lpSum([demand[i-1][1] * l[i] for i in n_customers])
                            # demand[i-1] because n_customer is a range from 1 to len(demand)+1


################################ adding constraints ####################################
# 1) frontward_sum(x_ji) = 1            for all nodes except node 0
for node in frontward_arcs.keys():

    if node != 0:                                                           # 0 not in arc
        model += lpSum([x[ arc[0] ][ arc[1] ] for arc in frontward_arcs[node] ]) == 1

# 2)  backward_sum(x_ij) = 1            for all nodes except node 0
for node in backward_arcs.keys():

    if node != 0:                                                           # 0 not in arc
        model += lpSum([x[ arc[0] ][ arc[1] ] for arc in backward_arcs[node] ]) == 1


# 3) backward_sum(x_i0) <= |demand|
model += lpSum([x[arc[0]][arc[1]] for arc in backward_arcs[0] ] ) <= ( len(demand)  )

# 4) frontward_sum(x_0i) <= |demand|
model += lpSum([x[arc[0]][arc[1]] for arc in frontward_arcs[0] ] ) <= ( len(demand)  )


# 5) backward_sum(v_ji) - frontward_sum(v_ij) = v_i         for all nodes except node 0
for node in n_nodes:

    if node != 0:
        model += lpSum([ v[ arc[0] ][ arc[1] ] for arc in backward_arcs[node]  ]) - \
                 lpSum([ v[ arc[0] ][ arc[1] ] for arc in frontward_arcs[node] ]) == demand[node-1][0]


# 6) backward_sum(w_ji) - frontward_sum(w_ij) = w_i         for all nodes except node 0
for node in n_nodes:

    if node != 0:
        model += lpSum([ w[ arc[0] ][ arc[1] ] for arc in backward_arcs[node]  ]) - \
                 lpSum([ w[ arc[0] ][ arc[1] ] for arc in frontward_arcs[node]  ]) == demand[node-1][1]


# 7) 0 <= v_ij <= C_v * x_ij                                for all arcs
for arc in arcs:
    model += v[arc[0]][arc[1]] >= 0
    model += v[arc[0]][arc[1]] <= const['cap_v'] * x[arc[0]][arc[1]]

# 8) 0 <= w_ij <= C_w * x_ij                                for all arcs
for arc in arcs:
    model += w[arc[0]][arc[1]] >= 0
    model += w[arc[0]][arc[1]] <= const['cap_w'] * x[arc[0]][arc[1]]

# 9) backward_sum(v_j0) - frontward_sum(v_0j) == - ( sum(v_i) + sum(l_i * v_i) )_execpt_node_0
model += lpSum([v[ arc[0] ][ arc[1] ] for arc in backward_arcs[0]] ) - \
         lpSum([v[ arc[0] ][ arc[1] ] for arc in frontward_arcs[0]] ) == \
            sum([ -demand[i][0] for i in range(0, len(demand)) ]) + lpSum( [ l[i+1] * demand[i][0] for i in range(0, len(demand)) ] )


# 10) backward_sum(w_j0) - frontward_sum(w_0j) == - ( sum(w_i) + sum(l_i * w_i) )_execpt_node_0
model += lpSum([w[ arc[0] ][ arc[1] ] for arc in backward_arcs[0]] ) - \
         lpSum([w[ arc[0] ][ arc[1] ] for arc in frontward_arcs[0]] ) == \
            sum( [ -demand[i][1] for i in range(0, len(demand)) ] ) + lpSum( [ l[i+1] * demand[i][1] for i in range(0, len(demand)) ] )


# 11) vehicle_sum(z_ik) == 1 - l_i      for all nodes except node 0
for node in n_nodes:
    if node != 0:
        model += lpSum( [ z[node][k] for k in n_vehicles ] ) == 1 - l[node]


# 12) vehicle_sum(z_0k) <= frontward sum(x_0i)
model += lpSum( [ z[0][k] for k in n_vehicles  ] ) <= \
            lpSum( [ [ x[arc[0]][arc[1]] for arc in frontward_arcs[0] ] ] )


# 13) sum(v_i * z_ik) <= C_v            for all vehciles
for vehicle in n_vehicles:
    model += lpSum( [ demand[node-1][0] * z[node][vehicle] for node in n_nodes if node != 0 ] ) <= const['cap_v']


# 14) sum(w_i * z_ik) <= C_w            for all vehciles
for vehicle in n_vehicles:
    model += lpSum( [ demand[node-1][1] * z[node][vehicle] for node in n_nodes if node != 0 ] ) <= const['cap_w']


############################################################################### new constraints ################################################################

#18) b_ijk >= z_ik + z_jk - 1      for all arcs in arcs_star and for all vehicles
for arc in arcs_star:
    for vehicle in n_vehicles:
        model+= b[arc[0]][arc[1]][vehicle] >=  z[arc[0]][vehicle] + z[arc[1]][vehicle] -1

#19) b_ijk <= z_ik      for all arcs in arcs_star and for all vehicles
for arc in arcs_star:
    for vehicle in n_vehicles:
        model+= b[arc[0]][arc[1]][vehicle] <= z[arc[0]][vehicle]

#20) b_ijk <= z_jk      for all arcs in arcs_star and for all vehicles
for arc in arcs_star:
    for vehicle in n_vehicles:
        model+= b[arc[0]][arc[1]][vehicle] <= z[arc[1]][vehicle]

#21) x_ij + x_ji <= b_ijk       for all arcs in arcs_star and for all vehicles
for arc in arcs_star:
    model += x[arc[0]][arc[1]] + x[arc[1]][arc[0]] <= lpSum([b[arc[0]][arc[1]][k] for k in n_vehicles])

#22) z_ik <= z_0k           for all nodes and for all vehicles
for node in n_nodes:
        for vehicle in n_vehicles:
            model += z[node][vehicle] <= z[0][vehicle]
#####################################################################################################################################################
'''
                                    # symmetry breaking constraints 

# Vehicle constraints   (VC)
for vehicle in n_vehicles:
    if vehicle != 1:
        model += z[0][vehicle] <= z[0][vehicle-1]

# Hierarchical constraints type 1       (HC 1)
for node in n_nodes:
    for vehicle in n_vehicles:
        if node != 0 and vehicle != 1:
            model += z[node][vehicle] <= lpSum([z[node2][vehicle-1] for node2 in n_nodes if node2 < node ])

'''

# defining the LP file
if lp_filename is None:
    pass
else:
    if not '.lp' in lp_filename:
        lp_filename += '.lp'
    model.writeLP( lp_filename )


print('Solving...')
model.solve(PULP_CBC_CMD(msg=1, timeLimit=timelimit))
print("status =" , LpStatus[ model.status ] )


if LpStatus[ model.status ] == "Not Solved":
    print('Finished')
    exit()

print( "objective =" , value( model.objective ) )


# it makes a txt file with the values of every variables of the model
if args.show_vars:

    vars_filename = csv_file_path + '_vars.txt'
    with open(vars_filename, mode='w', newline='') as file:

        for v in model.variables():
            if v.varValue >= 1:
                file.write( str(v.name) + "=" + str(v.varValue) + '\n' )


# it makes a graph for each activated vehicle, showing the route it would have to do
if args.show_graph:

    vehicle2customer = get_vehicle2customer(z, n_vehicles, n_nodes)
    activated_arcs = get_activated_arcs(x, n_nodes)
    route = get_route(vehicle2customer, activated_arcs)

    directory_name = csv_file_path.replace('.csv', '_route')

    if not os.path.exists(directory_name):
        os.mkdir(directory_name)

    os.chdir(directory_name)


    get_graphs(route)


print("finished")