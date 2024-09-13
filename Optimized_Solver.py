import os
import csv
import math
import argparse
from pulp import *
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt



############################################### Class #########################################################


class MasterRP:
# cr is the cost for every route and p is the list of patterns
    def __init__(self,demand, const, cr, p):
        self.const = const['outsourcing_cost']
        self.demand = [d[1] for d in demand]
        self.cr = cr
        self.p = p
        self.n_patterns = range( len(self.p) )
        self.n_customers = range( len(self.p[0]) )


    def set_vars(self):

        var1 = LpVariable.dicts('x', self.n_patterns, 0, 1, cat=LpInteger)
        var2 = LpVariable.dicts('l', self.n_customers, 0, 1, cat=LpInteger)

        return var1, var2


    def set_objFunction(self, var1, var2):
        return lpSum(  [ self.cr[pattern] * var1[pattern] for pattern in self.n_patterns ] )  \
                      + lpSum([self.const *  self.demand[customer] * var2[customer] for customer in self.n_customers])



    def build_RMP(self):

        # init vars model
        x, l = self.set_vars()


        # build model
        print('building restricted master problem...')
        self.model = LpProblem('ACVRP', LpMinimize)


        # defining objective function
        self.model += self.set_objFunction(x,l)

        # subject to:

            # l[customer] + sum_route( p_route_customer * x_route ) >= 1             for all customers
        for customer in self.n_customers:
            constraint = LpConstraint( l[customer] + lpSum( [  self.p[pattern][customer] * x[pattern] for pattern in self.n_patterns  ]  ), \
                              sense= LpConstraintGE, name="d" + str(customer), rhs=1)
            self.model += constraint


            # sum_route( x_route ) <= |P|
        self.model += lpSum( x[pattern] for pattern in self.n_patterns ) <= len(self.p)   # because n_patterns starts to 0


        return self.model


    def solve(self, relaxed):
        model = self.build_RMP()
        model.solve(PULP_CBC_CMD(mip = relaxed ,msg=0, options=['dualSimplex']))

        print("status MP =", LpStatus[model.status])
        print("objective MP =", value(model.objective))


        self.route_vars = []
        get_id = lambda str: int(str.split('_')[1])

        for v in model.variables():
            if 'x_' in v.name:
                id = get_id(v.name)
                self.route_vars.append([ id, int(v.varValue) ])


        duals = []
        for name, c in list(model.constraints.items()):
            if 'd' in name:
                duals.append(c.pi)


        return duals



    def get_varValue(self):
        return self.route_vars



class SubProbelm:
    def __init__(self, a, bs, fs, dist,
                 duals, constants, demand, heur= False):

        self.a = a
        self.bs = bs
        self.fs = fs
        self.dist = dist
        self.duals = duals
        self.const = constants
        self.demand = demand
        self.heur = heur


        self.n_nodes = range( len(self.dist) )
        self.n_customers = range( len(self.dist) -1 )      # -1 because there is the depot which is not a customer



    def solve_knapsack(self):

        n = range(len(self.duals))
        y = LpVariable.dicts('y', range(0, len(self.duals)), 0, 1, cat=LpInteger)

        # set knapsack problem
        self.ks = LpProblem("knapsack", LpMaximize)

        ######################### defining objective function ###################################

        # set objective fucntion
        self.ks += lpSum([self.duals[i] * y[i] for i in n])

        # subject to:
        self.ks += lpSum([self.demand[i][0] * y[i] for i in n]) <= self.const['cap_v']
        self.ks += lpSum([self.demand[i][1] * y[i] for i in n]) <= self.const['cap_w']
        self.ks += lpSum([y[i] for i in n]) >= 2

        self.ks.solve(PULP_CBC_CMD(mip=True, msg=0))


        pattern = [ 0 ] * len(self.duals)       # generate a list of n 0s


        for v in self.ks.variables():   # self.ks obtained from solve_knapsack method

            # strip the characters 'y_' in order to get just the index value
            idx = int(v.name.strip('y_'))
            pattern[idx] = int(v.varValue)

        return pattern



    def solve_subproblem(self):


        # make a copy of the dual list including the dual variable of the depot which is 0
        temp = [0]
        for dual in self.duals:
            temp.append(dual)


        x = LpVariable.dicts('x', (self.n_nodes, self.n_nodes), 0 ,1, cat = LpInteger)
        y = LpVariable.dicts('y', range(0,len(temp)), 0, 1, cat=LpInteger)
        v = LpVariable.dicts('v', (self.n_nodes, self.n_nodes), 0, cat=LpContinuous)
        w = LpVariable.dicts('w', (self.n_nodes, self.n_nodes), 0, cat=LpContinuous)


        self.model = LpProblem('PCTSP', LpMinimize)

        # objective function
        self.model += self.const['dist_cost'] * lpSum( [ self.dist[i][j] * x[i][j] for (i,j) in self.a ] ) - \
                        lpSum( [ temp[i] * y[i] for i in range(len(temp)) ] )


        # subject to:
        # 1) y_0 = 1
        self.model += y[0] == 1


        if self.heur:
            p = self.solve_knapsack()
            for i in range(len(p)):
                self.model += y[i+1] == p[i]

        # 2) frontward_sum(x_ji) = 1            for all nodes
        for node in self.fs.keys():
            self.model += lpSum( [ x[arc[0]][arc[1]]  for arc in self.fs[node]  ] ) == y[node]

        # 3) backward_sum(x_ij) = 1            for all nodes
        for node in self.bs.keys():
            self.model += lpSum([x[arc[0]][arc[1]] for arc in self.bs[node]]) == y[node]



        # 4) backward_sum(v_ji) - frontward_sum(v_ij) = v_i         for all nodes except node 0
        for node in self.n_nodes:
            if node != 0:

                self.model += lpSum( [ v[arc[0]][arc[1]] for arc in self.bs[node] ] ) - \
                         lpSum( [ v[arc[0]][arc[1]] for arc in self.fs[node] ] ) == self.demand[node-1][0] * y[node]
                                                                                                    # because the depot is not a customer

        # 5) backward_sum(w_ji) - frontward_sum(w_ij) = w_i         for all nodes except node 0
        for node in self.n_nodes:
            if node != 0:

                self.model += lpSum( [ w[arc[0]][arc[1]] for arc in self.bs[node] ] ) - \
                        lpSum( [ w[arc[0]][arc[1]] for arc in self.fs[node] ] ) == self.demand[node-1][1] * y[node]



        # 6) 0 <= v_ij <= C_v * x_ij                                for all arcs
        for arc in self.a:
            self.model += v[arc[0]][arc[1]] >= 0
            self.model += v[arc[0]][arc[1]] <= self.const['cap_v'] * x[arc[0]][arc[1]]

        # 7) 0 <= w_ij <= C_w * x_ij                                for all arcs
        for arc in self.a:
            self.model += w[arc[0]][arc[1]] >= 0
            self.model += w[arc[0]][arc[1]] <= self.const['cap_w'] * x[arc[0]][arc[1]]



        # 8) backward_sum(v_j0) - frontward_sum(v_0j) == - ( sum(v_i) + sum(l_i * v_i) )_execpt_node_0
        self.model += lpSum(  [  v[arc[0]][arc[1]] for arc in self.bs[0]  ]  ) - \
                            lpSum(  [  v[arc[0]][arc[1]] for arc in self.fs[0]  ]  ) == \
                            sum([-self.demand[customer][0] * y[customer+1] for customer in self.n_customers])


        # 9) backward_sum(w_j0) - frontward_sum(w_0j) == - ( sum(w_i) + sum(l_i * w_i) )_execpt_node_0
        self.model += lpSum(  [  w[arc[0]][arc[1]] for arc in self.bs[0]  ]  ) - \
                            lpSum(  [  w[arc[0]][arc[1]] for arc in self.fs[0]  ]  ) == \
                            sum([-self.demand[customer][1] * y[customer+1] for customer in self.n_customers])

        # 10) sum(v_i * y_i) <= C_v
        self.model += lpSum( [ self.demand[customer][0] * y[customer + 1] for customer in self.n_customers ] ) <= self.const['cap_v']

        # 11) sum(w_i * y_i) <= C_w
        self.model += lpSum( [ self.demand[customer][0] * y[customer + 1] for customer in self.n_customers ] ) <= self.const['cap_v']


        #self.model.writeLP('test_subproblem')

        self.model.solve(PULP_CBC_CMD(msg=0))
        print("status SubProblem =", LpStatus[self.model.status])
        print("objective SubProblem =", value(self.model.objective))
        rc = value(self.model.objective)    # reduced cost

        # getting the pattern of the selected customers
        self.pattern = [0] * (len(self.duals) + 1)
        for v in self.model.variables():
            if 'y_' in v.name:
                idx = int(v.name.strip('y_'))
                self.pattern[idx] = int(v.varValue)

        self.pattern.pop(0)     # taking out the depot from the pattern

        # because the reduced cost (rc) is equal to cj + sum_( dual_i * y_i )_for all i
        cj = rc + sum( [ self.duals[i] * self.pattern[i] for i in range(len(self.duals)) ] )

        return cj, rc



    def get_pattern(self):
        return self.pattern



    def get_graph(self):

        arcs = []

        for v in self.model.variables():
            if 'x_' in v.name and v.varValue == 1:
                arcs.append( [int(s) for s in re.findall(r'\d+', v.name)] )  # strip all the character and get only the digits that compose the arc


        # building graph through networkx
        g = nx.DiGraph()
        g.add_edges_from(arcs)

        return g




############################################# Functions ###############################################


# Read the csv file and return the constants of the model
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


# Set an initial set of patterns. It starts with a pattern where no customer is served by our vehicles
# and a set of patterns where each customer is served by just one vehicle. It returns the cost of each of these patterns
# and the patterns
def set_initial_patterns(const, dist):
    dist_const = const['dist_cost']
    # r will be a dictionary of routes where the key will be the route id and the value will be a graph
    # the route w
    r = {0: nx.empty_graph()}
    n = len(dist) - 1


    patterns = np.zeros(( 1, n))
    diag_arr = np.eye( n )          # this creates a matrix of 0s with a diagonal of 1s
    patterns = np.vstack((patterns, diag_arr)).astype(int).tolist()

    cost_vec = [0]


    for i in range(1, n+1):

        cost_vec.append(  (dist_const * dist[0][i]) + (dist_const * dist[i][0] ) )

        r[i] = nx.DiGraph()
        r[i].add_edges_from([ [0,i],[i,0] ])


    return cost_vec, patterns, r



###################################################### parser CLI #######################################################

parser = argparse.ArgumentParser()


parser.add_argument('filename', type=str, help='Specify the csv file where the data will be taken.')
parser.add_argument('-show_graph', action='store_true', help='It generates and saves graphical representations to illustrate the routes that each vehicle is required to take.')
parser.add_argument('-heur', action='store_true', help='It applies a heuristic to solve the subproblem faster')
parser.add_argument('-timelimit', type=int, help='It enables the imposition of a time constraint on the solver.')


args = parser.parse_args()


csv_file_path = args.filename
timelimit = args.timelimit
heur = args.heur


if heur is None:
    heur = False

if timelimit is None:
    timelimit = math.inf

################################### defining constants ################################

# Get data from CSV
const, demand, dist_matrix = read_csv(csv_file_path)


n_nodes = range(len(dist_matrix))
# set the arcs
arcs = [(i, j) for i in n_nodes for j in n_nodes if i != j]

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



cost_vec, patterns, route = set_initial_patterns(const, dist_matrix)


##################################### Column Generation ################################################

# set restricted master problem
relaxed = False
id_route = max([  id for id in route.keys() ])
import time
start_time = time.time()


rmp = MasterRP(demand, const, cost_vec, patterns)

while(not relaxed):
    duals = rmp.solve(relaxed)

    sp = SubProbelm(arcs, backward_arcs, frontward_arcs, dist_matrix,
                    duals, const, demand, heur)

    cj, rc = sp.solve_subproblem()


    t = time.time()
    t -= start_time
    print('elapsed time: ', round(t,2))
    print()


    if rc < -0.000001 and (t <= timelimit ):

        cost_vec.append(cj)
        patterns.append(sp.get_pattern())

        id_route += 1
        route[id_route] = sp.get_graph()

        rmp = MasterRP(demand, const, cost_vec, patterns)

    else:
        relaxed = True
        rmp = MasterRP(demand, const, cost_vec, patterns)
        duals = rmp.solve(relaxed)

        end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")


import shutil
route_vars = rmp.get_varValue()
if args.show_graph:

    # if the directory doesn't exist, create, and change working directory
    if heur:
        directory_name = csv_file_path.replace('.csv', '_route_heur')
    else:
        directory_name = csv_file_path.replace('.csv', '_route')


    if not os.path.exists(directory_name):
        os.mkdir(directory_name)            # make directory

    else:
        shutil.rmtree(directory_name)       # delete directory
        os.mkdir(directory_name)

    os.chdir(directory_name)

    print('saving the graphs')
    # get activated graphs and save
    for id, r in route_vars:
        if r == 1:
            nx.draw_networkx(route[id])
            plt.savefig(str(id) + ".png")
            plt.clf()


print('finished')




