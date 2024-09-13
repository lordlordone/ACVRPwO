import csv
import random
import numpy as np
import pandas as pd
import requests
import argparse

############################################################## Functions #####################################################################
def get_coord(df, seed, sample_size = None,):
    

    df = df.sample(n = sample_size, random_state= seed)
    # it is needed just the latitude and longitude
    col2delete = [col for col in df.columns if col not in ['longitude', 'latitude']]
    df.drop(columns=col2delete, inplace=True)
    # convert every lat and long from np.float into float
    df = df.astype(float)
    df = df[df.columns[::-1]]   # invert the longiude and latitude columns for format issues derived by OpenStreeMap


    # return a list of tuples, each tuple is a coordinate
    return [tuple(map(float, row)) for row in df.values]

def generate_batches(l, batch_size):
    groups = []
    d = {}

    for i in range(0, len(l), batch_size):

        d['coord'] = l[i:i + batch_size]
        d['start'] = i

        # handle the case where my list is not perfectly divisible by the batch size
        if len(l) < d['start'] + batch_size:
            d['end'] = d['start'] + (len(l) - d['start'])
        else:
            d['end'] = i + batch_size

        groups.append(d)
        d = {}

    return groups

def generate_URL(batch1, batch2):

    def format_coordinates(coords):
        return ";".join([f"{coord[0]},{coord[1]}" for coord in coords])


    source = list(range(batch1['start'], batch1['end']))
    dest = list(range(batch2['start'], batch2['end']))

    # it means it isn't necessary to format the parameters sources and destionations in the url
    if source[0] == dest[0]:

        coords = batch_1['coord'].copy()
        return f"http://router.project-osrm.org/table/v1/driving/{format_coordinates(coords)}?annotations=distance"

    # sources and destinations parameter need to be formatted
    else:
        coords = batch_1['coord'].copy()
        coords.extend(batch_2['coord'])

        # Handle cases where the row index (source) differs from the column index (dest)
        if source[0] < dest[0]:
            source = [x for x in range(0, len(source))]
            dest = [x for x in range(len(source), len(source) + len(dest))]
        else:
            source = [x for x in range(len(dest), len(dest) + len(source))]
            dest = [x for x in range(0, len(dest))]


        url = f"http://router.project-osrm.org/table/v1/driving/{format_coordinates(coords)}?annotations=distance"

        # deriving the strings for parameter sources and destinations
        source = [str(x) + ";" for x in range(len(source))]
        dest = [str(x) + ";" for x in range(len(dest))]

        url += f"&sources={''.join(source).rstrip(';')}&destinations={''.join(dest).rstrip(';')}"

        return url

def get_response(url):

    data = requests.get(url).json()
                                # divided by 1000 because the distances are meters
    return np.around( np.asarray( data['distances'])/1000, decimals= 2)

def random_generation(min, max):
    return round ( random.uniform(min,max) , 2 )

def random_demend(const, n, random_seed):

    random.seed(random_seed)

    scale = random.choice([10,100,1000])
    min_weight = round( random.random() * scale, 2 )
    min_volume = round ( random.random() * scale, 2 )


    while(min_weight > const['max_weight'] or min_volume > const['max_volume']):

        if min_weight > const['max_weight']:
            scale = random.choice([10, 100, 1000])
            min_weight = round(random.random() * scale, 2)

        if min_volume > const['max_volume']:
            scale = random.choice([10,100,1000])
            min_volume = round(random.random() * scale, 2)


    return [(random_generation(min_volume,const['max_volume']),
                random_generation(min_weight,const['max_weight'])) for _ in range(0,n-1)]

def export(const, matrix, d, name_file):

    with open(name_file, mode='w', newline='') as file:

        csv_writer = csv.writer(file)

        # write the constants of the model in the CSV file
        for value in const.values():
            csv_writer.writerow([value] )


        csv_writer.writerow(dist_matrix[0])
        matrix = np.delete(matrix, 0, axis= 0)

        for i in range(0, len(d)):

            # Write the data to the CSV file
            csv_writer.writerow(d[i])
            csv_writer.writerow(matrix[i])
            file.flush()


################################################# Setting command-line interface #############################################################

parser = argparse.ArgumentParser()

# Define command-line arguments
parser.add_argument('weight_capacity', type=int, help='Specify the weight capacity.')
parser.add_argument('volume_capacity', type=int, help='specify the volume capacity.')
parser.add_argument('sample_size', type=int, help='Sample size sample size. It must be integer.')
parser.add_argument('--seed', type=int, help='Select a seed or the script will do it for you.')
parser.add_argument('filename', type=str, help='Specify how you would like to name the file this script will create.')

# Parse the command-line arguments
args = parser.parse_args()


sample_size = args.sample_size

filename = args.filename


if args.seed is None:
    seed = random.randint(0, 10000)
    random.seed(seed)
else:
    seed = args.seed
    random.seed(args.seed)


# Access the values of the arguments
constants = {'transp_cost': round(random.random() * 10, 2),
             'outsourcing_cost': None,
             'max_weight':args.weight_capacity,
             'max_volume': args.volume_capacity}

constants['outsourcing_cost'] = round(constants['transp_cost'] * (1 + random.random() ), 2)


format = '-' + str(seed) + '-' + str(sample_size)
if '.csv' not in filename:
    filename += format + '.csv'
else:
    filename = filename.replace('.csv', '_{}.csv'.format(format))



#################################################### Main ######################################################################################

# this dataset is taken from this site: https://www.kaggle.com/datasets/dgomonov/new-york-city-airbnb-open-data
df = pd.read_csv("AB_NYC_2019.csv")


# OpenStreetMap can't take requests with over 100 coordinates
if sample_size < 100:
    MAX_REQUEST = sample_size
else:
    MAX_REQUEST = 100

list_of_coord = get_coord(df, seed, sample_size=sample_size)
batches = generate_batches(list_of_coord, MAX_REQUEST)

# this is for a sort of loading bar
remaining_time = 5 * ( ( len(list_of_coord) / MAX_REQUEST )**2 )   # OSM takes 5 seconds between a request and another one
print("creating distance matrix...")

row_matrix = None
dist_matrix = None
for batch_1 in batches:
    for batch_2 in batches:

        if row_matrix is None:
            row_matrix = get_response(generate_URL(batch_1, batch_2))

        else:
            arr = get_response(generate_URL(batch_1, batch_2))
            row_matrix = np.hstack((row_matrix, arr))

        remaining_time -= 5
        print('estimated ', remaining_time, ' seconds left...')

    if dist_matrix is None:
        dist_matrix = row_matrix
    else:
        dist_matrix = np.vstack( (dist_matrix, row_matrix) )

    row_matrix = None


print("creating random demand vector...")
demands = random_demend(constants, sample_size, seed)

print("exporting...")
export(constants, dist_matrix, demands, filename)

print('done')
