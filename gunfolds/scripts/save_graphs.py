import os
from gunfolds.utils import graphkit as gk
from gunfolds.utils import zickle as zkl
import argparse
import random
import distutils.util

REPEATS = 100

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-n", "--NODE", default=30,help="number of nodes.", type=int)
parser.add_argument("-u", "--UNDERSAMPLING", default="2,3,4",help="number of undersampling. e.g. -u=2,3,4", type=str)
parser.add_argument("-d", "--DENITIES",default="0.2,0.25,0.3", help="densities to be ran. e.g. -d=0.2,0.25,0.3", type=str)
parser.add_argument("-s", "--SCC",default="y", help="whether or not use scc for d_rasl. Use y for true and n for false.", type=str)
parser.add_argument("-g", "--DEGREE",default="0.9,2,3,5", help="average degree to be ran. e.g. -g=0.9,2,3,5", type=str)
parser.add_argument("-a", "--ARRAY",default=1, help="number of total batches.", type=int)
parser.add_argument("-b", "--BATCH",default=1, help="slurm batch.", type=int)
args = parser.parse_args()
SCCMODE=bool(distutils.util.strtobool(args.SCC))
dens_list = args.DENITIES.split(',')
dens_list = [float(item) for item in dens_list]
u_list = args.UNDERSAMPLING.split(',')
u_list = [int(item) for item in u_list]
deg_list = args.DEGREE.split(',')
deg_list = [float(item) for item in deg_list]


densities = {3: dens_list,
                 5: dens_list,
                 6: dens_list,
                 7: dens_list,
                 8: dens_list,
                 9: dens_list,
                 10: dens_list,
                 15: dens_list,
                 20: dens_list,  # 0.15, 0.2, 0.25, 0.3],
                 25: dens_list,
                 30: dens_list,
                 35: dens_list,
                 40: dens_list,
                 50: dens_list,
                 60: dens_list}

for nodes in [args.NODE]: #np.sort(densities.keys())[0:]:
    print (nodes, ': ----')
    print ('')
    graphs = []
    # for dens in densities[nodes]:         #for generating graphs with density
    #   e = bfutils.dens2edgenum(dens, n=nodes)
    if not os.path.exists('datasets'):
        os.makedirs('datasets')
    if not SCCMODE:
        for deg in deg_list:                    #for generating graphs with average degree

            for j in range(REPEATS):
                g = gk.bp_mean_degree_graph(nodes, deg)
                dens = gk.density(g)

                for u in u_list:
                    tup ={}
                    tup['gt'],tup['dens'],tup['u'], tup['deg'] = g,dens,u,deg
                    graphs.append(tup)
        random.shuffle(graphs)
        zkl.save(graphs, 'datasets/graphs_node_'+str(nodes)+'.zkl')
    else:
        for deg in deg_list:  # for generating graphs with average degree and ensuring gcd =1
            B = int(REPEATS / args.ARRAY)
            for j in range((int(args.BATCH) -1)*B,(int(args.BATCH))*B):
                g = gk.gcd1_bp_mean_degree_graph(nodes, deg)
                dens = gk.density(g)

                for u in u_list:
                    tup = {}
                    tup['gt'],  tup['dens'], tup['u'], tup['deg'] = g, dens, u, deg
                    graphs.append(tup)
        random.shuffle(graphs)
        zkl.save(graphs, 'datasets/graphs_node_' + str(nodes) + '_deg_'+str(deg)+'_gcd1_batch_'+str(args.BATCH)+'.zkl')
