from gunfolds.utils import bfutils
from gunfolds.utils import graphkit as gk
from gunfolds.utils import zickle as zkl
import argparse
import random

REPEATS = 100

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-n", "--NODE", default=50,help="number of nodes.", type=int)
parser.add_argument("-u", "--UNDERSAMPLING", default="2,3,4",help="number of undersampling. e.g. -u=2,3,4", type=str)
parser.add_argument("-d", "--DENITIES",default="0.2,0.25,0.3", help="densities to be ran. e.g. -d=0.2,0.25,0.3", type=str)
parser.add_argument("-g", "--DEGREE",default="1,2,3,5", help="average degree to be ran. e.g. -g=0.9,2,3,5", type=str)
args = parser.parse_args()
dens_list = args.DENITIES.split(',')
dens_list = [float(item) for item in dens_list]
u_list = args.UNDERSAMPLING.split(',')
u_list = [int(item) for item in u_list]
deg_list = args.DEGREE.split(',')
deg_list = [int(item) for item in deg_list]


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

    # for deg in deg_list:                    #for generating graphs with average degree

    for j in range(REPEATS):
        g = gk.ring_sccs(25,2, dens=0.05, max_cross_connections=2)
        dens = gk.density(g)

        for u in u_list:
            tup ={}
            g2 = bfutils.undersample(g, u)
            tup['gt'],tup['gn'],tup['dens'],tup['u']= g,g2,dens,u
            graphs.append(tup)
    random.shuffle(graphs)
    zkl.save(graphs, 'datasets/graph_scc_node_'+str(nodes)+'_25_2.zkl')
