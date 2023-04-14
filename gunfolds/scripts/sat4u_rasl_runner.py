import os
from gunfolds.solvers import traversal
from gunfolds.utils import bfutils
from gunfolds.utils import graphkit as gk
from gunfolds.utils import zickle as zkl
import time, socket
import scipy
from gunfolds.solvers.clingo_rasl import rasl
from timeout_decorator import TimeoutError
import argparse
TIMEOUT = 3600 # seconds
MSLTIMEOUT = TIMEOUT
SATTIMEOUT = TIMEOUT
POSTFIX = 'clingo_RASL'
UMAX = 1
INPNUM = 1 # number of randomized starts per graph
REPEATS = 100

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-c", "--CAPSIZE",default=1000, help="stop traversing after growing equivalence class to this size.", type=int)
parser.add_argument("-n", "--NODE", default=6,help="number of nodes.", type=int)
parser.add_argument("-u", "--UNDERSAMPLING", default=2,help="number of undersampling.", type=int)
parser.add_argument("-d", "--DENITIES",default="0.2,0.25,0.3", help="densities to be ran. e.g. -d=0.2,0.25,0.3", type=str)
args = parser.parse_args()
dens_list = args.DENITIES.split(',')
dens_list = [float(item) for item in dens_list]


#@timeout_decorator.timeout(SATTIMEOUT, use_signals=False)
def sat_caller(g):
    startTime = int(round(time.time() * 1000))
    c = rasl(g,capsize=args.CAPSIZE)
    endTime = int(round(time.time() * 1000))
    sat_time = endTime-startTime
    return c, sat_time


def fan_wrapper(n=10,k=10):
    scipy.random.seed()

    sat_time = None
    s = None
    c = None
    while True:
        try:
            g = gk.ringmore(n,k)
            gdens = traversal.density(g)
            g_n = bfutils.undersample(g, args.UNDERSAMPLING - 1)

            try:
                c, sat_time = sat_caller(g_n)
            except TimeoutError:
                c = None
                sat_time = None

            if sat_time is not None:
                print ("{:8} : {:4}  {:10} seconds".\
                  format( round(gdens,3), len(c),
                             round(sat_time/1000.,3)))
            output = {'gt'  : g,
                      'solutions' : {'eq':c,'ms':sat_time}}
        except MemoryError:
            print ('memory error... retrying')
            continue
        break

    return output

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
    z = {}
    for dens in densities[nodes]:
        print ("{:2}: {:8} : {:10}  {:10}".format('id', 'density', 'eq class', 'time'))
        e = bfutils.dens2edgenum(dens, n=nodes)
        folds =[]
        for i in range(REPEATS):
            eqclasses = fan_wrapper( n=nodes, k=e)
            folds.append(eqclasses)
        z[dens] = folds
        if not os.path.exists('res_CAP_'+str(args.CAPSIZE) + '_'):
            os.makedirs('res_CAP_' + str(args.CAPSIZE) + '_')
        zkl.save(z[dens],
                 'res_CAP_'+str(args.CAPSIZE) + '_/'+socket.gethostname().split('.')[0]+\
                     '_nodes_'+str(nodes)+'_density_'+str(dens)+'_G'+str(args.UNDERSAMPLING)+'_'+POSTFIX+'_CAPSIZE_'+\
                     str(args.CAPSIZE)+'_.zkl')
        print ('')

