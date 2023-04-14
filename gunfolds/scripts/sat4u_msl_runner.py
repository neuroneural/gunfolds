import os
from gunfolds.solvers import traversal
from gunfolds.utils import graphkit as gk
from gunfolds.utils import bfutils
from multiprocessing import Pool
from functools import partial
from gunfolds.utils import zickle as zkl
import time, socket
import scipy
import numpy as np
from gunfolds.solvers.clingo_msl import msl
import timeout_decorator
from timeout_decorator import TimeoutError
from gunfolds.utils.calc_procs import get_process_count
import argparse

TIMEOUT=3600 # seconds
MSLTIMEOUT = TIMEOUT
SATTIMEOUT = TIMEOUT
POSTFIX='sat4u_clingo'
UMAX = 1
INPNUM = 1 # number of randomized starts per graph
REPEATS = 100
PNUM = get_process_count(INPNUM)
print ('processes: ',PNUM, INPNUM)

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-c", "--CAPSIZE",default=1000, help="stop traversing after growing equivalence class to this size.", type=int)
parser.add_argument("-n", "--NODE", default=6,help="number of nodes.", type=int)
parser.add_argument("-d", "--DENITIES",default="0.2,0.25,0.3", help="densities to be ran. e.g. -d=0.2,0.25,0.3", type=str)
args = parser.parse_args()
dens_list = args.DENITIES.split(',')
dens_list = [float(item) for item in dens_list]

@timeout_decorator.timeout(MSLTIMEOUT, use_signals=True)
def msl_caller(g2):
    s = set()
    startTime = int(round(time.time() * 1000))
    s = traversal.v2g22g1(g2, capsize=args.CAPSIZE)
    endTime = int(round(time.time() * 1000))
    msl_time = endTime-startTime
    return s, msl_time

#@timeout_decorator.timeout(SATTIMEOUT, use_signals=False)
def sat_caller(g2, fold):
    startTime = int(round(time.time() * 1000))
    c = msl(g2)
    endTime = int(round(time.time() * 1000))
    sat_time = endTime-startTime
    c = {x[0] for x in c}
    return c, sat_time

def fan_wrapper(fold,n=10,k=10):
    scipy.random.seed()

    msl_time = None
    sat_time = None
    s = None
    c = None
    while True:
        try:
            g = gk.ringmore(n,k)
            gdens = traversal.density(g)
            g2 = bfutils.increment(g)

            try:
                s, msl_time = msl_caller(g2)
            except TimeoutError:
                s = None
                msl_time = None

            try:
                c, sat_time = sat_caller(g2, fold)
            except TimeoutError:
                c = None
                sat_time = None

            if msl_time is not None:
                print ("msl: {:2}: {:8} : {:4}  {:10} seconds".\
                  format(fold, round(gdens,3), len(s),
                             round(msl_time/1000.,3)))
            if sat_time is not None:
                print ("sat: {:2}: {:8} : {:4}  {:10} seconds".\
                  format(fold, round(gdens,3), len(c),
                             round(sat_time/1000.,3)))
            output = {'gt'  : g,
                      'MSL' : {'eq':s,'ms':msl_time},
                      'SAT' : {'eq':c,'ms':sat_time}}
        except MemoryError:
            print ('memory error... retrying')
            continue
        break

    return output

densities = {6: [0.15],
             8: [0.15],
             10:[0.15],
             15:[0.15],
             20:[0.15],
             25:[0.15],
             30:[0.15],
             35:[0.15],
             40:[0.15],
             45:[0.15],
             50:[0.15],
             55:[0.15],
             60:[0.15],
             65:[0.15],
             70:[0.15]}

for nodes in np.sort(densities.keys())[2:]: # [args.NODE]
    print (nodes, ': ----')
    print ('')
    z = {}
    pool=Pool(processes=PNUM)
    for dens in densities[nodes]:
        print ("{:2}: {:8} : {:10}  {:10}".format('id', 'density', 'eq class', 'time'))
        e = bfutils.dens2edgenum(dens, n=nodes)
        eqclasses = pool.map(partial(fan_wrapper, n=nodes, k=e), range(REPEATS))
        z[dens] = eqclasses
        zkl.save(z[dens],
                 socket.gethostname().split('.')[0]+\
                     '_nodes_'+str(nodes)+'_density_'+str(dens)+'_'+POSTFIX+'_CAPSIZE_'+\
                     str(args.CAPSIZE)+'_.zkl')
        print ('')
    pool.close()
    pool.join()
    zkl.save(z,socket.gethostname().split('.')[0]+'_nodes_'+str(nodes)+'_'+POSTFIX+'_CAPSIZE_'+\
                     str(args.CAPSIZE)+'_.zkl')
    for dens in densities[nodes]:
        os.remove(socket.gethostname().split('.')[0]+\
                  '_nodes_'+str(nodes)+'_density_'+str(dens)+'_'+POSTFIX+'_CAPSIZE_'+\
                     str(args.CAPSIZE)+'_.zkl')
