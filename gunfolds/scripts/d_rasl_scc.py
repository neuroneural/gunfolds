import os
from gunfolds.utils import bfutils
from gunfolds.utils import zickle as zkl
import numpy as np
import time
from gunfolds.solvers.clingo_rasl import drasl
from gunfolds.utils.calc_procs import get_process_count
import distutils.util
import pickle

import argparse
from gunfolds.utils import graphkit as gk

CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
TIMEOUT = 24 * 60 * 60  # seconds = 24 hours
POSTFIX = 'DraslScc'

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-c", "--CAPSIZE", default=10000,
                    help="stop traversing after growing equivalence class to this size.", type=int)
parser.add_argument("-n", "--NODE", default=64, help="number of nodes.", type=int)
parser.add_argument("-u", "--UNDERSAMPLING", default="2,3,4", help="number of undersampling. e.g. -u=2,3,4", type=str)
parser.add_argument("-d", "--DENITIES", default="0.2,0.25,0.3", help="densities to be ran. e.g. -d=0.2,0.25,0.3",
                    type=str)
parser.add_argument("-m", "--MODEDOUBLE", default="t",
                    help="whether or not use two input for d_rasl. Use y for true and n for false.", type=str)
parser.add_argument("-z", "--SCCSIZE", default=8, help="size of each SCC in the graph.", type=int)
parser.add_argument("-b", "--BATCH", default=1, help="slurm batch.", type=int)
parser.add_argument("-a", "--ARRAY", default=1, help="number of total batches.", type=int)
parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
parser.add_argument("-x", "--MAXU", default=20, help="maximum number of undersampling to look for solution.", type=int)
args = parser.parse_args()
MODEDOUBLE = bool(distutils.util.strtobool(args.MODEDOUBLE))


# dens_list = args.DENITIES.split(',')
# dens_list = [float(item) for item in dens_list]
# u_list = args.UNDERSAMPLING.split(',')
# u_list = [int(item) for item in u_list]


# @timeout_decorator.timeout(TIMEOUT)
def clingo_caller(g):
    start_time = int(round(time.time() * 1000))
    c = drasl(g, capsize=args.CAPSIZE, urate=min(args.MAXU, (3 * len(g[0]) + 1)), timeout=TIMEOUT, scc=True,
              pnum=args.PNUM)
    end_time = int(round(time.time() * 1000))
    sat_time = end_time - start_time
    return c, sat_time


def fan_wrapper(graph_true, g1_dens, rate):
    den_distribution = np.zeros(11)
    state = 'normal'
    output = {}
    max_rate = rate
    if MODEDOUBLE:
        max_rate = rate + 1
    if not max_rate <= len(bfutils.all_undersamples(graph_true)):
        print("input graph will converge if undersampled. Moving on to next graph")
        state = 'converged'
        output = {'gt': graph_true,
                  'solutions': {'eq_size': 0, 'ms': 0},
                  'u': rate, 'dens': g1_dens, 'state': state}
        return output, den_distribution
    try:
        # try:
        graphs = [bfutils.undersample(graph_true, rate)]
        if MODEDOUBLE:
            graphs.append(bfutils.undersample(graph_true, rate + 1))
        if not np.prod([bfutils.g2num(x) for x in graphs]):
            print("input graph is empty. Moving on to next graph")
            return output, den_distribution
        else:
            c, sat_time = clingo_caller(graphs)

        # except TimeoutError:
        #     c = None
        #     sat_time = None
        #     print("Time Out. {:10} seconds have passed. moving to next graph".format(TIMEOUT))
        #     return output, den_distribution

    except MemoryError:
        print('memory error... moving on')
        output = {'gt': graph_true,
                  'solutions': {'eq_size': 0, 'ms': 0},
                  'u': rate, 'dens': g1_dens, 'state': 'OutOfMemory'}
        return output, den_distribution

    if sat_time is not None:
        print("{:8} : {:4}  {:10} seconds".format(round(g1_dens, 3), len(c), round(sat_time / 1000., 3)))
    n = len(graph_true)
    if sat_time / 1000 > TIMEOUT:
        state = 'timeout'
    if c is not None:
        for answer in c:
            index = int((gk.density(bfutils.num2CG(answer[0], n))) * 10)
            den_distribution[index] += 1
        output = {'gt': graph_true,
                  'solutions': {'eq_size': len(c), 'ms': sat_time},
                  'u': rate, 'dens': g1_dens, 'state': state}

    return output, den_distribution


for nodes in [args.NODE]:
    all_graphs = zkl.load(
        'datasets/graph_scc_node_' + str(args.NODE) + '_scc_sizes_' + str(args.SCCSIZE) + '_connected.zkl')
    B = int(len(all_graphs) / args.ARRAY)
    batch_graph = all_graphs[(int(args.BATCH) - 1) * B:(int(args.BATCH)) * B]
    print("nodes : {0:}, max_U : {1:} SCC Size : {2:}, Double Mode : {3:}, Batch : {4:} -----".format(nodes,
                                                                                                      args.MAXU,
                                                                                                      args.SCCSIZE,
                                                                                                      MODEDOUBLE,
                                                                                                      args.BATCH))
    print('')
    density_distibution = np.zeros(11)
    print("{:2}: {:8} : {:10}  {:10}".format('id', 'density', 'eq class', 'time'))
    if not os.path.exists('res_CAP_' + str(args.CAPSIZE) + '_'):
        os.makedirs('res_CAP_' + str(args.CAPSIZE) + '_')
    filename = 'res_CAP_' + str(args.CAPSIZE) + '_/' + \
               'nodes_' + str(nodes) + '_maxU_' + str(args.MAXU) + '_batch_' + str(args.BATCH) + '_' + \
               'SccSize_' + str(args.SCCSIZE) + '_' + \
               POSTFIX + '_CAPSIZE_' + str(args.CAPSIZE) + '_doubleMode_' + str(MODEDOUBLE)
    for item in batch_graph:
        if not bfutils.g2num(item['gt']) == 0:
            eqclasses, den_dist = fan_wrapper(graph_true=item['gt'],
                                              g1_dens=item['dens'], rate=item['u'])
            if not len(eqclasses) == 0:
                density_distibution += den_dist
                with open(filename, mode='a+b') as fp:
                    pickle.dump(eqclasses, fp)

np.savetxt(filename + '_.csv', density_distibution, delimiter=",")
