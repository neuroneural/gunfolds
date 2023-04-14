import argparse
import os
import pickle
import socket
import time

import numpy as np
from timeout_decorator import TimeoutError

from gunfolds.solvers.clingo_rasl import rasl
from gunfolds.utils import bfutils
from gunfolds.utils import graphkit as gk
from gunfolds.utils import zickle as zkl
from gunfolds.utils.calc_procs import get_process_count

CLINGO_LIMIT = 64
PNUM = int(min(CLINGO_LIMIT, get_process_count(1)))
TIMEOUT = 24 * 60 * 60  # seconds = one day
POSTFIX = 'c_RASL'

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-c", "--CAPSIZE", default=10000,
                    help="stop traversing after growing equivalence class to this size.", type=int)
parser.add_argument("-n", "--NODE", default=10, help="number of nodes.", type=int)
parser.add_argument("-u", "--UNDERSAMPLING", default="2,3,4", help="number of undersampling. e.g. -u=2,3,4", type=str)
parser.add_argument("-d", "--DENITIES", default="0.2,0.25,0.3", help="densities to be ran. e.g. -d=0.2,0.25,0.3",
                    type=str)
parser.add_argument("-b", "--BATCH", default=1, help="slurm batch.", type=int)
parser.add_argument("-a", "--ARRAY", default=1, help="number of total batches.", type=int)
parser.add_argument("-p", "--PNUM", default=PNUM, help="number of CPUs in machine.", type=int)
parser.add_argument("-x", "--MAXU", default=30, help="maximum number of undersampling to look for solution.", type=int)
args = parser.parse_args()


# dens_list = args.DENITIES.split(',')
# dens_list = [float(item) for item in dens_list]
# u_list = args.UNDERSAMPLING.split(',')
# u_list = [int(item) for item in u_list]


def clingo_caller(g):
    startTime = int(round(time.time() * 1000))
    c = rasl(g, capsize=args.CAPSIZE, urate=min(args.MAXU, (3 * len(g) + 1)), pnum=args.PNUM, timeout=TIMEOUT)
    endTime = int(round(time.time() * 1000))
    sat_time = endTime - startTime
    return c, sat_time


def fan_wrapper(graph_true, g1_dens, rate, g1_deg):
    den_distribution = np.zeros(11)
    output = {}
    if not rate < len(bfutils.all_undersamples(graph_true)):
        print("input graph reaches super clique if undersampled. Moving on to next graph")
        return output, den_distribution
    try:
        try:
            graph = bfutils.undersample(graph_true, rate)
            if not np.prod(bfutils.g2num(graph)):
                print("input graph is empty. Moving on to next graph")
                return output, den_distribution
            else:
                c, sat_time = clingo_caller(graph)

        except TimeoutError:
            c = None
            sat_time = None
            print("Time Out. {:10} seconds have passed. moving to next graph".format(TIMEOUT))

    except MemoryError:
        c = None
        sat_time = None
        print('memory error...')

    if sat_time is not None:
        print("{:8} : {:4}  {:10} seconds".format(round(g1_dens, 3), len(c), round(sat_time / 1000., 3)))
    n = len(graph_true)
    if c is not None:
        for answer in c:
            index = int((gk.density(bfutils.num2CG(answer[0], n))) * 10)
            den_distribution[index] += 1
        output = {'gt': graph_true,
                  'solutions': {'eq_size': len(c), 'ms': sat_time},
                  'u': rate, 'dens': g1_dens, 'deg': g1_deg}

    return output, den_distribution


for nodes in [args.NODE]:
    print(nodes, ': ----')
    print('')
    # all_graphs = zkl.load('datasets/graphs_node_'+str(nodes)+'.zkl')
    # all_graphs = zkl.load('datasets/graphs.zkl')
    all_graphs = zkl.load('datasets/graphs_node_' + str(nodes) + '_gcd1_.zkl')
    B = int(len(all_graphs) / args.ARRAY)
    batch_graph = all_graphs[(int(args.BATCH) - 1) * B:(int(args.BATCH)) * B]
    density_distibution = np.zeros(11)
    print("{:2}: {:8} : {:10}  {:10}".format('id', 'density', 'eq class', 'time'))
    if not os.path.exists('res_CAP_' + str(args.CAPSIZE) + '_'):
        os.makedirs('res_CAP_' + str(args.CAPSIZE) + '_')
    filename = 'res_CAP_' + str(args.CAPSIZE) + '_/' + \
               socket.gethostname().split('.')[0] + '_maxU_' + str(args.MAXU) + \
               '_nodes_' + str(nodes) + '_batch_' + str(args.BATCH) + '_' + \
               POSTFIX + '_CAPSIZE_' + str(args.CAPSIZE)
    for item in batch_graph:
        if not bfutils.g2num(item['gt']) == 0:
            eqclasses, den_dist = fan_wrapper(graph_true=item['gt'],
                                              g1_dens=item['dens'], rate=item['u'], g1_deg=item['deg'])
            if not len(eqclasses) == 0:
                density_distibution += den_dist
                with open(filename, mode='a+b') as fp:
                    pickle.dump(eqclasses, fp)

    np.savetxt(filename + '_.csv', density_distibution, delimiter=",")
    print(density_distibution)
