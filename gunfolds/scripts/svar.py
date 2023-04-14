from __future__ import division
import copy
import functools
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.utils import bfutils as bfu
from gunfolds.conversions import num2CG, g2vec, vec2g
from gunfolds.utils import graphkit as gk
from gunfolds.estimation import linear_model as lm
from gunfolds.estimation import pc
from gunfolds.solvers import traversal as trv
from gunfolds.utils import zickle as zkl
import itertools
from multiprocessing import Pool
import numpy as np
import pprint
from progressbar import ProgressBar, Percentage
import scipy
import socket
import sys
import time

NOISE_STD = '0.1'
DEPTH = 2
DIST = 'beta'
BURNIN = 100
SAMPLESIZE = 2000
PARALLEL = True
POSTFIX = '_H'
EST = 'pc'
INPNUM = 1  # number of randomized starts per graph
CAPSIZE = 100  # stop traversing after growing equivalence class tothis size
REPEATS = 100
PNUM = get_process_count(INPNUM)


def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    import signal

    class TimeoutError(Exception):
        pass

    def handler(signum, frame):
        raise TimeoutError()

    # set the timeout handler
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout_duration)
    try:
        result = func(*args, **kwargs)
    except TimeoutError as exc:
        result = default
    finally:
        signal.alarm(0)

    return result


def hamming_neighbors(v, step):
    l = []
    for e in itertools.combinations(range(len(v)), step):
        b = copy.copy(v)
        for i in e:
            b[i] = int(not b[i])
        l.append(b)
    return l


def find_nearest_reachable(g2, max_depth=4):
    s = trv.v2g22g1(g2, capsize=CAPSIZE, verbose=False)
    # s = trv.edge_backtrack2g1_directed(g2, capsize=CAPSIZE)
    if s:
        return s
    step = 1
    n = len(g2)
    v = g2vec(g2)
    while True:
        l = hamming_neighbors(v, step)
        pbar = ProgressBar(
            widgets=['neighbors checked @ step ' + str(step) + ': ', Percentage(), ' '], maxval=len(l)).start()
        c = 0
        for e in l:
            g = vec2g(e, n)
            if not gk.scc_unreachable(g):
                # s = trv.edge_backtrack2g1_directed(g2, capsize=CAPSIZE)
                s = trv.v2g22g1(g, capsize=CAPSIZE, verbose=False)
            else:
                s = set()
            if s:
                return s
            pbar.update(c)
            c += 1
        pbar.finish()
        if step > max_depth:
            return set()
        step += 1


def wrapper(fold, n=10, dens=0.1):
    scipy.random.seed()
    rate = 2

    r = None
    s = set()
    counter = 0
    while not s:
        scipy.random.seed()
        sst = 0.9
        r = None
        while not r:
            r = lm.getAring(n, dens, sst, False, dist=DIST)
            print sst,
            sys.stdout.flush()
            if sst < 0.03:
                sst -= 0.001
            else:
                sst -= 0.01
            if sst < 0:
                sst = 0.02
        # pprint.pprint(r['transition'].round(2),width=200)
        # d = zkl.load('leibnitz_nodes_'+str(n)+'_OCE_model_.zkl')
        # r = d[dens][fold]
        g = r['graph']
        true_g2 = bfu.undersample(g, rate - 1)
        data = lm.drawsamplesLG(r['transition'], samples=BURNIN + SAMPLESIZE * 2,
                                nstd=np.double(NOISE_STD))
        data = data[:, BURNIN:]
        if np.max(data) > 1000.:
            pprint.pprint(r['transition'].round(2), width=200)
            # raise ValueError
        startTime = int(round(time.time() * 1000))
        if EST == 'pc':
            g2 = pc.dpc(data[:, ::2], pval=0.0001)
        elif EST == 'svar':
            g2 = lm.data2graph(data[:, ::2])
        if trv.density(g2) < 0.7:
            print gk.OCE(g2, true_g2)
            # s = examine_bidirected_flips(g2, depth=DEPTH)
            s = find_nearest_reachable(g2, max_depth=1)
            # s = trv.v2g22g1(g2, capsize=CAPSIZE, verbose=False)
            # s = trv.edge_backtrack2g1_directed(g2, capsize=CAPSIZE)
            # s = timeout(trv.v2g22g1,
            # s = timeout(trv.edge_backtrack2g1_directed,
            #            args=(g2,CAPSIZE),
            #            timeout_duration=1000, default=set())
            print 'o',
            sys.stdout.flush()
            if -1 in s:
                s = set()
        endTime = int(round(time.time() * 1000))
        # if counter > 3:
        #    print 'not found'
        #    return None
        counter += 1
    print ''
    oce = [gk.OCE(num2CG(x, n), g) for x in s]
    cum_oce = [sum(x['directed']) + sum(x['bidirected']) for x in oce]
    idx = np.argmin(cum_oce)
    print "{:2}: {:8} : {:4}  {:10} seconds".\
          format(fold, round(dens, 3), cum_oce[idx],
                 round((endTime - startTime) / 1000, 3))
    # np.set_printoptions(formatter={'float': lambda x: format(x, '6.3f')+", "})
    # pprint.pprint(r['transition'].round(2))
    # np.set_printoptions()

    return {'gt': r,
            'eq': s,
            'OCE': oce[idx],
            'tries_till_found': counter,
            'estimate': g2,
            'graphs_tried': counter,
            'strength': sst + 0.01,
            'ms': endTime - startTime}


def wrapgen(fold, n=10, dens=0.1):
    scipy.random.seed()
    rate = 2

    s = set()
    sst = 0.06
    r = None
    while not r:
        r = timeout(lm.getAring, args=(n, dens, sst, False),
                    timeout_duration=3)
        print sst,
        if sst < 0.03:
            sst -= 0.002
        else:
            sst -= 0.01
        if sst < 0:
            break
    print 'model ' + str(fold) + ' found \n' + str(r['transition'].round(2))
    sys.stdout.flush()
    return r

if __name__ == '__main__':

    print 'processes: ', PNUM, INPNUM

    densities = {6: [0.25, 0.3, 0.35],
                 8: [.15, .2, 0.25, 0.3],
                 10: [0.3],
                 15: [0.1],
                 20: [0.1],
                 25: [0.1],
                 30: [0.1],
                 35: [0.1]}

    wrp = wrapper

    for nodes in [10]:
        z = {}
        pool = Pool(processes=PNUM)
        for dens in densities[nodes]:
            print "{:2}: {:8} : {:10}  {:10}".format('id', 'density', 'OCE', 'time')

            if PARALLEL:
                errors = pool.map(functools.partial(wrp, n=nodes,
                                                    dens=dens),
                                  range(REPEATS))
                print 'done'
            else:
                errors = []
                for i in range(REPEATS):
                    errors.append(wrp(i, n=nodes, dens=dens))
            print 'computed'
            z[dens] = errors
            zkl.save(z[dens],
                     socket.gethostname().split('.')[0] + '_nodes_' + str(nodes) + '_samples_' + str(SAMPLESIZE) + '_density_' + str(dens) + '_noise_' + NOISE_STD + '_OCE_b_' + EST + '_' + DIST + POSTFIX + '.zkl')
            print ''
            print '----'
            print ''
        pool.close()
        pool.join()
        zkl.save(z, socket.gethostname().split('.')[0] + '_nodes_' + str(nodes) + '_samples_' + str(
            SAMPLESIZE) + '_noise_' + NOISE_STD + '_OCE_b_' + EST + '_' + DIST + POSTFIX + '.zkl')
