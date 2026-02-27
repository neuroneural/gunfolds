import functools
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.utils import bfutils
import gunfolds.utils.graphkit as gk
from gunfolds.solvers import traversal
import gunfolds.solvers.unknownrate as ur
import gunfolds.utils.zickle as zkl
from multiprocessing import Pool, Process, Queue, current_process
import scipy
import socket
import time

UMAX = 6
INPNUM = 1  # number of randomized starts per graph
CAPSIZE = 10000  # stop traversing after growing equivalence class tothis size
REPEATS = 100
PNUM = get_process_count(INPNUM)


def wrapper_rate_agnostic(fold, n=10, k=10):
    scipy.random.seed()
    l = {}
    while True:
        try:
            g = gk.ringmore(n, k)  # random ring of given density
            gs = bfutils.call_undersamples(g)
            for u in range(1, min([len(gs), UMAX])):
                g2 = bfutils.undersample(g, u)
                print fold, ': ', traversal.density(g), ':',
                startTime = int(round(time.time() * 1000))
                s = ur.iteqclass(g2, verbose=False)
                endTime = int(round(time.time() * 1000))
                print len(s)
                l[u] = {'eq': s, 'ms': endTime - startTime}
        except MemoryError:
            print 'memory error... retrying'
            continue
        break
    return {'gt': g, 'solutions': l}


def killall(l):
    for e in l:
        e.join(timeout=0.001)
        if not e.is_alive():
            # print 'first result'
            for p in l:
                if p != e:
                    # print 'terminating ', p.name
                    p.terminate()
                    p.join()
                else:
                    p.join()
            return True
    return False


def fan_wrapper(fold, n=10, k=10):
    scipy.random.seed()
    curr_proc = current_process()
    curr_proc.daemon = False
    output = Queue()
    while True:
        try:
            g = gk.ringmore(n, k)
            gdens = traversal.density(g)
            g2 = bfutils.increment_u(g, g)
            # g2 = bfutils.undersample(g,2)

            def inside_wrapper():
                scipy.random.seed()
                try:
                    startTime = int(round(time.time() * 1000))
                    s = traversal.v2g22g1(g2, capsize=CAPSIZE)
                    # s = traversal.backtrack_more2(g2, rate=2, capsize=CAPSIZE)
                    endTime = int(round(time.time() * 1000))
                    print "{:2}: {:8} : {:4}  {:10} seconds".\
                        format(fold, round(gdens, 3), len(s),
                               round((endTime - startTime) / 1000., 3))
                    output.put({'gt': g, 'eq': s, 'ms': endTime - startTime})
                except MemoryError:
                    print 'memory error...'
                    raise
            pl = [Process(target=inside_wrapper) for x in range(INPNUM)]
            for e in pl:
                e.start()
            while True:
                if killall(pl):
                    break
            r = output.get()
        except MemoryError:
            print 'memory error... retrying'
            for p in pl:
                p.terminate()
                p.join()
            continue
        break
    for p in pl:
        p.join()
    return r

if __name__ == '__main__':
    print 'processes: ', PNUM, INPNUM
    densities = {6: [0.2, 0.25, 0.3, 0.35],
                 8: [0.3],
                 10: [0.1],  # 0.15, 0.2, 0.25, 0.3],
                 15: [0.25, 0.3],
                 20: [0.1],  # 0.15, 0.2, 0.25, 0.3],
                 25: [0.1],
                 30: [0.1],
                 35: [0.1],
                 40: [0.1],
                 50: [0.05, 0.1],
                 60: [0.05, 0.1]}

    for nodes in [15]:
        z = {}
        pool = Pool(processes=PNUM)
        for dens in densities[nodes]:
            print "{:2}: {:8} : {:10}  {:10}".format('id', 'density', 'eq class', 'time')
            e = bfutils.dens2edgenum(dens, n=nodes)
            eqclasses = pool.map(functools.partial(fan_wrapper, n=nodes, k=e),
                                 range(REPEATS))
            z[dens] = eqclasses
            zkl.save(z[dens],
                     socket.gethostname().split('.')[0] +
                     '_nodes_' + str(nodes) + '_density_' + str(dens) + '_newp_.zkl')
            print ''
            print '----'
            print ''
        pool.close()
        pool.join()
        zkl.save(z, socket.gethostname().split('.')[0] + '_nodes_' + str(nodes) + '_newp_.zkl')
