from gunfolds.utils.calc_procs import get_process_count
from gunfolds.utils import bfutils
import gunfolds.utils.graphkit as gk
from gunfolds.solvers import traversal
import gunfolds.solvers.unknownrate as ur
import gunfolds.utils.zickle as zkl
import multiprocessing.pool

class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess
import functools
import time, os
import socket
import scipy
import timeout_decorator
from timeout_decorator import TimeoutError
import argparse
TIMEOUT=24*3600 # seconds timeout
KEY = 'pyRASL'
UMAX = 4
INPNUM = 1  # numbersss of randomized starts per graph
REPEATS = 100
PNUM = int(get_process_count(INPNUM))

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-c", "--CAPSIZE",default=1000, help="stop traversing after growing equivalence class to this size.", type=int)
parser.add_argument("-n", "--NODE", default=6,help="number of nodes.", type=int)
parser.add_argument("-d", "--DENITIES",default="0.2,0.25,0.3", help="densities to be ran. e.g. -d=0.2,0.25,0.3", type=str)
args = parser.parse_args()
dens_list = args.DENITIES.split(',')
dens_list = [float(item) for item in dens_list]

@timeout_decorator.timeout(TIMEOUT, use_signals=False)
def ra_caller(g2, fold):
    startTime = int(round(time.time() * 1000))
    s = ur.liteqclass(g2, verbose=True, capsize=args.CAPSIZE)
    endTime = int(round(time.time() * 1000))
    ra_time = endTime-startTime
    return s, ra_time

def ra_wrapper(fold, n=10, k=10):
    scipy.random.seed()
    l = {}
    while True:
        try:
            g = gk.ringmore(n, k)  # random ring of given density
            gs = bfutils.call_undersamples(g)
            for u in range(1, min([len(gs), UMAX])):
                g2 = bfutils.undersample(g, u)
                print (fold, ': ', traversal.density(g), '\n',)
                try:
                    s, ra_time = ra_caller(g2, fold)
                except TimeoutError:
                    s = None
                    ra_time = None
                if s is not None:
                    print (len(s), u)
                else:
                    print ('timeout')
                l[u] = {'eq': s, 'ms': ra_time}
        except MemoryError:
            print ('memory error... retrying')
            continue
        break
    return {'gt': g, 'solutions': l}


if __name__ == '__main__':
    print ('processes: ', PNUM, INPNUM)

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

    for nodes in [args.NODE]:
        z = {}
        pool= MyPool(processes=PNUM, maxtasksperchild=1)
        for dens in densities[nodes]:
            print ("{:2}: {:8} : {:10}  {:10}".format('id', 'density', 'eq class', 'time'))
            zkl.save(10,
                     socket.gethostname().split('.')[0] + \
                     '_nodes_' + str(nodes) + '_density_' + \
                     str(dens) + '_test_' + KEY + '_.zkl')
            e = bfutils.dens2edgenum(dens, n=nodes)
            eqclasses =  pool.map(functools.partial(ra_wrapper, n=nodes, k=e),range(REPEATS))
            z[dens] = eqclasses
            if not os.path.exists('res_CAP_' + str(args.CAPSIZE) + '_'):
                os.makedirs('res_CAP_' + str(args.CAPSIZE) + '_')
            zkl.save(z[dens],
                     'res_CAP_'+str(args.CAPSIZE) + '_/'+socket.gethostname().split('.')[0]+\
                     '_nodes_'+str(nodes)+'_density_'+\
                     str(dens)+'_CAPSIZE_'+str(args.CAPSIZE)+'_'+'_'+KEY+'_.zkl')

            print ('')
            print ('----')
            print ('\n')
        pool.close()
        pool.join()
        # zkl.save(z,socket.gethostname().split('.')[0]+'_nodes_'+str(nodes)+'_'+KEY+'_.zkl')
        # for dens in densities[nodes]:
        #     os.remove(socket.gethostname().split('.')[0]+\
        #               '_nodes_'+str(nodes)+'_density_'+str(dens)+'_'+KEY+'_.zkl')
