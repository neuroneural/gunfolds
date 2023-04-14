from gunfolds.utils.calc_procs import get_process_count
import os
from gunfolds.solvers import traversal
from gunfolds.utils import zickle as zkl
import gunfolds.solvers.unknownrate as ur
import time, socket, functools
import scipy
import argparse
import multiprocessing.pool
import timeout_decorator
from timeout_decorator import TimeoutError

TIMEOUT=86400 # seconds = one day
POSTFIX='RASL_compare'
UMAX = 4
INPNUM = 1 # number of randomized starts per graph
REPEATS = 100
PNUM = get_process_count(INPNUM)
if PNUM>45:
    PNUM = 45

parser = argparse.ArgumentParser(description='Run settings.')
parser.add_argument("-c", "--CAPSIZE",default=100, help="stop traversing after growing equivalence class to this size.", type=int)
parser.add_argument("-n", "--NODE", default=6,help="number of nodes.", type=int)
parser.add_argument("-d", "--DENITIES",default="0.2,0.25,0.3", help="densities to be ran. e.g. -d=0.2,0.25,0.3", type=str)
parser.add_argument("-b", "--BATCH", help="slurm batch.", type=int)
parser.add_argument("-a", "--ARRAY", help="number of total batches.", type=int)
args = parser.parse_args()
# dens_list = args.DENITIES.split(',')
# dens_list = [float(item) for item in dens_list]


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False
    def _set_daemon(self, value):
        pass
    daemon = property(_get_daemon, _set_daemon)

class MyPool(multiprocessing.pool.Pool):
    Process = NoDaemonProcess

@timeout_decorator.timeout(TIMEOUT, use_signals=False)
def ra_caller(g2, fold):
    startTime = int(round(time.time() * 1000))
    s = ur.liteqclass(g2, verbose=True, capsize=args.CAPSIZE)
    endTime = int(round(time.time() * 1000))
    ra_time = endTime-startTime
    return s, ra_time

def ra_wrapper(fold, graphs):
    scipy.random.seed()
    l = {}
    while True:
        try:
            g = graphs[fold]['gt']
            g2 = graphs[fold]['gn']
            print (fold, ': ', traversal.density(g), '\n',)
            try:
                s, ra_time = ra_caller(g2, fold)
            except TimeoutError:
                s = None
                ra_time = None
            if s is not None:
                print (len(s))
            else:
                print ('timeout')
            l = {'eq': s, 'ms': ra_time}
        except MemoryError:
            print ('memory error... retrying')
            continue
        break
    return {'gt': g, 'solutions': l,'u' : graphs[fold]['u'],'dens':graphs[fold]['dens']}



'''densities = {3: dens_list,
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
                 60: dens_list}'''

for nodes in [args.NODE]: #np.sort(densities.keys())[0:]:
    print (nodes, ': ----')
    print ('')

    all_graphs = zkl.load('graphs.zkl')
    B = len(all_graphs) / args.ARRAY
    if PNUM > B:
        PNUM = B
    batch_graph = all_graphs[(int(args.BATCH) - 1) * B:(int(args.BATCH)) * B]
    pool = MyPool(processes=PNUM, maxtasksperchild=1)
    x = []
    eqclasses2 = pool.map(functools.partial(ra_wrapper, graphs=batch_graph),
                          range(len(batch_graph)))
    x = eqclasses2
    if not os.path.exists('res_CAP_' + str(args.CAPSIZE) + '_'):
        os.makedirs('res_CAP_' + str(args.CAPSIZE) + '_')
    zkl.save(x,'res_CAP_' + str(args.CAPSIZE) + '_/' + \
             socket.gethostname().split('.')[0] + \
             '_nodes_' + str(nodes) + '_batch_' + str(args.BATCH) + '_' + \
             POSTFIX + '_pyRASL_CAPSIZE_' + str(args.CAPSIZE) + '_.zkl')

    print ('')
    print ('----')
    print ('\n')
    pool.close()
    pool.join()
