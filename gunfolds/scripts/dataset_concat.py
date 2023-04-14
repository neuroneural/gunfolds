from os import listdir
from gunfolds.utils import zickle as zkl
import random
nodes =30

l_batches = listdir('./datasets2')
l_batches.sort()
if l_batches[0].startswith('.'):
    l_batches.pop(0)
gis = [zkl.load('./datasets2/'+item) for item in l_batches]
g_list = [item for sublist in gis for item in sublist]
shufflelist = sorted(g_list, key=lambda k: k['deg'])
random.shuffle(shufflelist)

zkl.save(shufflelist, 'datasets/graphs_node_' + str(nodes) +'_gcd1_.zkl')
