from gunfolds.utils import zickle as zkl
from gunfolds.utils import bfutils
from matplotlib import pyplot as plt
import numpy as np
from gunfolds.utils.graphkit import density
from os import listdir
CAPSIZE = 10000

density_dict = [0.2,0.25,0.3]
degree_lsit = [0.9,2,3,5]
undersampling_dict = [2,3,4]

l_sccs = listdir('./res_CAP_'+str(CAPSIZE)+'_/50 nodes')
l_sccs.sort()
if l_sccs[0].startswith('.'):
     l_sccs.pop(0)
l_sccs = [int(item) for item in l_sccs]
l_sccs.sort()
size_list = []
for size in l_sccs:
    l = listdir('./res_CAP_'+str(CAPSIZE)+'_/50 nodes/'+str(size))
    l.sort()
    if l[0].startswith('.'):
        l.pop(0)
    gis = [zkl.load('./res_CAP_'+str(CAPSIZE)+'_/50 nodes/'+str(size)+'/'+item) for item in l]
    size_list.append([item for sublist in gis for item in sublist])


if __name__ == '__main__':
    all_ys_c = [[[] for i in range(len(undersampling_dict)*len(l_sccs))] for j in range(len(l_sccs))]
    all_xs_c = [[[] for i in range(len(undersampling_dict)*len(l_sccs))] for j in range(len(l_sccs))]

    index = -1
    for l in size_list:
        index += 1
        for item in l:
            y = undersampling_dict.index(item['u'])
            z =  (3*index)+y
            sol = item['solutions']
            check = sol['ms']
            if not sol['ms'] == None:
                all_ys_c[index][z].append((float(sol['ms'])/(1000*60)))
            H = bfutils.undersample(item['gt'], item['u'])
            dens = (density(H))
            all_xs_c[index][z].append((z + 1) + (1.5)*dens)

        x = [len(all_ys_c[index][i]) for i in range(len(all_ys_c[index]))]
        for i in range(len(all_ys_c[index])):
            all_ys_c[index][i].extend([0 for j in range(max(x) - len(all_ys_c[index][i]))])
            all_xs_c[index][i].extend([0 for j in range(max(x) - len(all_xs_c[index][i]))])

    all_xs_c = np.asarray(all_xs_c)
    all_ys_c = np.asarray(all_ys_c)



    for i in range(len(l_sccs)):
        x = all_xs_c[i].reshape(1200)
        y = all_ys_c[i].reshape(1200)
        plt.plot(x, y, 'o', alpha=0.3,label=str(int(50/l_sccs[i]))+' SCCs each consisting of'+str(l_sccs[i])+ 'nodes')

    plt.title(' comparison for SCC size in d_rasl with scc = True')
    plt.xlabel(' for each color [u=2,u=3,u=4]')
    plt.ylabel('minutes')
    plt.yscale('log')
    plt.legend()
    plt.savefig("node_50_SCC_size_compare.svg")
    plt.show()
