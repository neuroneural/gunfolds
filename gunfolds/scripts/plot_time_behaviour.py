from gunfolds.utils import zickle as zkl
from gunfolds.utils import bfutils
from matplotlib import pyplot as plt
plt.style.use('seaborn-whitegrid')
import numpy as np
from gunfolds.utils.graphkit import density
from os import listdir
CAPSIZE = 10000

density_dict = [0.2,0.25,0.3]
degree_lsit = [0.9,2,3,5]
undersampling_dict = [2,3,4]
# l_py = listdir('./res_CAP_'+str(CAPSIZE)+'_/pyRASL')
# l_py.sort()
# if l_py[0].startswith('.'):
#     l_py.pop(0)
# gis = [zkl.load('./res_CAP_'+str(CAPSIZE)+'_/pyRASL/'+item) for item in l_py]
# py_list = [item for sublist in gis for item in sublist]

'''#To load from pickle file
data = []
with open(filename, 'rb') as fr:
    try:
        while True:
            data.append(pickle.load(fr))
    except EOFError:
        pass'''

l_cRASL = listdir('./res_CAP_'+str(CAPSIZE)+'_/10 nodes')
l_cRASL.sort()
if l_cRASL[0].startswith('.'):
    l_cRASL.pop(0)
gis = [zkl.load('./res_CAP_'+str(CAPSIZE)+'_/10 nodes/'+item) for item in l_cRASL]
c_list = [item for sublist in gis for item in sublist]

l_mslRASL = listdir('./res_CAP_'+str(CAPSIZE)+'_/20 nodes')
l_mslRASL.sort()
if l_mslRASL[0].startswith('.'):
    l_mslRASL.pop(0)
gis = [zkl.load('./res_CAP_'+str(CAPSIZE)+'_/20 nodes/'+item) for item in l_mslRASL]
msl_list = [item for sublist in gis for item in sublist]

l_dRASL = listdir('./res_CAP_'+str(CAPSIZE)+'_/30 nodes')
l_dRASL.sort()
if l_dRASL[0].startswith('.'):
    l_dRASL.pop(0)
gis = [zkl.load('./res_CAP_'+str(CAPSIZE)+'_/30 nodes/'+item) for item in l_dRASL]
d_list = [item for sublist in gis for item in sublist]

if __name__ == '__main__':
    # all_ys_py = [[] for i in range(len(density_dict)*len(undersampling_dict))]
    # all_xs_py = [[] for i in range(len(density_dict)*len(undersampling_dict))]
    all_ys_c = [[] for i in range(len(degree_lsit)*len(undersampling_dict))]
    all_xs_c = [[] for i in range(len(degree_lsit)*len(undersampling_dict))]
    all_ys_msl = [[] for i in range(len(degree_lsit)*len(undersampling_dict))]
    all_xs_msl = [[] for i in range(len(degree_lsit)*len(undersampling_dict))]
    all_ys_d = [[] for i in range(len(degree_lsit) * len(undersampling_dict))]
    all_xs_d = [[] for i in range(len(degree_lsit) * len(undersampling_dict))]
    # for item in py_list:
    #     x = density_dict.index(item['dens'])
    #     y = undersampling_dict.index(item['u'])
    #     z = (3*x)+y
    #     sol = item['solutions']
    #     check = sol['ms']
    #     if not sol['ms']== None:
    #         all_ys_py[z].append((float(sol['ms'])/(1000*60)))
    #     H = bfutils.undersample(item['gt'], item['u'])
    #     dens=(density(H))*(1.1)
    #     all_xs_py[z].append((z+1)+ dens)
    #
    # all_xs_py = np.asarray(all_xs_py)
    # all_ys_py = np.asarray(all_ys_py)

    for item in c_list:
        x = degree_lsit.index(item['deg'])
        y = undersampling_dict.index(item['u'])
        z = (3 * x) + y
        sol = item['solutions']
        check = sol['ms']
        if not sol['ms'] == None:
            all_ys_c[z].append((float(sol['ms'])/(1000*60)))
        H = bfutils.undersample(item['gt'], item['u'])
        dens = (density(H))
        all_xs_c[z].append((z + 1) + dens)

    x = [len(all_ys_c[i]) for i in range(len(all_ys_c))]
    for i in range(len(all_ys_c)):
        all_ys_c[i].extend([0 for j in range(max(x) - len(all_ys_c[i]))])
        all_xs_c[i].extend([0 for j in range(max(x) - len(all_xs_c[i]))])
    all_xs_c = np.asarray(all_xs_c)
    all_ys_c = np.asarray(all_ys_c)

    for item in msl_list:
        x = degree_lsit.index(item['deg'])
        y = undersampling_dict.index(item['u'])
        z = (3 * x) + y
        sol = item['solutions']
        check = sol['ms']
        if not sol['ms'] == None:
            all_ys_msl[z].append((float(sol['ms'])/(1000*60)))
        H = bfutils.undersample(item['gt'], item['u'])
        dens = (density(H))
        all_xs_msl[z].append((z + 1) + dens)

    x = [len(all_ys_msl[i]) for i in range(len(all_ys_msl))]
    for i in range(len(all_ys_c)):
        all_ys_msl[i].extend([0 for j in range(max(x) - len(all_ys_msl[i]))])
        all_xs_msl[i].extend([0 for j in range(max(x) - len(all_xs_msl[i]))])
    all_xs_msl = np.asarray(all_xs_msl)
    all_ys_msl = np.asarray(all_ys_msl)

    for item in d_list:
        x = degree_lsit.index(item['deg'])
        y = undersampling_dict.index(item['u'])
        z = (3 * x) + y
        sol = item['solutions']
        check = sol['ms']
        if not sol['ms'] == None:
            all_ys_d[z].append((float(sol['ms'])/(1000*60)))
        H = bfutils.undersample(item['gt'], item['u'])
        dens = (density(H))
        all_xs_d[z].append((z + 1) + dens)

    x = [len(all_ys_d[i]) for i in range(len(all_ys_d))]
    for i in range(len(all_ys_c)):
        all_ys_d[i].extend([0 for j in range(max(x) - len(all_ys_d[i]))])
        all_xs_d[i].extend([0 for j in range(max(x) - len(all_xs_d[i]))])
    all_xs_d = np.asarray(all_xs_d)
    all_ys_d = np.asarray(all_ys_d)

    # plt.plot(all_xs_py, all_ys_py, 'o', color='blue' ,alpha=0.5)
plt.plot(all_xs_c.flatten(), all_ys_c.flatten(), 'o', color='red', alpha=0.2,label='10 nodes')
plt.plot(all_xs_msl.flatten(), all_ys_msl.flatten(), 'o', color='yellow', alpha=0.2,label='20 nodes')
plt.plot(all_xs_d.flatten(), all_ys_d.flatten(), 'o', color='green', alpha=0.2,label='30 nodes')
plt.title(' Clingo d_rasl for 10,20,30 nodes. CAPSIZE = 10k, TIMEOUT=one day\n d_rasl has two inputs on bp_mean')
plt.xlabel('degree 0.9 [u=2,u=3,u=4],degree 2 [u=2,u=3,u=4], \n'
           'degree 3 [u=2,u=3,u=4],degree 5 [u=2,u=3,u=4]')
plt.ylabel('minutes')
plt.yscale('log')
plt.grid(which='both',axis = 'y')
plt.legend()
# plt.savefig("node_10_20_30_drasl_compare.svg")
plt.show()
