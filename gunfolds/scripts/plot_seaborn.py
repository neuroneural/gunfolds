import pickle
from gunfolds.utils import bfutils
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from gunfolds.utils.graphkit import density
from os import listdir
from matplotlib.ticker import MultipleLocator

CAPSIZE = 10000
density_dict = [0.2,0.25,0.3]
degree_lsit = [0.9,2,3,5]
undersampling_dict = [2,3,4]


l1 = listdir('./res_CAP_'+str(CAPSIZE)+'_/10 nodes/drasl/FT/zkl')
l1.sort()
if l1[0].startswith('.'):
    l1.pop(0)
list1 = []
for name in l1:
 with open('./res_CAP_'+str(CAPSIZE)+'_/10 nodes/drasl/FT/zkl/'+name, 'rb') as fr:
     try:
         while True:
             list1.append(pickle.load(fr))
     except EOFError:
         pass

l2 = listdir('./res_CAP_'+str(CAPSIZE)+'_/10 nodes/drasl/TF/zkl')
l2.sort()
if l2[0].startswith('.'):
    l2.pop(0)
list2 = []
for name in l2:
 with open('./res_CAP_'+str(CAPSIZE)+'_/10 nodes/drasl/TF/zkl/'+name, 'rb') as fr:
     try:
         while True:
             list2.append(pickle.load(fr))
     except EOFError:
         pass

l3 = listdir('./res_CAP_'+str(CAPSIZE)+'_/10 nodes/drasl/TT/zkl')
l3.sort()
if l3[0].startswith('.'):
    l3.pop(0)
list3 = []
for name in l3:
 with open('./res_CAP_'+str(CAPSIZE)+'_/10 nodes/drasl/TT/zkl/'+name, 'rb') as fr:
     try:
         while True:
             list3.append(pickle.load(fr))
     except EOFError:
         pass


if __name__ == '__main__':
    df = pd.DataFrame()
    group = []
    EQ = []
    deg = []
    den1 = []
    denH = []
    time = []
    node =[]
    method = []
    u = []

    for item in list1:
        x = degree_lsit.index(item['deg'])
        y = undersampling_dict.index(item['u'])
        z = (3 * x) + y
        group.append(z)

        den1.append(item['dens'])
        deg.append(item['deg'])
        node.append(10)
        sol = item['solutions']
        EQsize = sol['eq_size']
        method.append('FT')
        EQ.append(EQsize)
        times = sol['ms']
        if not times == None:
            time.append((float(times)/(1000*60)))
        else:
            time.append(0)
        H = bfutils.undersample(item['gt'], item['u'])
        denH.append(density(H))
        u.append(item['u'] + density(H))


    for item in list2:
        x = degree_lsit.index(item['deg'])
        y = undersampling_dict.index(item['u'])
        z = (3 * x) + y
        group.append(z)
        den1.append(item['dens'])
        deg.append(item['deg'])
        node.append(10)
        sol = item['solutions']
        EQsize = sol['eq_size']
        method.append('TF')
        EQ.append(EQsize)
        times = sol['ms']
        if not times == None:
            time.append((float(times) / (1000 * 60)))
        else:
            time.append(0)
        H = bfutils.undersample(item['gt'], item['u'])
        denH.append(density(H))
        u.append(item['u'] + density(H))


    for item in list3:
        x = degree_lsit.index(item['deg'])
        y = undersampling_dict.index(item['u'])
        z = (3 * x) + y
        group.append(z)
        den1.append(item['dens'])
        deg.append(item['deg'])
        node.append(10)
        method.append('TT')
        sol = item['solutions']
        EQsize = sol['eq_size']
        EQ.append(EQsize)
        times = sol['ms']
        if not times == None:
            time.append((float(times) / (1000 * 60)))
        else:
            time.append(0)
        H = bfutils.undersample(item['gt'], item['u'])
        denH.append(density(H))
        u.append(item['u'] + density(H))


    df["group"] = group
    df["method"] = method
    df["eq"] = EQ
    df['ms'] = time
    df['deg'] = deg
    df['node'] = node
    df['den1'] = den1
    df['denH'] = denH
    df['u'] = u

    sns.set({"xtick.minor.size": 0.2})
    ticks = [0.1, 0.3, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000, 30000]
    labels = [i for i in ticks]
    g = sns.FacetGrid(df, col="deg",hue="method",palette='dark',height=10,aspect=0.2, margin_titles=True)
    g.map(sns.scatterplot, "u", "ms", s= 60,alpha=.3,x_jitter=.1)
    g.add_legend()
    g.set_axis_labels("undersampling","minutes")
    g.axes[0][0].xaxis.grid(True, "minor", linewidth=.5)
    g.axes[0][1].xaxis.grid(True, "minor", linewidth=.5)
    g.axes[0][2].xaxis.grid(True, "minor", linewidth=.5)
    g.axes[0][3].xaxis.grid(True, "minor", linewidth=.5)
    g.axes[0][0].xaxis.grid(True, "major", linewidth=3)
    g.axes[0][1].xaxis.grid(True, "major", linewidth=3)
    g.axes[0][2].xaxis.grid(True, "major", linewidth=3)
    g.axes[0][3].xaxis.grid(True, "major", linewidth=3)
    g.set(yscale ='log')
    g.axes[0][0].xaxis.set_major_locator(MultipleLocator(1))
    g.axes[0][0].xaxis.set_minor_locator(MultipleLocator(0.25))
    g.axes[0][1].xaxis.set_major_locator(MultipleLocator(1))
    g.axes[0][1].xaxis.set_minor_locator(MultipleLocator(0.25))
    g.axes[0][2].xaxis.set_major_locator(MultipleLocator(1))
    g.axes[0][2].xaxis.set_minor_locator(MultipleLocator(0.25))
    g.axes[0][3].xaxis.set_major_locator(MultipleLocator(1))
    g.axes[0][3].xaxis.set_minor_locator(MultipleLocator(0.25))
    plt.savefig("test.svg")
    # plt.show()
