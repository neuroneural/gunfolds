from gunfolds.utils import zickle as zkl
from matplotlib import pyplot as plt
import numpy as np
from os import listdir
import pandas as pd
import seaborn as sns
from matplotlib import ticker as mticker
plt.style.use('seaborn-whitegrid')

l = listdir('./res_CAP_100_/8 nodes')
l.sort()

l_new = listdir('./res_CAP_10000_/8 nodes')
l_new.sort()
if __name__ == '__main__':
    df = pd.DataFrame()
    count = -1
    group = []
    EQ = []
    CAP = []
    count2 = -1
    for fname in l:
        count += 1
        count2 += 1
        d = zkl.load('./res_CAP_100_/8 nodes/'+fname)
        for i in range(len(d)):
            item = d[i]
            sol = item['solutions']
            EQsize = len(sol['eq'])
            EQ.append(np.log10(EQsize))
            group.append(count)
            CAP.append("100")

    count = -1
    for fname in l_new:
        count += 1
        d = zkl.load('./res_CAP_10000_/8 nodes/' + fname)
        for i in range(len(d)):
            item = d[i]
            sol = item['solutions']
            EQsize = len(sol['eq'])
            EQ.append(np.log10(EQsize))
            group.append(count)
            CAP.append("10k")

    df["group"] = group
    df["eq"] = EQ
    df["CAP"] = CAP
    ax = sns.violinplot(x="group", y="eq", hue="CAP", data=df, palette="muted",split=True)
    plt.title(' Equal class for Clingo RASL for 8 nodes and CAPSIZE 10k Vs. 100')
    plt.xlabel('density 0.2 [u=2,u=3,u=4],density 0.25 [u=2,u=3,u=4],density 0.3 [u=2,u=3,u=4]')
    plt.ylabel('size of Equal class')
    ax.yaxis.set_major_formatter(mticker.StrMethodFormatter("$10^{{{x:.0f}}}$"))
    ax.yaxis.set_ticks([np.log10(x) for p in range(-2, 1) for x in np.linspace(10 ** p, 10 ** (p + 1), 10)],minor=True)
    plt.legend(loc="upper left")
    plt.savefig("node_8_CAP_compare.svg")