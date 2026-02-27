from gunfolds.scripts.stackedbars import get_counts
from gunfolds.utils import zickle as zkl
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pylab as pl
import seaborn as sns


def gettimes(d):
    t = [x['ms'] for x in d]
    time = map(lambda x: x / 1000. / 60., t)
    return time


if __name__ == '__main__':

    SBDIR = '~/soft/src/dev/tools/stackedBarGraph/'
    import sys
    import os
    sys.path.append(os.path.expanduser(SBDIR))
    from stackedBarGraph import StackedBarGrapher
    SBG = StackedBarGrapher()

    l = [(0.15, 'leibnitz_nodes_15_density_0.1_newp_.zkl'),
         (0.20, 'leibnitz_nodes_20_density_0.1_newp_.zkl'),
         (0.25, 'leibnitz_nodes_25_density_0.1_newp_.zkl'),
         (0.30, 'leibnitz_nodes_30_density_0.1_newp_.zkl'),
         (0.35, 'leibnitz_nodes_35_density_0.1_newp_.zkl')]

    fig = pl.figure(figsize=[10, 3])
    # Read in data & create total column

    d = zkl.load("hooke_nodes_6_g32g1_.zkl")  # hooke_nodes_35_newp_.zkl")
    densities = [.15, .20, .25, .30, .35]
    d = {}
    for fname in l:
        d[fname[0]] = zkl.load(fname[1])

    # unique size
    usz = set()
    dc = {}
    for u in densities:
        dc[u] = get_counts(d[u])
        for v in dc[u]:
            usz.add(v)

    for u in densities:
        for c in usz:
            if not c in dc[u]:
                dc[u][c] = 0

    A = []
    for u in densities:
        A.append([dc[u][x] for x in np.sort(dc[u].keys())])

    # print A
    # A = np.array(A)
    pp = mpl.colors.LinearSegmentedColormap.from_list("t", sns.color_palette("Paired", len(usz)))
    # pp = mpl.colors.LinearSegmentedColormap.from_list("t",sns.dark_palette("#5178C7",len(usz)))
    # pp =
    # mpl.colors.LinearSegmentedColormap.from_list("t",sns.blend_palette(["mediumseagreen",
    # "ghostwhite", "#4168B7"],len(usz)))
    scalarMap = mpl.cm.ScalarMappable(norm=lambda x: x / np.double(len(usz)),
                                      cmap=pp)

    d_widths = [.5] * len(densities)
    d_labels = map(lambda x: str(int(x * 100)) + "%", densities)
    # u = np.sort(list(usz))
    d_colors = [scalarMap.to_rgba(i) for i in range(len(A[0]))]
    # d_colors = ['#2166ac', '#fee090', '#fdbb84', '#fc8d59', '#e34a33',
    # '#b30000', '#777777','#2166ac', '#fee090', '#fdbb84', '#fc8d59',
    # '#e34a33', '#b30000', '#777777','#2166ac', '#fee090']

    # ax = fig.add_subplot(211)
    ax = plt.subplot2grid((3, 1), (0, 0), rowspan=2)

    SBG.stackedBarPlot(ax,
                       A,
                       d_colors,
                       xLabels=d_labels,
                       yTicks=3,
                       widths=d_widths,
                       gap=0.005,
                       scale=False
                       )

    for i in range(len(A)):
        Ai = [x for x in A[i] if x > 0]
        y = [x / 2.0 for x in Ai]
        for j in range(len(Ai)):
            if j > 0:
                yy = y[j] + np.sum(Ai[0:j])
            else:
                yy = y[j]
            pl.text(0.5 * i - 0.02, yy - 1.2, str(Ai[j]), fontsize=12, zorder=10)

    # Set general plot properties
    # sns.set_style("white")
    # sns.set_context({"figure.figsize": (24, 10)})

    # for i in np.sort(list(usz))[::-1]:
    #     y = [100-dc[u][i] for u in np.sort(dc.keys())]
    #     bottom_plot=sns.barplot(x=np.asarray(densities)*100, y=y)
    # color=scalarMap.to_rgba(i))
    # y = (sbd[i+1]-sbd[i])/2.+sbd[i]scala
    # for j in range(len(sbd.Density)):
    # pl.text(j-0.1,y[j],'1',fontsize=16,zorder=i)

    # Optional code - Make plot look nicer
    sns.despine(left=True)
    # Set fonts to consistent 16pt size
    ax.set(xticklabels="")
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 # ax.get_xticklabels() +
                 ax.get_yticklabels()):
        item.set_fontsize(12)

    alltimes_new = []
    for fname in l:
        dp = zkl.load(fname[1])
        alltimes_new.append(gettimes(dp))

    shift = 0.15
    wds = 0.3
    fliersz = 2
    lwd = 1

    ax = plt.subplot2grid((3, 1), (2, 0))
    g = sb.boxplot(alltimes_new, names=map(lambda x: str(int(x * 100)) + "",
                                           densities),
                   widths=wds, color="Reds", fliersize=fliersz,
                   linewidth=lwd,
                   **{'positions': np.arange(len(densities)) + shift,
                      'label': 'MSL'})

    # plt.plot(np.arange(len(densities))-shift,
    #         map(np.median,alltimes_old), 'ro-', lw=0.5, mec='k')
    # plt.plot(np.arange(len(densities))+shift,
    #         map(np.median,alltimes_new), 'bo-', lw=0.5, mec='k')
    g.figure.get_axes()[1].set_yscale('log')
    plt.xlabel('number of nodes in a graph')
    plt.ylabel('computation time\n(minutes)')
    # plt.title('100 6 node graphs per density\n$G_2 \\rightarrow G_1$',
    #          multialignment='center')
    # plt.subplots_adjust(right=0.99, left=0.2)
    plt.legend(loc=0)
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() +
                 ax.get_yticklabels()):
        item.set_fontsize(12)

    pl.subplots_adjust(bottom=0.1, hspace=0.01, top=0.98)
    # plt.show()

    pl.show()
