import gunfolds.utils.zickle as zkl
import matplotlib as mpl
from matplotlib import pyplot as plt
import numpy as np
import pylab as pl
import seaborn as sb



def gettimes(d):
    t = [x['ms'] for x in d]
    time = map(lambda x: x / 1000. / 60., t)
    return time

def get_counts(d, rate=1):
    eqc = [len(x['eq']) for x in d if x['rate'] == rate]
    keys = np.sort(np.unique(eqc))
    c = {}
    for k in keys:
        c[k] = len(np.where(eqc == k)[0])
    return c

if __name__ == '__main__':

    SBDIR = '~/soft/src/dev/tools/stackedBarGraph/'
    import sys
    import os
    sys.path.append(os.path.expanduser(SBDIR))
    from stackedBarGraph import StackedBarGrapher
    SBG = StackedBarGrapher()

    l = [(0.15, 'leibnitz_nodes_10_density_0.15_rasl_multicore_.zkl'),
         (0.20, 'leibnitz_nodes_10_density_0.15_rasl_multicore_.zkl'),         
         (0.25, 'leibnitz_nodes_10_density_0.15_rasl_multicore_.zkl')]

    fig = pl.figure(figsize=[4.6/3, 2.25])
    # Read in data & create total column

    densities = [0.15]
    d = {}
    for fname in l:
        d[fname[0]] = zkl.load(fname[1])

    def sbars(densities, rate=1):
        # unique size
        usz = set()
        dc = {}
        for u in densities:
            dc[u] = get_counts(d[u],rate=rate)
            for v in dc[u]:
                usz.add(v)

        for u in densities:
            for c in usz:
                if not c in dc[u]:
                    dc[u][c] = 0

        A = []
        for u in densities:
            A.append([dc[u][x] for x in np.sort(dc[u].keys())])
        A = np.asarray(A)
        A = (100*np.dot(np.diag(1./np.sum(A,axis=1)),A)).astype('int')
        A[:,0] += 100-np.sum(A,axis=1)
        return A,usz

    A, usz = sbars(densities, rate=1)
    A = np.asarray([[100,0]])
    usz = set([1,0])
    # print A
    # A = np.array(A)
    pp = mpl.colors.LinearSegmentedColormap.from_list("t", sb.color_palette("Paired", len(usz)))
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
    ax = plt.subplot2grid((4, 1), (0, 0), rowspan=2)

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


    ax = plt.subplot2grid((4, 1), (2, 0),rowspan=2)
    A, usz = sbars(densities, rate=2)
    pp = mpl.colors.LinearSegmentedColormap.from_list("t", sb.color_palette("Paired", len(usz)))
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

    pl.subplots_adjust(bottom=0.1, hspace=0.01, top=0.98)
    # plt.show()

    pl.show()
