from gunfolds.utils import zickle as zkl
import matplotlib as mpl
import numpy as np
import pylab as pl
import seaborn as sns


def get_counts(d):
    eqc = [len(x['eq']) for x in d]
    keys = np.sort(np.unique(eqc))
    c = {}
    for k in keys:
        c[k] = len(np.where(eqc == k)[0])
    return c
    

if __name__ == '__main__':

    import sys
    sys.path.append('/na/homes/splis/soft/src/dev/tools/stackedBarGraph/')
    from stackedBarGraph import StackedBarGrapher
    SBG = StackedBarGrapher()

    fig = pl.figure(figsize=[10, 1.3])
    # Read in data & create total column

    d = zkl.load("hooke_nodes_6_g32g1_.zkl")  # hooke_nodes_35_newp_.zkl")
    densities = np.sort(d.keys())

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

    ax = fig.add_subplot(111)
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
    for item in ([ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(12)
    pl.subplots_adjust(bottom=0.2)
    # plt.show()
    pl.show()
