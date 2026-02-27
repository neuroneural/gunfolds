from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from gunfolds.utils import graphkit as gk
from gunfolds.utils import bfutils as bfu
import seaborn as sb

#

uplimit = 0.25

#

# d05 = zkl.load('leibnitz_nodes_5_samples_2000_noise_0.1_OCE_b_svar_beta_rasl_more.zkl')
# d05 = zkl.load('oranos_nodes_5_samples_2000_noise_0.1_OCE_b_svar_beta_rasl_more.zkl')
# d05 = zkl.load('leibnitz_nodes_6_samples_2000_noise_0.1_OCE_b_svar_beta_rasl.zkl')
# d05 = zkl.load('leibnitz_nodes_6_samples_2000_noise_0.1_OCE_b_svar_beta_rasl.zkl')


def estOE(d):
    gt = d['gt']['graph']
    gt = bfu.undersample(gt, 1)
    e = gk.OCE(d['estimate'], gt)
    N = np.double(len(gk.edgelist(gt))) +\
        np.double(len(gk.bedgelist(gt)))
    return (e['directed'][0] + e['bidirected'][0]) / N


def estCOE(d):
    gt = d['gt']['graph']
    gt = bfu.undersample(gt, 1)
    e = gk.OCE(d['estimate'], gt)
    n = len(gt)
    N = np.double(n ** 2 + (n - 1) ** 2 / 2.0
                  - len(gk.edgelist(gt))
                  - len(gk.bedgelist(gt)))
    return (e['directed'][1] + e['bidirected'][1]) / N


if __name__ == '__main__':
    d = d05  # d08_01
    density = np.sort(d.keys())
    n = len(d[density[0]][0]['gt']['graph'])
    OE = [[gk.oerror(x) for x in d[dd]] for dd in density]
    COE = [[gk.cerror(x) for x in d[dd]] for dd in density]

    eOE = [[estOE(x) for x in d[dd]] for dd in density]
    eCOE = [[estCOE(x) for x in d[dd]] for dd in density]

    samplesnum = 20
    denscol = [0.25] * samplesnum + [0.30] * samplesnum + [0.35] * samplesnum
    OE = OE[0] + OE[1] + OE[2]
    eOE = eOE[0] + eOE[1] + eOE[2]
    COE = COE[0] + COE[1] + COE[2]
    eCOE = eCOE[0] + eCOE[1] + eCOE[2]

    OE = pd.DataFrame(
        np.asarray([denscol + denscol, OE + eOE, pd.Categorical(['RASL'] * samplesnum * 3 + ['SVAR'] * samplesnum * 3)]).T, columns=['density', 'time', 'OE'])

    COE = pd.DataFrame(
        np.asarray([denscol + denscol, COE + eCOE, pd.Categorical(['RASL'] * samplesnum * 3 + ['SVAR'] * samplesnum * 3)]).T, columns=['density', 'time', 'COE'])

    shift = 0.15
    wds = 0.3
    fliersz = 2
    lwd = 1

    plt.figure(figsize=[14, 6])

    plt.subplot(121)
    ax = sb.boxplot(x="density", y="time", hue="OE",
                    data=OE,
                    palette="Set2",
                    linewidth=lwd,
                    width=wds,
                    fliersize=fliersz)
    sb.stripplot(x="density", y="time", hue="OE",
                 data=OE,
                 palette='Set2',
                 size=4, jitter=True, edgecolor="gray")
    # ax.figure.get_axes()[0].set_yscale('log')
    plt.xticks([0, 1, 2], ('25%', '30%', '35%'))

    plt.ylim([-0.02, uplimit])
    plt.xlabel('density (% of ' + str(n ** 2) + ' total possible edges)')
    plt.ylabel('Edge omission error')
    plt.title(str(samplesnum) + ' ' + str(n) + '-node graphs per density',
              multialignment='center')
    plt.legend(loc=0)

    plt.subplot(122)
    ax = sb.boxplot(x="density", y="time", hue="COE",
                    data=COE,
                    palette="Set2",
                    linewidth=lwd,
                    width=wds,
                    fliersize=fliersz)
    sb.stripplot(x="density", y="time", hue="COE",
                 data=COE,
                 palette='Set2',
                 linewidth=lwd,
                 size=4, jitter=True, edgecolor="gray")

    # ax.figure.get_axes()[0].set_yscale('log')
    plt.xticks([0, 1, 2], ('25%', '30%', '35%'))

    plt.ylim([-0.02, uplimit])
    plt.xlabel('density (% of ' + str(n ** 2) + ' total possible edges)')
    plt.ylabel('Edge comission error')
    plt.title(str(samplesnum) + ' ' + str(n) + '-node graphs per density',
              multialignment='center')
    plt.legend(loc=2)

    sb.set_context('poster')
    plt.savefig('/tmp/RASL_simulation.svgz', transparent=False, bbox_inches='tight', pad_inches=0)
    plt.show()
