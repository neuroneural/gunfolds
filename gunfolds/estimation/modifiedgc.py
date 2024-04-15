from gunfolds.utils import graphkit as gk
from gunfolds import conversions as conv
from itertools import combinations
import numpy as np
import statsmodels.api as sm
import copy

# Written by John Cook
# https://github.com/neuroneural/Undersampled_Graph_Estimation/blob/master/dbnestimation/Tools/grangercausality.py
# Modified to return edge weights by Sergey Plis


def gc(data, pval=0.05, bidirected=True):
    """
    :param data: time series data
    :type data: numpy matrix

    :param pval: p-value for statistical significance
    :type pval: float

    :param bidirected: flag to indicate bidirectionality
    :type bidirected: boolean

    :returns: modified graph and matrices D and B
    :rtype: tuple
    """
    n = data.shape[0]
    # Initialize D and B matrices with zeros
    D = np.zeros((n, n))
    B = np.zeros((n, n))

    data = np.asarray(np.r_[data[:, :-1], data[:, 1:]])

    def FastPCtest(y, x):
        y = y - 1
        x = x - 1
        nodelist = list(range(n))
        nodelist.remove(x)
        nodelist.remove(y)
        yd = data[n + int(y), :].T
        Xalt = data[:n]
        Xnull = np.vstack([Xalt, data[n + x, :]]).T
        Xalt = Xalt.T
        Xnull = sm.add_constant(Xnull)
        Xalt = sm.add_constant(Xalt)
        estnull = sm.OLS(yd, Xnull).fit()
        estalt = sm.OLS(yd, Xalt).fit()
        diff = sm.stats.anova_lm(estalt, estnull)
        # Return F-statistic and condition with the correct indices
        Fval = diff.iloc[1, 4]
        return Fval, diff.iloc[1, 5] > pval

    def grangertest(y, x):
        y = data[n + int(y) - 1, :].T
        Xnull = data[:n, :].T
        Xalt = np.vstack([data[: (x - 1), :], data[x:n, :]]).T
        Xnull = sm.add_constant(Xnull)
        Xalt = sm.add_constant(Xalt)
        estnull = sm.OLS(y, Xnull).fit()
        estalt = sm.OLS(y, Xalt).fit()
        diff = sm.stats.anova_lm(estalt, estnull)
        # Return F-statistic and condition with the correct indices
        Fval = diff.iloc[1, 4]  # F statistic comparing to previous model
        return Fval, diff.iloc[1, 5] > pval  # PR(>F) is the p-value

    # Assuming gk.superclique and conv.ian2g are defined elsewhere
    new_g = gk.superclique(n)
    g = conv.ian2g(new_g)

    for i in range(1, n + 1):
        for j in range(1, n + 1):
            Fval, condition = grangertest(j, i)
            if condition:
                sett = copy.deepcopy(g[str(i)][str(j)])
                sett.remove((0, 1))
                g[str(i)][str(j)] = sett
            else:
                D[i - 1, j - 1] = Fval

    biedges = combinations(range(1, n + 1), 2)
    for i, j in biedges:
        Fval, condition = FastPCtest(j, i)
        if bidirected and condition:
            sett = copy.deepcopy(g[str(i)][str(j)])
            sett.remove((2, 0))
            g[str(i)][str(j)] = sett
            g[str(j)][str(i)] = sett
        else:
            B[i - 1, j - 1] = Fval
            B[j - 1, i - 1] = Fval

    return conv.dict_format_converter(g), D, B
