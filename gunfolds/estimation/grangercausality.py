from gunfolds.utils import graphkit as gk
from gunfolds import conversions as conv
from itertools import combinations
import numpy as np
import statsmodels.api as sm
import copy

# Written by John Cook
# https://github.com/neuroneural/Undersampled_Graph_Estimation/blob/master/dbnestimation/Tools/grangercausality.py


def gc(data, pval=0.05, bidirected=True):
    """
    :param data: time series data
    :type data: numpy matrix
    
    :param pval: (Ask)
    :type pval: float
    
    :param bidirected: (Ask) 
    :type bidirected: boolean
    
    :returns: 
    :rtype: 
    """
    n = data.shape[0]
    # stack the data: first n rows is t-1 slice, the next n are slice t
    data = np.asarray(np.r_[data[:, :-1], data[:, 1:]])

    def FastPCtest(y, x):
        y = y-1
        x = x-1
        nodelist = list(range(n))
        nodelist.remove(x)
        nodelist.remove(y)
        yd = data[n+int(y), :].T
        #        for L in range(1,len(nodelist)+1):
        #            combs=combinations(nodelist,L)
        #            for sub in combs:
        Xalt = data[:n]
        Xnull = np.vstack([Xalt, data[n+x, :]]).T
        Xalt = Xalt.T
        Xnull = sm.add_constant(Xnull)
        Xalt = sm.add_constant(Xalt)
        estnull = sm.OLS(yd, Xnull).fit()
        estalt = sm.OLS(yd, Xalt).fit()
        diff = sm.stats.anova_lm(estalt, estnull)
        if diff.iloc[1, 5] > pval:
            return True
        return False

    def grangertest(y, x):
        y = data[n+int(y)-1, :].T
        Xnull = data[:n, :].T
        Xalt = np.vstack([data[:(x-1), :], data[x:n, :]]).T
        Xnull = sm.add_constant(Xnull)
        Xalt = sm.add_constant(Xalt)
        estnull = sm.OLS(y, Xnull).fit()
        estalt = sm.OLS(y, Xalt).fit()
        diff = sm.stats.anova_lm(estalt, estnull)
        return diff.iloc[1, 5] > pval
    
    new_g = gk.superclique(n)
    # print(new_g)
    g = conv.ian2g(new_g)
    for i in range(1, n+1):
        for j in range(1, n+1):
            if grangertest(j, i):
                sett = copy.deepcopy(g[str(i)][str(j)])
                sett.remove((0, 1))
                g[str(i)][str(j)] = sett
    biedges = combinations(range(1, n+1), 2)
    for i, j in biedges:
        if bidirected:
            if FastPCtest(j, i):
                sett = copy.deepcopy(g[str(i)][str(j)])
                sett.remove((2, 0))
                g[str(i)][str(j)] = sett
                g[str(j)][str(i)] = sett
        else:
            sett = copy.deepcopy(g[str(i)][str(j)])
            sett.remove((2, 0))
            g[str(i)][str(j)] = sett
            g[str(j)][str(i)] = sett
    # print(g)
    return conv.dict_format_converter(g)
