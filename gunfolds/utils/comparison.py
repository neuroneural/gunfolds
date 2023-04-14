from gunfolds.conversions import num2CG, graph2nx
from gunfolds.utils.bfutils import undersample
from gunfolds.utils import ecj
import networkx as nx
import numpy as np
import operator
from functools import reduce

np.random.RandomState()


def has_self_loops(G):
    """
    Checks if a graph has self loop and vise versa

    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: True, if the graph has self loop and vise versa
    :rtype: boolean
    """
    for u in G:
        if u in G[u]:
            return True
    return False


def randSCC(n):
    """
    Returns a random ``gunfolds`` graph with gcd>1

    :param n: number of nodes
    :type n: integer
    
    :returns: random ``gunfolds`` graph with gcd>1
    :rtype: dictionary (``gunfolds`` graphs)
    """
    G = num2CG(np.random.randint(2 ** (n ** 2)), n)
    while (len(ecj.scc(G)) > 1) or gcd4scc(G) > 1:
        G = num2CG(np.random.randint(2 ** (n ** 2)), n)
    return G


def SM_fixed(Gstar, G, iter=5):
    """ 
    (Ask)

    :param Gstar:
    :type Gstar:
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :param iter:
    :type iter: integer
    
    :returns: 
    :rtype: 
    """
    compat = []
    for j in range(0, iter):
        if Gstar == undersample(G, j):
            compat.append(j)
    return compat


def SM_converging(Gstar, G):
    """
    Gstar is the undersampled reference graph, while G is the starting
    graph. The  code searches  over all undersampled  version of  G to
    find all matches with Gstar
    
    :param Gstar:
    :type Gstar:
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: 
    :rtype: 
    """
    compat = []
    GG = G
    Gprev = G
    if G == Gstar:
        return [0]
    j = 1
    G = undersample(GG, j)
    while not (G == Gprev):
        if Gstar == G:
            compat.append(j)
        j += 1
        Gprev = G
        G = undersample(GG, j)
    return compat


def search_match(Gstar, G, iter=5):
    """ 
    :param Gstar:
    :type Gstar:
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :param iter:
    :type iter: integer
    
    :returns: 
    :rtype: 
    """
    if gcd4scc(G) > 1:
        return SM_fixed(Gstar, G, iter=iter)
    return SM_converging(Gstar, G)


def has_sink(G):
    """ 
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: 
    :rtype: 
    """
    return not reduce(operator.and_, [bool(G[n]) for n in G], True)


def has_root(G):
    """ 
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: 
    :rtype: 
    """
    return has_sink(ecj.tr(G))


def gcd4scc(SCC):
    """ 
    Returns the greatest common divisor of simple loop lengths and in one SCC

    :param SCC: ``gunfolds`` graph
    :type SCC: dictionary (``gunfolds`` graphs)
    
    :returns: the greatest common divisor of simple loop lengths and in one SCC
    :rtype: integer
    """
    g = graph2nx(SCC)
    return ecj.listgcd([len(x) for x in nx.simple_cycles(g)])


def compatible_at_u(uGstar):
    """ 
    :param uGstar:
    :type uGstar:
    
    :returns: 
    :rtype: 
    """
    compat = []
    n = len(uGstar)
    numG = 2 ** (n ** 2)
    # pbar = Percentage()
    for i in range(1, numG):
        G = num2CG(i, n)
        # pbar.update(i+1)
        if len(ecj.scc(G)) > 1:
            continue
        l = search_match(uGstar, G, iter=5)
        if l:
            compat.append((l, G))
    # pbar.finish()
    return compat
