from gunfolds.utils import clingo as clg
from gunfolds.utils import graphkit as gk
from progressbar import ProgressBar, Percentage
import itertools
from gunfolds.conversions import g2vec, vec2g
import operator as op
import copy
import sys
from functools import reduce


def ncr(n, r):
    """
    :param n:
    :type n:
    
    :param r:
    :type r:
    
    :returns: 
    :rtype: 
    """
    r = min(r, n-r)
    if r == 0:
        return 1
    numer = reduce(op.mul, range(n, n-r, -1))
    denom = reduce(op.mul, range(1, r+1))
    return numer//denom


def num_nATstep(g, step):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param step: (GUESS)Hamming distance from `v` of the vectors to generate
    :type step:
    
    :returns: 
    :rtype: 
    """

    n = len(g)
    b = n*n + ncr(n, 2)
    return ncr(b, step)


def num_neighbors(g, step):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param step: (GUESS)Hamming distance from ``v`` of the vectors to generate
    :type step:
    
    :returns: 
    :rtype: 
    """
    n = len(g)
    b = n*n + ncr(n, 2)
    l = 0
    for i in range(step+1):
        l += ncr(b, i)
    return l


def hamming_neighbors(v, step):
    """
    Returns an iterator over all neighbors of the binary vector ``v`` Hamming ``step`` away from it

    :param v: a binary vector representing a G^u graph
    :type v:
    
    :param step: Hamming distance from ``v`` of the vectors to generate
    :type step:
    
    :returns: 
    :rtype: 
    """
    for e in itertools.combinations(range(len(v)), step):
        b = copy.deepcopy(v)
        for i in e:
            b[i] = int(not b[i])
        yield b


def find_nearest_reachable(g2, maxsolutions=100,
                           max_depth=5, timeout=3600,
                           cpath='',
                           verbose=True):
    """
    :param g2:
    :type g2:
    
    :param maxsolutions:
    :type maxsolutions:
    
    :param max_depth:
    :type max_depth:
    
    :param timeout:
    :type timeout:
    
    :param cpath:
    :type cpath:
    
    :param verbose:
    :type verbose:
    
    :returns: 
    :rtype: 
    """
    c = 0
    s = clg.eqclass(g2, capsize=maxsolutions, timeout=timeout, cpath=cpath)
    if s:
        return s, c
    c += 1
    step = 1
    n = len(g2)
    v = g2vec(g2)
    while True:
        c = 0
        if verbose:
            w = ['neighbors checked @ step ' + str(step) + ': ', Percentage(), ' ']
            pbar = ProgressBar(maxval=num_nATstep(g2, step), widgets=w).start()

        for e in hamming_neighbors(v, step):
            g = vec2g(e, n)
            if not gk.scc_unreachable(g):
                s = clg.eqclass(g,
                                capsize=maxsolutions,
                                timeout=timeout,
                                cpath=cpath)
            else:
                s = set()
            if s:
                if verbose:
                    pbar.finish()
                return s, num_neighbors(g2, step-1) + c

            if verbose:
                pbar.update(c)
                sys.stdout.flush()

            c += 1
        if verbose:
            pbar.finish()

        if step >= max_depth:
            return set(), num_neighbors(g2, step-1) + c
        step += 1
