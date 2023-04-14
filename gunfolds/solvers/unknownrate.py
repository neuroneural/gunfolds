""" BFS implementation of subgraph and supergraph Gu to G1 algorithm """
import gunfolds.utils.bfutils as bfu
from gunfolds.conversions import g2num, ug2num, num2CG
import gunfolds.utils.graphkit as gk
import gunfolds.utils.simpleloops as sls
from gunfolds.utils import load_data
from itertools import combinations
import numpy as np
import operator
from progressbar import ProgressBar, Bar
from functools import reduce


def prune_conflicts(H, g, elist):
    """Checks if adding an edge from the list to graph g causes a
    conflict with respect to H and if it does removes the edge
    from the list

    :param H: the undersampled graph
    :type H:
    
    :param g: a graph under construction
    :type g:
        
    :param elist: list of edges to check
    :type elist:
    
    :returns: 
    :rtype:
    """
    l = []
    for e in elist:
        gk.addanedge(g, e)
        if not bfu.call_u_conflicts(g, H):
            l.append(e)
        gk.delanedge(g, e)
    return l


def e2num(e, n):
    """
    :param e:
    :type e:
    
    :param n:
    :type n:
    
    :returns: 
    :rtype:
    """
    return 1 << (n*n + n - e[0]*n - e[1])


def le2num(elist, n):
    """
    :param elist: list of edges to check
    :type elist:
    
    :param n:
    :type n:
    
    :returns: 
    :rtype:
    """
    num = 0
    for e in elist:
        num |= e2num(e, n)
    return num


def ekey2e(ekey, n):
    """
    :param ekey:
    :type ekey:
    
    :param n:
    :type n:
    
    :returns: 
    :rtype:
    """
    idx = np.unravel_index(n * n - ekey .bit_length() - 1 + 1, (n, n))
    idx = tuple([x + 1 for x in idx])
    return ('%i %i' % idx).split(' ')


def cacheconflicts(num, cache):
    """Given a number representation of a graph and an iterable of
    conflicting subgraphs return True if the graph conflicts with any
    of them and false otherwise

    :param num: the number representation of a graph
    :type num:
    
    :param cache: an iterable of number representations of conflicting graphs
    :type cache:
          
    :returns: 
    :rtype:
    """
    for c in cache:
        if num & c == c:
            return True
    return False


class nobar:

    def update(self):
        return None

    def finish(self):
        return None


def start_progress_bar(iter, n, verbose=True):
    """
    :param iter:
    :type iter:
    
    :param n:
    :type n:
    
    :param verbose:
    :type verbose:
    
    :returns: 
    :rtype:
    """
    if verbose:
        pbar = ProgressBar(widgets=['%3s' % str(iter) +
                           '%10s' % str(n) + ' ', Bar('-'), ' '],
                           maxval=n).start()
    else:
        pbar = nobar()
    return pbar


def add2set_loop(ds, H, cp, ccf, iter=1, verbose=True,
                 capsize=100, currsize=0):
    """
    :param ds:
    :type ds:
    
    :param H:
    :type H:
    
    :param cp:
    :type cp:
    
    :param ccf:
    :type ccf:
    
    :param iter:
    :type iter:
    
    :param verbose:
    :type verbose:
    
    :param capsize:
    :type capsize:
    
    :param currsize:
    :type currsize:
    
    :returns: 
    :rtype:
    """
    n = len(H)
    dsr = {}
    s = set()
    ss = set()
    pbar = start_progress_bar(iter, len(ds), verbose=verbose)
    c = 0

    for gnum in ds:
        c += 1
        pbar.update(c)
        gset = set()
        eset = set()
        for sloop in ds[gnum]:
            if sloop & gnum == sloop:
                continue
            num = sloop | gnum
            if sloop in ccf and skip_conflictors(num, ccf[sloop]):
                continue
            if not (num in s):
                g = num2CG(num, n)
                if not bfu.call_u_conflicts(g, H):
                    s.add(num)
                    gset.add((num, sloop))
                    eset.add(sloop)
                    if bfu.call_u_equals(g, H):
                        ss.add(num)
                        if capsize <= len(ss) + currsize:
                            return dsr, ss

        for gn, e in gset:
            if e in cp:
                dsr[gn] = eset - cp[e] - set([e])
            else:
                dsr[gn] = eset - set([e])

    pbar.finish()
    return dsr, ss


def add2set_(ds, H, cp, ccf, iter=1, verbose=True, capsize=100):
    """
    :param ds:
    :type ds:
    
    :param H:
    :type H:
    
    :param cp:
    :type cp:
    
    :param ccf:
    :type ccf:
    
    :param iter:
    :type iter:
    
    :param verbose:
    :type verbose:
    
    :param capsize:
    :type capsize:
    
    :returns: 
    :rtype:
    """
    n = len(H)
    n2 = n * n + n
    dsr = {}
    s = set()
    ss = set()

    pbar = start_progress_bar(iter, len(ds), verbose=verbose)

    c = 0

    for gnum in ds:
        g = num2CG(gnum, n)
        c += 1
        pbar.update(c)
        glist = []
        elist = []
        eset = set()
        for e in ds[gnum]:
            if not e[1] in g[e[0]]:
                gk.addanedge(g, e)
                num = g2num(g)
                ekey = (1 << (n2 - e[0]*n - e[1]))
                if ekey in ccf and skip_conflictors(num, ccf[ekey]):
                    gk.delanedge(g, e)
                    continue
                if not (num in s):
                    s.add(num)
                    if not bfu.call_u_conflicts(g, H):
                        # cf, gl2 = bfu.call_u_conflicts2(g, H)
                        # if not cf:
                        glist.append((num, ekey))
                        elist.append(e)
                        eset.add(ekey)
                        if bfu.call_u_equals(g, H):
                            ss.add(num)
                            # if bfu.call_u_equals2(g, gl2, H): ss.add(num)
                        if capsize <= len(ss):
                            break
                gk.delanedge(g, e)

        for gn, e in glist:
            if e in cp:
                dsr[gn] = [ekey2e(k, n) for k in eset - cp[e]]
            else:
                dsr[gn] = elist
        if capsize <= len(ss):
            return dsr, ss

    pbar.finish()
    return dsr, ss


def skip_conflictors(gnum, ccf):
    """
    :param gnum:
    :type gnum:
    
    :param ccf:
    :type ccf:
    
    :returns: 
    :rtype:
    """
    pss = False
    for xx in ccf:
        if xx & gnum == xx:
            pss = True
            break
    return pss


def bconflictor(e, H):
    """
    :param e:
    :type e:
    
    :param H:
    :type H:
    
    :returns: 
    :rtype:
    """
    n = len(H)
    s = set()
    for v in H:
        s.add(e2num((v, e[0]), n) | e2num((v, e[1]), n))
    return s


def conflictor(e, H):
    """
    :param e:
    :type e:
    
    :param H:
    :type H:
    
    :returns: 
    :rtype:
    """
    n = len(H)

    def pairs(e):
        ekey = e2num(e, n)
        return [ekey | e2num((e[0], e[0]), n),
                ekey | e2num((e[1], e[1]), n)]

    def trios(e, H):
        s = set()
        for v in H:
            if not (v in e):
                s.add(e2num((e[0], v), n) |
                      e2num((v, e[1]), n) |
                      e2num((e[1], e[1]), n))
                s.add(e2num((e[0], e[0]), n) |
                      e2num((e[0], v), n) |
                      e2num((v, e[1]), n))
                s.add(e2num((e[0], v), n) |
                      e2num((v, v), n) |
                      e2num((v, e[1]), n))
        return s

    return trios(e, H).union(pairs(e))


def conflictor_set(H):
    """
    :param H:
    :type H:
    
    :returns: 
    :rtype:
    """
    s = set()
    for x in gk.inedgelist(H):
        s = s | conflictor(x, H)
    for x in gk.inbedgelist(H):
        s = s | bconflictor(x, H)
    return s


def conflictors(H):
    """
    :param H:
    :type H:
    
    :returns: 
    :rtype:
    """
    s = conflictor_set(H)
    ds = {}
    num = reduce(operator.or_, s)
    for i in range(len(bin(num))-2):
        if num & 1 << i:
            ds[1 << i] = [x for x in s if x & (1 << i)]
    return ds


def prune_loops(loops, H):
    """
    :param loops:
    :type loops:
    
    :param H:
    :type H:
    
    :returns: 
    :rtype:
    """
    l = []
    n = len(H)
    for loop in loops:
        g = num2CG(loop, n)
        if not bfu.call_u_conflicts_d(g, H):
            l.append(loop)
    return l


def lconflictors(H, sloops=None):
    """
    :param H:
    :type H:
    
    :param sloops:
    :type sloops:
    
    :returns: 
    :rtype:
    """
    if not sloops:
        sloops = prune_loops(allsloops(len(H)), H)
    s = conflictor_set(H)
    ds = {}
    num = reduce(operator.or_, s)
    for i in range(len(bin(num))-2):
        if num & 1 << i:
            cset = [x for x in s if x & (1 << i)]
            for sloop in sloops:
                if sloop & 1 << i:
                    ds.setdefault(sloop, []).extend(cset)
    return ds


def confpairs(H):
    """
    :param H:
    :type H:
    
    :returns: 
    :rtype:
    """
    n = len(H)
    g = {n: {} for n in H}
    d = {}

    edges = gk.edgelist(gk.complement(g))
    edges = prune_conflicts(H, g, edges)

    for p in combinations(edges, 2):
        gk.addedges(g, p)
        if bfu.call_u_conflicts(g, H):
            n1 = e2num(p[0], n)
            n2 = e2num(p[1], n)
            d.setdefault(n1, set()).add(n2)
            d.setdefault(n2, set()).add(n1)
        gk.deledges(g, p)

    return d


def lconfpairs(H, cap=10, sloops=None):
    """
    :param H:
    :type H:
    
    :param cap:
    :type cap:
    
    :param sloops:
    :type sloops:
    
    :returns: 
    :rtype:
    """
    n = len(H)
    d = {}
    if not sloops:
        sloops = prune_loops(allsloops(len(H)), H)
    c = 0
    for p in combinations(sloops, 2):
        g = num2CG(p[0] | p[1], n)
        if bfu.call_u_conflicts(g, H):
            d.setdefault(p[0], set()).add(p[1])
            d.setdefault(p[1], set()).add(p[0])
        if c >= cap:
            break
        c += 1
    return d


def iteqclass(H, verbose=True, capsize=100):
    """
    Find all graphs in the same equivalence class with respect to
    graph H and any undesampling rate.
    
    :param H:
    :type H:

    :param verbose:
    :type verbose:
    
    :param capsize:
    :type capsize:
    
    :returns: 
    :rtype:
    """
    if bfu.is_sclique(H):
        print('not running on superclique')
        return None
    g = {n: {} for n in H}
    s = set()
    Hnum = ug2num(H)
    if Hnum[1] == 0:
        s.add(Hnum[0])

    cp = confpairs(H)
    ccf = conflictors(H)

    edges = gk.edgelist(gk.complement(g))
    ds = {g2num(g): edges}

    if verbose:
        print('%3s' % 'i'+'%10s' % ' graphs')
    for i in range(len(H) ** 2):
        ds, ss = add2set_(ds, H, cp, ccf, iter=i,
                          verbose=verbose,
                          capsize=capsize)
        s = s | ss
        if capsize <= len(ss):
            break
        if not ds:
            break

    return s


def liteqclass(H, verbose=True, capsize=100, asl=None):
    """
    Find all graphs in the same equivalence class with respect to
    graph H and any undesampling rate.
    
    :param H:
    :type H:

    :param verbose:
    :type verbose:
    
    :param capsize:
    :type capsize:
    
    :param asl:
    :type asl:
    
    :returns: 
    :rtype:
    """
    if bfu.is_sclique(H):
        print('not running on superclique')
        return set([-1])
    s = set()
    cp = lconfpairs(H)
    if asl:
        sloops = asl
    else:
        sloops = prune_loops(allsloops(len(H)), H)
    ccf = lconflictors(H, sloops=sloops)
    ds = {0: sloops}
    if verbose:
        print('%3s' % 'i'+'%10s' % ' graphs')
    i = 0
    while ds:
        ds, ss = add2set_loop(ds, H, cp, ccf, iter=i,
                              verbose=verbose,
                              capsize=capsize,
                              currsize=len(s))
        s = s | ss
        i += 1
        if capsize <= len(s):
            break
    return s


def skip_conflict(g1, g2, ds):
    """
    :param g1:
    :type g1:

    :param g2:
    :type g2:
    
    :param ds:
    :type ds:
    
    :returns: 
    :rtype:
    """
    pss = False
    for ekey in ds:
        if (g1 & ekey) == ekey:
            if ekey in ds and cacheconflicts(g2, ds[ekey]):
                pss = True
                break
    return pss


def loop2graph(l, n):
    """
    :param l:
    :type l: (GUESS)integer

    :param n:
    :type n: (GUESS)integer
    
    :returns: 
    :rtype:
    """
    g = {i: {} for i in range(1, n+1)}
    for i in range(len(l) - 1):
        g[l[i]][l[i + 1]] = 1
    g[l[-1]][l[0]] = 1
    return g


def perm_circular(l, cp=None):
    """
    :param l:
    :type l: (GUESS)integer

    :param cp:
    :type cp:
    
    :returns: 
    :rtype:
    """
    if cp is None:
        # Delay import until its used
        cp = load_data.circp
    r = []
    n = len(l)
    for e in cp[n]:
        r.append([l[i] for i in e])
    return r


def gen_loops(n):
    """
    :param n:
    :type n: (GUESS)integer
    
    :returns: 
    :rtype:
    """
    l = [i for i in range(1, n + 1)]
    s = []
    for i in range(1, n + 1):
        for e in combinations(l, i):
            s.extend(perm_circular(e))
    return s


def allsloops(n, asl=None):
    """
    :param n:
    :type n: (GUESS)integer
    
    :param asl:
    :type asl:
    
    :returns: 
    :rtype:
    """
    if asl is None:
        asl = load_data.alloops
    if asl:
        return asl[n]
    s = []
    l = gen_loops(n)
    for e in l:
        s.append(g2num(loop2graph(e, n)))
    return s


def reverse(H, verbose=True, capsize=1000):
    """
    :param H:
    :type H:

    :param verbose:
    :type verbose:
    
    :param capsize:
    :type capsize:
    
    :returns: 
    :rtype:
    """
    n = len(H)
    s = set()

    g = gk.superclique(n)
    sloops = set(allsloops(n))

    ds = {g2num(g): sloops}

    if verbose:
        print('%3s' % 'i'+'%10s' % ' graphs')
    i = 0
    while ds:
        ds, ss = del_loop(ds, H, iter=i,
                          verbose=verbose,
                          capsize=capsize)
        s = s | ss
        i += 1
        if capsize <= len(s):
            break

    return s

# ----------------------


def build_loop_step(ds, loop, n, iter=1):
    """
    :param ds:
    :type ds:

    :param loop:
    :type loop:
    
    :param n:
    :type n:
    
    :param iter:
    :type iter:
    
    :returns: 
    :rtype:
    """
    dsr = {}
    s = set()
    ss = set()

    pbar = start_progress_bar(iter, len(ds))

    c = 0

    for gnum in ds:
        c += 1
        pbar.update(c)
        gset = set()
        eset = set()
        for sloop in ds[gnum]:
            num = sloop | gnum
            if not (num in s):
                g = num2CG(num, n)
                s.add(num)
                if bfu.forms_loop(g, loop):
                    ss.add(num)
                else:
                    gset.add((num, sloop))
                    eset.add(sloop)

        for gn, e in gset:
            dsr[gn] = eset - set([e])

    pbar.finish()
    return dsr, ss


def forward_loop_match(loop, n):
    """
    Start with an empty graph and keep adding simple loops until
    the loop is generated at some undersampling rate

    :param loop: binary encoding of the loop
    :type loop:
    
    :param n: number of nodes in the graph
    :type n:
    
    :returns: 
    :rtype:
    """
    s = set()
    sloops = allsloops(n)
    ds = {0: sloops}

    i = 0
    while ds:
        ds, ss = build_loop_step(ds, loop, n, iter=i)
        s = s | ss
        i += 1

    return s


def delAloop(g, loop):
    """
    :param g: (GUESS)graph that generates the loop
    :param g:
    
    :param loop: (GUESS)the reference loop
    :type loop:
    
    :returns: 
    :rtype:
    """
    n = len(g)
    l = []
    l = [g2num(loop2graph(s, n)) for s in sls.simple_loops(g, 0)]
    l = [num for num in l if not num == loop]
    print(loop, ': ',  l)
    return num2CG(reduce(operator.or_, l), n)


def reverse_loop_match(g, loop):
    """
    Start with a graph and keep removing loops while the loop is still matched

    :param g: graph that generates the loop
    :param g:
    
    :param loop: the reference loop
    :type loop:
    
    :returns: 
    :rtype:
    """
    s = set()
    n = len(g)

    def prune(g):
        cannotprune = True
        for l in sls.simple_loops(gk.digonly(g), 0):
            gg = delAloop(g, g2num(loop2graph(l, n)))
            if bfu.forms_loop(gg, loop):
                cannotprune = False
                prune(gg)
        if cannotprune:
            print('one')
            s.add(g)

    prune(g)
    return s


def reverse_edge_match(g, loop):
    """
    Start with a graph and keep removing loops while the loop is still matched

    :param g: graph that generates the loop
    :param g:
    
    :param loop: the reference loop
    :type loop:
    
    :returns: 
    :rtype:
    """
    s = set()
    n = len(g)

    def prune(g):
        cannotprune = True
        for l in gk.edgelist(gk.digonly(g)):
            gk.delanedge(g, l)
            if bfu.forms_loop(g, loop):
                cannotprune = False
                prune(g)
            gk.addanedge(g, l)
        if cannotprune:
            s.add(g2num(g))

    prune(g)
    return s


def matchAloop(loop, n):
    """Returns a set of minimal graphs that generate this loop

    :param loop: binary encoding of the loop
    :type loop:
    
    :param n: number of nodes in the graph
    :type n:
    
    :returns: 
    :rtype:
    """
    s = set()
    l = forward_loop_match(loop, n)
    print(len(l))
    for g in l:
        s = s | reverse_edge_match(num2CG(g, n), loop)

    return s

# ----------------------


def del_loop(ds, H, iter=0):
    """
    :param ds:
    :type ds:
    
    :param H:
    :type H:
    
    :param iter:
    :type iter:
    
    :returns: 
    :rtype:
    """
    n = len(H)

    dsr = {}
    ss = set()
    print(iter,)
    for gnum in ds:
        gset = []
        s = set()
        for sloop in ds[gnum]:
            rset = ds[gnum] - set([sloop])
            num = reduce(operator.or_, rset)
            if not (num in s):
                g = num2CG(num, n)
                if bfu.overshoot(g, H):
                    s.add(num)
                    gset.append((num, rset))

        if gset == []:
            print('.',)
            ss.add(gnum)

        for gn in gset:
            dsr[gn[0]] = gn[1]
    print('')
    return dsr, ss


def main():
    g = gk.ringmore(6, 1)
    H = bfu.undersample(g, 1)
    ss = liteqclass(H)
    print(ss)


if __name__ == "__main__":
    main()
