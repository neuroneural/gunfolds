from __future__ import division
import copy
from functools import wraps
from gunfolds.utils import bfutils as bfu
from gunfolds.conversions import g2num, graph2nx
from gunfolds.utils import graphkit as gk
from gunfolds.utils.graphkit import density
import itertools
from networkx import strongly_connected_components
import numpy as np
import random
import scipy
import time
from functools import reduce


def chunks(l, n):
    """ 
    Yield successive n-sized chunks from l.
    
    :param l:
    :type l:
    
    :param n:
    :type n: integer
    """
    for i in range(0, len(l), n):
        yield l[i:i + n]


def purgepath(path, l):
    """
    :param path:
    :type path:
    
    :param l:
    :type l:
    """
    for i in range(1, len(path) - 1):
        l.remove((path[i], path[i + 1]))


def try_till_d_path(g, d, gt, order=None):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param d:
    :type d:
    
    :param gt:
    :type gt:
    
    :param order:
    :type order:
    
    :returns: 
    :rtype: 
    """
    k = []
    i = 1
    while not k:
        if order:
            k = [x for x in length_d_paths(g, order(i), d)]
        else:
            k = [x for x in length_d_paths(g, i, d)]
        i += 1
        if i > len(g):
            return []

    ld = []
    for i in range(min(10, len(k))):
        ld.append(len(checkApath([2] + k[i], gt)))
    return k[0]


def try_till_path(g, gt):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param gt:
    :type gt:
    
    :returns: 
    :rtype: 
    """
    gx = graph2nx(g)
    sccl = [x for x in strongly_connected_components(gx)]
    # take the largest
    ln = [len(x) for x in sccl]
    idx = np.argsort(ln)
    d = len(sccl[idx[-1]]) - 1
    k = []
    while not k:
        if d < 5:
            return []
        k = try_till_d_path(g, d, gt)
        d -= 1
    return k


def gpurgepath(g, path):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param path:
    :type path:
    """
    for i in range(1, len(path) - 1):
        del g[path[i]][path[i + 1]]


def forks(n, c, el, bl, doempty=lambda x, y: True):
    """
    :param n: single node string
    :type n: (guess)string
    
    :param c: mutable list of children of node n (will be changed as a side effect)
    :type c:
    
    :param el: list of edges available for construction (side effect change)
    :type el:
    
    :param bl: list of bidirected edges
    :type bl:
    
    :param doempty:
    :type doempty:
    
    :returns: 
    :rtype: 
    """
    l = []
    r = set()
    for p in [x for x in itertools.combinations(c, 2)]:
        if doempty(p, bl) and (n, p[0]) in el and (n, p[1]) in el:
            l.append((n,) + p)
            el.remove((n, p[0]))
            el.remove((n, p[1]))
            r.add(p[0])
            r.add(p[1])
    for e in r:
        c.remove(e)
    return l


def childrenedges(n, c, el):
    """
    :param n: single node string
    :type n: (guess)string
    
    :param c: mutable list of children of node n (will be changed as a side effect)
    :type c:
    
    :param el: list of edges available for construction (side effect change)
    :type el:
    
    :returns: 
    :rtype: 
    """
    l = []
    r = set()
    for ch in c:
        if (n, ch) in el:
            l.append((n, ch))
            el.remove((n, ch))
            r.add(ch)
    for e in r:
        c.remove(e)
    return l


def make_emptyforks(n, c, el, bl):
    """ 
    An empty fork is a fork without the bidirected edge 

    :param n: single node string
    :type n: (guess)string
    
    :param c: mutable list of children of node n (will be changed as a side effect)
    :type c:
    
    :param el: list of edges available for construction (side effect change)
    :type el:
    
    :param bl: list of bidirected edges
    :type bl:
    
    :returns: 
    :rtype: 
    """
    return forks(n, c, el, bl, doempty=lambda x, y: not (x in y))


def make_fullforks(n, c, el, bl):
    """
    :param n: single node string
    :type n: (guess)string
    
    :param c: mutable list of children of node n (will be changed as a side effect)
    :type c:
    
    :param el: list of edges available for construction (side effect change)
    :type el:
    
    :param bl: list of bidirected edges
    :type bl:
    
    :returns: 
    :rtype: 
    """
    return forks(n, c, el, bl, doempty=lambda x, y: x in y)


def make_longpaths(g, el):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param el: (GUESS)list of edges available for construction (side effect change)
    :type el:
    
    :returns: 
    :rtype: 
    """
    l = []
    gc = copy.deepcopy(g)
    for i in range(16):
        k = try_till_path(gc, g)
        if len(k) < 5:
            break
        if k:
            l.append((2,) + tuple(k))
            purgepath(l[-1], el)
            gpurgepath(gc, l[-1])
        else:
            break
    return l


def make_allforks_and_rest(g, el, bl, dofullforks=True):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param el: list of edges available for construction (side effect change)
    :type el:
    
    :param bl: (GUESS)list of bidirected edges
    :type bl:
    
    :param dofullforks:
    :type dofullforks:
    
    :returns: 
    :rtype: 
    """
    l = []
    r = []
    nodes = [n for n in g]
    random.shuffle(nodes)
    for n in nodes:

        c = [e for e in g[n] if g[n][e] in (1, 3)]  # all children
        if len(c) == 1:
            if (n, c[0]) in el:
                r.append((n, c[0]))
                el.remove((n, c[0]))
        elif len(c) > 1:
            l.extend(make_emptyforks(n, c, el, bl))
            if dofullforks:
                l.extend(make_fullforks(n, c, el, bl))
            r.extend(childrenedges(n, c, el, bl))
    return l, r


def vedgelist(g, pathtoo=False):
    """ 
    Returns a list of tuples for edges of g and forks
    a superugly organically grown function that badly needs refactoring
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param pathtoo:
    :type pathtoo:
    
    :returns: 
    :rtype: 
    """
    l = []
    el = gk.edgelist(g)
    bl = gk.bedgelist(g)

    if pathtoo:
        l.extend(make_longpaths(g, el))
    l2, r = make_allforks_and_rest(g, el, bl, dofullforks=True)
    l.extend(l2)

    A, singles = makechains(r)

    if singles:
        B, singles = makesinks(singles)
    else:
        B, singles = [], []

    l = longpaths_pick(l) + threedges_pick(l) + A + B + singles
    return l


def twoedges_pick(l):
    """
    :param l:
    :type l:
    
    :returns: 
    :rtype: 
    """
    return [e for e in l if len(e) == 2]


def threedges_pick(l):
    """
    :param l:
    :type l:
    
    :returns: 
    :rtype: 
    """
    return [e for e in l if len(e) == 3]


def longpaths_pick(l):
    """
    :param l:
    :type l:
    
    :returns: 
    :rtype: 
    """
    return [e for e in l if len(e) > 3 and e[0] == 2]


def makechains(l):
    """ 
    Greedily construct 2 edge chains from edge list
    
    :param l:
    :type l:
    
    :returns: 
    :rtype: 
    """
    ends = {e[1]: e for e in l}
    starts = {e[0]: e for e in l}
    r = []
    singles = []
    while l:

        e = l.pop()
        if e[1] in starts and e[0] != e[1] and starts[e[1]] in l:
            r.append((0, e[0],) + starts[e[1]])
            l.remove(starts[e[1]])
        elif e[0] in ends and e[0] != e[1] and ends[e[0]] in l:
            r.append((0,) + ends[e[0]] + (e[1],))
            l.remove(ends[e[0]])
        else:
            singles.append(e)
    return r, singles


def makesink(es):
    """
    :param es:
    :type es:
    
    :returns: 
    :rtype: 
    """
    return (1, es[0][0],) + es[1]


def makesinks(l):
    """ 
    Greedily construct 2 edge sinks ( b->a<-c ) from edge list

    :param l:
    :type l:
    
    :returns: 
    :rtype: 
    """
    sinks = {}
    for e in l:
        if e[1] in sinks:
            sinks[e[1]].append(e)
        else:
            sinks[e[1]] = [e]
    r = []
    singles = []
    for e in sinks:
        if len(sinks[e]) > 1:
            for es in chunks(sinks[e], 2):
                if len(es) == 2:
                    r.append(makesink(es))
                else:
                    singles.append(es[0])
        else:
            singles.append(sinks[e][0])
    return r, singles


def selfloops(l, g):
    """
    :param l:
    :type l:
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: 
    :rtype: 
    """
    return reduce(lambda x, y: x and y, map(lambda x: gk.selfloop(x, g), l))


def checkbedges(v, bel, g2):
    """
    :param v:
    :type v:
    
    :param bel:
    :type bel:
   
    :param g2:
    :type g2:
    
    :returns: 
    :rtype: 
    """
    r = []
    for e in bel:
        if e == tuple(v[1:]) and not selfloops(e, g2):
            r.append(e)
        if e == (v[2], v[1]) and not selfloops(e, g2):
            r.append(e)
    for e in r:
        bel.remove(e)
    return bel


def checkedge(e, g2):
    """
    :param e:
    :type e:
   
    :param g2:
    :type g2:
    
    :returns: 
    :rtype: 
    """
    if e[0] == e[1]:
        l = [n for n in g2 if n in g2[n]]
        s = set()
        for v in g2[e[0]]:
            s.add(g2[e[0]][v])
        if 2 not in s and 3 not in s:
            l.remove(e[0])
        return l
    else:
        return g2.keys()


def single_nodes(v, g2):
    """ 
    Returns a list of singleton nodes allowed for merging with ``v``

    :param v:
    :type v:
   
    :param g2:
    :type g2:
    
    :returns: a list of singleton nodes allowed for merging with ``v``
    :rtype: 
    """
    l = [(n, n) for n in g2 if not (n in v) and len(g2[n]) > 1]
    return l


def checkvedge(v, g2):
    """ 
    Nodes to check to merge the virtual nodes of ``v`` ( b<-a->c )
    
    :param v:
    :type v:
   
    :param g2:
    :type g2:
    
    :returns: 
    :rtype: 
    """
    l = gk.bedgelist(g2)
    if (v[1], v[2]) in l:
        l = single_nodes(v, g2) + checkbedges(v, l, g2)
        for n in v:
            if n in g2[n]:
                l.append((n, n))
    else:
        l = checkbedges(v, l, g2)
    return list(set(l))


def checkAedge(v, g2):
    """ 
    Nodes to check to merge the virtual nodes of A ( b->a<-c )

    :param v:
    :type v:

    :param g2:
    :type g2:
    
    :returns: 
    :rtype: 
    """
    l = []
    # try all pairs but the sources
    for pair in itertools.combinations(g2, 2):
        # if pair == (v[1],v[2]): continue
        # if pair == (v[2],v[1]): continue
        l.append(pair)
        l.append(pair[::-1])
    for n in g2:
        l.append((n, n))
    return l


def checkcedge(c, g2):
    """ 
    Nodes to check to merge the virtual nodes of ``c`` ( a->b->c )
    
    :param c:
    :type c:
   
    :param g2:
    :type g2:
    
    :returns: 
    :rtype: 
    """
    l = gk.edgelist(g2)
    return list(set(l))


def checkApath(p, g2):
    """
    :param p:
    :type p:
   
    :param g2:
    :type g2:
    
    :returns: 
    :rtype: 
    """
    d = len(p) - 2
    l = []
    for n in g2:
        l.extend([tuple(x) for x in length_d_loopy_paths(g2, n, d, p[1:])])
    # k = prunepaths_1D(g2, p, l)
    return l


def isedge(v):
    """
    :param v:
    :type v:
    
    :returns: 
    :rtype: 
    """
    return len(v) == 2  # a->b


def isvedge(v):
    """
    :param v:
    :type v:
    
    :returns: 
    :rtype: 
    """
    return len(v) == 3  # b<-a->c


def isCedge(v):
    """
    :param v:
    :type v:
    
    :returns: 
    :rtype: 
    """
    return len(v) == 4 and v[0] == 0  # a->b->c


def isAedge(v):
    """
    :param v:
    :type v:
    
    :returns: 
    :rtype: 
    """
    return len(v) == 4 and v[0] == 1  # a->c<-b


def isApath(v):
    """
    :param v:
    :type v:
    
    :returns: 
    :rtype: 
    """
    return len(v) >= 4 and v[0] == 2  # a->b->...->z


def checker(n, ee):
    """
    :param n:
    :type n:
    
    :param ee:
    :type ee:
    
    :returns: 
    :rtype: 
    """
    g = gk.ringmore(n, ee)
    g2 = bfu.increment(g)
    d = checkable(g2)
    t = [len(d[x]) for x in d]
    r = []
    n = len(g2)
    ee = len(gk.edgelist(g2))
    for i in range(1, len(t)):
        r.append(sum(np.log10(t[:i])) - ee * np.log10(n))
    return r


def checkerDS(n, ee):
    """
    :param n:
    :type n:
    
    :param ee:
    :type ee:
    
    :returns: 
    :rtype: 
    """
    g = gk.ringmore(n, ee)
    g2 = bfu.increment(g)
    gg = checkable(g2)
    d, p, idx = conformanceDS(g2, gg, gg.keys())
    t = [len(x) for x in p]
    r = []
    n = len(g2)
    ee = len(gk.edgelist(g2))
    for i in range(1, len(t)):
        r.append(sum(np.log10(t[:i])) - ee * np.log10(n))
    return r


def checkable(g2):
    """
    :param g2:
    :type g2:
    
    :returns: 
    :rtype: 
    """
    d = {}
    g = cloneempty(g2)
    vlist = vedgelist(g2, pathtoo=False)
    for v in vlist:
        if isvedge(v):
            d[v] = checkvedge(v, g2)
        elif isCedge(v):
            d[v] = checkcedge(v, g2)
        elif isAedge(v):
            d[v] = checkAedge(v, g2)
        elif isApath(v):
            d[v] = checkApath(v, g2)
        else:
            d[v] = checkedge(v, g2)

    # check if some of the otherwise permissible nodes still fail
    c = [ok2add2edges,
         ok2addavedge,
         ok2addacedge,
         ok2addaAedge,
         ok2addapath]

    for e in d:
        checks_ok = c[edge_function_idx(e)]
        for n in d[e]:
            if not checks_ok(e, n, g, g2):
                d[e].remove(n)

    return d


def inorder_check2(e1, e2, j1, j2, g2, f=[], c=[]):
    """
    :param e1:
    :type e1:
    
    :param e2:
    :type e2:
    
    :param j1:
    :type j1:
    
    :param j2:
    :type j2:
    
    :param g2:
    :type g2:
    
    :param f:
    :type f:
    
    :param c:
    :type c:
    
    :returns: 
    :rtype: 
    """
    g = cloneempty(g2)  # the graph to be used for checking

    if f == []:
        f = [(add2edges, del2edges, mask2edges),
             (addavedge, delavedge, maskavedge),
             (addacedge, delacedge, maskaCedge),
             (addaAedge, delaAedge, maskaAedge),
             (addapath, delapath, maskapath)]

    if c == []:
        c = [ok2add2edges,
             ok2addavedge,
             ok2addacedge,
             ok2addaAedge,
             ok2addapath]

    adder, remover, masker = f[edge_function_idx(e1)]
    checks_ok = c[edge_function_idx(e2)]

    d = {}
    s1 = set()
    s2 = set()
    for c1 in j1:  # for each connector
        mask = adder(g, e1, c1)
        d[c1] = set()
        for c2 in j2:
            if checks_ok(e2, c2, g, g2):
                d[c1].add(c2)
                s2.add(c2)
        remover(g, e1, c1, mask)
        if d[c1]:
            s1.add(c1)
    return d, s1, s2


def check3(e1, e2, e3, j1, j2, j3, g2, f=[], c=[]):
    """
    :param e1:
    :type e1:
    
    :param e2:
    :type e2:

    :param e3:
    :type e3:
    
    :param j1:
    :type j1:
    
    :param j2:
    :type j2:
    
    :param j3:
    :type j3:
    
    :param g2:
    :type g2:
    
    :param f:
    :type f:
    
    :param c:
    :type c:
    
    :returns: 
    :rtype: 
    """
    g = cloneempty(g2)  # the graph to be used for checking
    if f == []:
        f = [(add2edges, del2edges, mask2edges),
             (addavedge, delavedge, maskavedge),
             (addacedge, delacedge, maskaCedge),
             (addaAedge, delaAedge, maskaAedge),
             (addapath, delapath, maskapath)]
    if c == []:
        c = [ok2add2edges,
             ok2addavedge,
             ok2addacedge,
             ok2addaAedge,
             ok2addapath]

    adder1, remover1, masker1 = f[edge_function_idx(e1)]
    adder2, remover2, masker2 = f[edge_function_idx(e2)]

    checks_ok2 = c[edge_function_idx(e2)]
    checks_ok3 = c[edge_function_idx(e3)]

    s1 = set()
    s2 = set()
    s3 = set()

    for c1 in j1:  # for each connector
        mask1 = adder1(g, e1, c1)
        append_set1 = False
        for c2 in j2:
            append_set2 = False
            if checks_ok2(e2, c2, g, g2):
                mask2 = adder2(g, e2, c2)
                for c3 in j3:
                    if checks_ok3(e3, c3, g, g2):
                        append_set1 = append_set2 = True
                        s3.add(c3)
                remover2(g, e2, c2, mask2)
            if append_set2:
                s2.add(c2)
        if append_set1:
            s1.add(c1)
        remover1(g, e1, c1, mask1)
    return s1, s2, s3


def del_empty(d):
    """
    :param d:
    :type d:
    
    :returns: 
    :rtype: 
    """
    l = [e for e in d]
    for e in l:
        if d[e] == set():
            del d[e]
    return d


def inorder_checks(g2, gg):
    """
    :param g2:
    :type g2:
    
    :param gg:
    :type gg:
    
    :returns: 
    :rtype: 
    """
    # idx = np.argsort([len(gg[x]) for x in gg.keys()])
    # ee = [gg.keys()[i] for i in idx] # to preserve the order
    ee = [e for e in gg]  # to preserve the order
    # cds = conformanceDS(g2, ee)
    # oo = new_order(g2, ee, repeats=100, cds=None)
    # ee = oo[0]
    random.shuffle(ee)
    d = {}  # new datastructure
    d[ee[0]] = {0: gg[ee[0]]}
    for i in range(len(ee) - 1):
        d[ee[i + 1]] = del_empty(inorder_check2(ee[i], ee[i + 1],
                                                gg[ee[i]], gg[ee[i + 1]], g2)[0])
    return ee, d


def cloneempty(g):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: 
    :rtype: 
    """
    return {n: {} for n in g}  # return a graph with no edges


def ok2addanedge1(s, e, g, g2, rate=1):
    """
    :param s : start
    :type s:
    
    :param e : end
    :type e:
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param g2:
    :type g2:
    
    :param rate:
    :type rate:
    
    :returns: 
    :rtype: 
    """
    # directed edges
    # self-loop
    if s == e and not (e in g2[s]):
        return False
    for u in g:  # Pa(s) -> e
        if s in g[u] and not (e in g2[u] and g2[u][e] in (1, 3)):
            return False
    for u in g[e]:  # s -> Ch(e)
        if not (u in g2[s] and g2[s][u] in (1, 3)):
            return False
    # bidirected edges
    for u in g[s]:  # e <-> Ch(s)
        if u != e and not (u in g2[e] and g2[e][u] in (2, 3)):
            return False
    return True


def ok2addanedge2(s, e, g, g2, rate=1):
    """
    :param s : start
    :type s:
    
    :param e : end
    :type e:
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
   
    :param g2:
    :type g2:
    
    :param rate:
    :type rate:
    
    :returns: 
    :rtype: 
    """
    mask = addanedge(g, (s, e))
    value = bfu.undersample(g, rate) == g2
    delanedge(g, (s, e), mask)
    return value


def ok2addanedge(s, e, g, g2, rate=1):
    """
    :param s : start
    :type s:
    
    :param e : end
    :type e:
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param g2:
    :type g2:
    
    :param rate:
    :type rate:
    
    :returns: 
    :rtype: 
    """
    f = [ok2addanedge1, ok2addanedge2]
    return f[min([1, rate - 1])](s, e, g, g2, rate=rate)


def ok2add2edges(e, p, g, g2):
    """
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param g2:
    :type g2:
    
    :returns: 
    :rtype: 
    """
    return edge_increment_ok(e[0], p, e[1], g, g2)


def maskanedge(g, e):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param e:
    :type e:
    
    :returns: 
    :rtype: 
    """
    return [e[1] in g[e[0]]]


def mask2edges(g, e, p):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :returns: 
    :rtype: 
    """
    return [p in g[e[0]], e[1] in g[p]]


def maskavedge(g, e, p):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :returns: 
    :rtype: 
    """
    return [p[0] in g[e[0]], p[1] in g[e[0]],
            e[1] in g[p[0]], e[2] in g[p[1]]]


def maskaAedge(g, e, p):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :returns: 
    :rtype: 
    """
    return [p[0] in g[e[1]], p[1] in g[e[2]],
            e[3] in g[p[0]], e[3] in g[p[1]]]


def maskaCedge(g, e, p):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :returns: 
    :rtype: 
    """
    return [p[0] in g[e[1]], e[2] in g[p[0]],
            p[1] in g[e[2]], e[3] in g[p[1]]]


def maskapath(g, e, p):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :returns: 
    :rtype: 
    """
    mask = []
    for i in range(len(p)):
        mask.append(p[i] in g[e[i + 1]])
        mask.append(e[i + 2] in g[p[i]])
    return mask


def maskaVpath(g, e, p):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :returns: 
    :rtype: 
    """
    mask = []
    mask.extend([p[0] in g[e[0]], e[1] in g[p[-1]]])
    for i in range(1, len(p)):
        mask.append(p[i] in g[p[i - 1]])
    return mask


def addanedge(g, e):
    """
    Add edge e[0] -> e[1] to g

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param e:
    :type e:
    
    :returns: 
    :rtype: 
    """
    mask = maskanedge(g, e)
    g[e[0]][e[1]] = 1
    return mask


def delanedge(g, e, mask):
    """
    Delete edge e[0] -> e[1] from g if it was not there before

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param e:
    :type e:
    
    :param mask:
    :type mask:
    
    :returns: 
    :rtype: 
    """
    if not mask[0]:
        g[e[0]].pop(e[1], None)


def add2edges(g, e, p):
    """
    Break edge e[0] -> e[1] into two pieces
    e[0] -> p and p -> e[1]
    and add them to g

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :returns: 
    :rtype: 
    """
    mask = mask2edges(g, e, p)
    g[e[0]][p] = g[p][e[1]] = 1
    return mask


def del2edges(g, e, p, mask):
    """
    Restore the graph as it was before adding e[0]->p and p->e[1]

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :param mask:
    :type mask:
    
    :returns: 
    :rtype: 
    """
    if not mask[0]:
        g[e[0]].pop(p, None)
    if not mask[1]:
        g[p].pop(e[1], None)


def ok2addavedge(e, p, g, g2):
    """
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param g2:
    :type g2:
    
    :returns: 
    :rtype: boolean
    """
    if p[1] == e[0]:
        if p[0] != p[1] and p[0] != e[2] and not (e[2] in g2[p[0]] and g2[p[0]][e[2]] in (2, 3)):
            return False
        if p[0] == p[1] and not (e[2] in g2[e[1]] and g2[e[1]][e[2]] in (2, 3)):
            return False
        if p[0] == e[1] and not (e[2] in g2[e[1]] and g2[e[1]][e[2]] in (2, 3)):
            return False

    if p[0] == e[0]:
        if p[0] != p[1] and p[1] != e[1] and not (e[1] in g2[p[1]] and g2[p[1]][e[1]] in (2, 3)):
            return False
        if p[0] == p[1] and not (e[2] in g2[e[1]] and g2[e[1]][e[2]] in (2, 3)):
            return False
        if p[1] == e[2] and not (e[2] in g2[e[1]] and g2[e[1]][e[2]] in (2, 3)):
            return False

    if p[0] == e[1] and p[1] == e[2] and not (e[2] in g2[e[1]] and g2[e[1]][e[2]] in (2, 3)):
        return False
    if p[0] == e[2] and not (e[1] in g2[p[1]] and g2[p[1]][e[1]] in (1, 3)):
        return False
    if p[1] == e[1] and not (e[2] in g2[p[0]] and g2[p[0]][e[2]] in (1, 3)):
        return False
    if p[0] == p[1] == e[0] and not (e[2] in g2[e[1]] and g2[e[1]][e[2]] in (2, 3)):
        return False

    if not edge_increment_ok(e[0], p[0], e[1], g, g2):
        return False
    if not edge_increment_ok(e[0], p[1], e[2], g, g2):
        return False

    return True


def addavedge(g, v, b):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param v:
    :type v:
    
    :param b:
    :type b:
    
    :returns: 
    :rtype: 
    """
    mask = maskavedge(g, v, b)
    g[v[0]][b[0]] = g[v[0]][b[1]] = g[b[0]][v[1]] = g[b[1]][v[2]] = 1
    return mask


def delavedge(g, v, b, mask):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param v:
    :type v:
    
    :param b:
    :type b:
    
    :param mask:
    :type mask:
    """
    if not mask[0]:
        g[v[0]].pop(b[0], None)
    if not mask[1]:
        g[v[0]].pop(b[1], None)
    if not mask[2]:
        g[b[0]].pop(v[1], None)
    if not mask[3]:
        g[b[1]].pop(v[2], None)


def ok2addaAedge(e, p, g, g2):
    """
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param g2:
    :type g2:
    
    :returns: 
    :rtype: boolean
    """
    if p[1] == e[1] and not (p[0] in g2[e[2]] and g2[e[2]][p[0]] in (1, 3)):
        return False
    if p[0] == e[2] and not (p[1] in g2[e[1]] and g2[e[1]][p[1]] in (1, 3)):
        return False

    if not edge_increment_ok(e[1], p[0], e[3], g, g2):
        return False
    if not edge_increment_ok(e[2], p[1], e[3], g, g2):
        return False

    return True


def addaAedge(g, v, b):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param v:
    :type v:
    
    :param b:
    :type b:
    
    :returns: 
    :rtype: 
    """
    mask = maskaAedge(g, v, b)
    g[v[1]][b[0]] = g[v[2]][b[1]] = g[b[0]][v[3]] = g[b[1]][v[3]] = 1
    return mask


def delaAedge(g, v, b, mask):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param v:
    :type v:
    
    :param b:
    :type b:
    
    :param mask:
    :type mask:
    """
    if not mask[0]:
        g[v[1]].pop(b[0], None)
    if not mask[1]:
        g[v[2]].pop(b[1], None)
    if not mask[2]:
        g[b[0]].pop(v[3], None)
    if not mask[3]:
        g[b[1]].pop(v[3], None)


def cleanedges(e, p, g, mask):
    """
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param mask:
    :type mask:
    """
    i = 0
    for m in mask:
        if not m[0]:
            g[e[i + 1]].pop(p[i], None)
        if not m[1]:
            g[p[i]].pop(e[i + 2], None)
        i += 1


def cleanVedges(g, e, p, mask):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :param mask:
    :type mask:
    """
    if mask:
        if not mask[0]:
            g[e[0]].pop(p[0], None)
        if not mask[1]:
            g[p[-1]].pop(e[1], None)

        i = 0
        for m in mask[2:]:
            if not m:
                g[p[i]].pop(p[i + 1], None)
            i += 1


def ok2addapath(e, p, g, g2):
    """
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param g2:
    :type g2:
    
    :returns: 
    :rtype: boolean
    """
    mask = []
    for i in range(len(p)):
        if not edge_increment_ok(e[i + 1], p[i], e[i + 2], g, g2):
            cleanedges(e, p, g, mask)
            return False
        mask.append(add2edges(g, (e[i + 1], e[i + 2]), p[i]))
    cleanedges(e, p, g, mask)
    return True


def ok2addaVpath(e, p, g, g2, rate=2):
    """
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param g2:
    :type g2:
    
    :param rate:
    :type rate:
    
    :returns: 
    :rtype: boolean
    """
    mask = addaVpath(g, e, p)
    if not gk.isedgesubset(bfu.undersample(g, rate), g2):
        cleanVedges(g, e, p, mask)
        return False
    cleanVedges(g, e, p, mask)
    return True


def addapath(g, v, b):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param v:
    :type v:
    
    :param b:
    :type b:
    
    :returns: 
    :rtype: 
    """
    mask = maskapath(g, v, b)
    for i in range(len(b)):
        g[v[i + 1]][b[i]] = g[b[i]][v[i + 2]] = 1

    return mask


def addaVpath(g, v, b):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param v:
    :type v:
    
    :param b:
    :type b:
    
    :returns: 
    :rtype: 
    """
    mask = maskaVpath(g, v, b)
    l = [v[0]] + list(b) + [v[1]]
    for i in range(len(l) - 1):
        g[l[i]][l[i + 1]] = 1
    return mask


def delaVpath(g, v, b, mask):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param v:
    :type v:
    
    :param b:
    :type b:
    
    :param mask:
    :type mask:
    
    :returns: 
    :rtype: 
    """
    cleanVedges(g, v, b, mask)


def delapath(g, v, b, mask):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param v:
    :type v:
    
    :param b:
    :type b:
    
    :param mask:
    :type mask:
    """
    for i in range(len(b)):
        if not mask[2 * i]:
            g[v[i + 1]].pop(b[i], None)
        if not mask[2 * i + 1]:
            g[b[i]].pop(v[i + 2], None)


def prunepaths_1D(g2, path, conn):
    """
    :param g2:
    :type g2:
    
    :param path:
    :type path:
    
    :param conn:
    :type conn:
    
    :returns: 
    :rtype: 
    """
    c = []
    g = cloneempty(g2)
    for p in conn:
        mask = addapath(g, path, p)
        if gk.isedgesubset(bfu.increment(g), g2):
            c.append(tuple(p))
        delapath(g, path, p, mask)
    return c


def ok2addacedge(e, p, g, g2):
    """
    :param e:
    :type e:
    
    :param p:
    :type p:
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param g2:
    :type g2:
    
    :returns: 
    :rtype: boolean
    """
    if p[0] == p[1]:
        if not e[2] in g2[e[2]]:
            return False
        if not p[0] in g2[p[0]]:
            return False
        if not (e[3] in g2[e[1]] and g2[e[1]][e[3]] in (1, 3)):
            return False

    if not edge_increment_ok(e[1], p[0], e[2], g, g2):
        return False
    if not edge_increment_ok(e[2], p[1], e[3], g, g2):
        return False

    return True


def addacedge(g, v, b):  # chain
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param v:
    :type v:
    
    :param b:
    :type b:
    
    :returns: 
    :rtype: 
    """
    mask = maskaCedge(g, v, b)
    g[v[1]][b[0]] = g[v[2]][b[1]] = g[b[0]][v[2]] = g[b[1]][v[3]] = 1
    return mask


def delacedge(g, v, b, mask):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param v:
    :type v:
    
    :param b:
    :type b:
    
    :param mask:
    :type mask:
    """
    if not mask[0]:
        g[v[1]].pop(b[0], None)
    if not mask[1]:
        g[b[0]].pop(v[2], None)
    if not mask[2]:
        g[v[2]].pop(b[1], None)
    if not mask[3]:
        g[b[1]].pop(v[3], None)


def esig(l, n):
    """
    Turns edge list into a hash string
    
    :param l:
    :type l:
    
    :param n:
    :type n:
    
    :returns: 
    :rtype: integer
    """
    z = len(str(n))
    n = map(lambda x: ''.join(map(lambda y: str(y).zfill(z), x)), l)
    n.sort()
    n = ''.join(n[::-1])
    return int('1' + n)


def gsig(g):
    """
    Turns input graph ``g`` into a hash string using edges
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: 
    :rtype:
    """
    return g2num(g)


def signature(g, edges):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param edges:
    :type edges:
    
    :returns: 
    :rtype:
    """
    return (gsig(g), esig(edges, len(g)))


def supergraphs_in_eq(g, g2, rate=1):
    """
    Find  all supergraphs of ``g``  that are also in  the same equivalence
    class with respect to g2 and the rate.
    Currently works only for bfu.undersample by 1
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param g2:
    :type g2:
    
    :param rate:
    :type rate:
    
    :returns: 
    :rtype:
    """
    if bfu.undersample(g, rate) != g2:
        raise ValueError('g is not in equivalence class of g2')

    s = set()

    def addnodes(g, g2, edges):
        if edges:
            masks = []
            for e in edges:
                if ok2addanedge(e[0], e[1], g, g2, rate=rate):
                    masks.append(True)
                else:
                    masks.append(False)
            nedges = [edges[i] for i in range(len(edges)) if masks[i]]
            n = len(nedges)
            if n:
                for i in range(n):
                    mask = addanedge(g, nedges[i])
                    s.add(g2num(g))
                    addnodes(g, g2, nedges[:i] + nedges[i + 1:])
                    delanedge(g, nedges[i], mask)

    edges = gk.edgelist(gk.complement(g))
    addnodes(g, g2, edges)
    return s


def edge_function_idx(edge):
    """
    :param edge:
    :type edge:
    
    :returns: 
    :rtype:
    """
    return min(4, len(edge)) - 2 + min(max(3, len(edge)) - 3, 1) * int(edge[0])


def memo_no_return(func):
    """
    :param func:
    :type func:
    
    :returns: 
    :rtype:
    """
    cache = {}                        # Stored subproblem solutions

    @wraps(func)                      # Make wrap look like func
    def wrap(*args):                  # The memoized wrapper
        s = signature(args[0], args[2])  # Signature: g and edges
        if s not in cache:            # Not already computed?
            cache[s] = func(*args)    # Compute & cache the solution
        return
    return wrap


def v2g22g1(g2, capsize=None, verbose=True):
    """
    Computes all g1 that are in the equivalence class for g2
    
    :param g2:
    :type g2:
    
    :param capsize:
    :type capsize:
    
    :param verbose:
    :type verbose:
    
    :returns: 
    :rtype:
    """
    if bfu.is_sclique(g2):
        print('Superclique - any SCC with GCD = 1 fits')
        return set([-1])

    f = [(add2edges, del2edges, mask2edges),
         (addavedge, delavedge, maskavedge),
         (addacedge, delacedge, maskaCedge),
         (addaAedge, delaAedge, maskaAedge),
         (addapath, delapath, maskapath)]
    c = [ok2add2edges,
         ok2addavedge,
         ok2addacedge,
         ok2addaAedge,
         ok2addapath]

    def predictive_check(g, g2, pool, checks_ok, key):
        s = set()
        for u in pool:
            if not checks_ok(key, u, g, g2):
                continue
            s.add(u)
        return s

    @memo_no_return  # memoize the search
    def nodesearch(g, g2, order, inlist, s, cds, pool, pc):
        if order:
            if bfu.increment(g) == g2:
                s.add(g2num(g))
                if capsize and len(s) > capsize:
                    raise ValueError('Too many elements')
                s.update(supergraphs_in_eq(g, g2))
                return g

            key = order[0]
            if pc:
                tocheck = [x for x in pc if x in cds[len(inlist) - 1][inlist[0]]]
            else:
                tocheck = cds[len(inlist) - 1][inlist[0]]

            if len(order) > 1:
                kk = order[1]
                pc = predictive_check(g, g2, pool[len(inlist)],
                                      c[edge_function_idx(kk)], kk)
            else:
                pc = set()

            adder, remover, masker = f[edge_function_idx(key)]
            checks_ok = c[edge_function_idx(key)]

            for n in tocheck:
                if not checks_ok(key, n, g, g2):
                    continue
                masked = np.prod(masker(g, key, n))
                if masked:
                    nodesearch(g, g2, order[1:], [n] + inlist, s, cds, pool, pc)
                else:
                    mask = adder(g, key, n)
                    nodesearch(g, g2, order[1:], [n] + inlist, s, cds, pool, pc)
                    remover(g, key, n, mask)

        elif bfu.increment(g) == g2:
            s.add(g2num(g))
            if capsize and len(s) > capsize:
                raise ValueError('Too many elements')
            return g

    # find all directed g1's not conflicting with g2

    startTime = time.time()
    gg = checkable(g2)

    idx = np.argsort([len(gg[x]) for x in gg.keys()])
    keys = [gg.keys()[i] for i in idx]

    cds, order, idx = conformanceDS(g2, gg, keys)
    if verbose:
        print("precomputed in {:10.3f} seconds".format(time.time() - startTime))
    if 0 in [len(x) for x in order]:
        return set()
    g = cloneempty(g2)

    s = set()
    try:
        nodesearch(g, g2, [keys[i] for i in idx], [0], s, cds, order, set())
    except ValueError as e:
        print(e)
        s.add(0)
    return s


def conformanceDS(g2, gg, order, f=[], c=[]):
    """
    :param g2:
    :type g2:
    
    :param gg:
    :type gg:
    
    :param order:
    :type order:
    
    :param f:
    :type f:
    
    :param c:
    :type c:
    
    :returns: 
    :rtype:
    """
    CDS = {}
    CDS[0] = set(gg[order[0]])
    pool = [set(gg[order[i]]) for i in range(len(order))]

    for x in itertools.combinations(range(len(order)), 2):

        d, s_i1, s_i2 = inorder_check2(order[x[0]], order[x[1]],
                                       pool[x[0]], pool[x[1]],
                                       g2, f=f, c=c)

        pool[x[0]] = pool[x[0]].intersection(s_i1)
        pool[x[1]] = pool[x[1]].intersection(s_i2)

        d = del_empty(d)
        if not x[1] in CDS:
            CDS[x[1]] = {}
            CDS[x[1]][x[0]] = d
        else:
            CDS[x[1]][x[0]] = d
    if density(g2) > 0.35:
        itr3 = [x for x in itertools.combinations(range(len(order)), 3)]
        for x in random.sample(itr3, min(10, np.int(scipy.special.comb(len(order), 3)))):
            s1, s2, s3 = check3(order[x[0]], order[x[1]], order[x[2]],
                                pool[x[0]], pool[x[1]], pool[x[2]],
                                g2, f=f, c=c)

            pool[x[0]] = pool[x[0]].intersection(s1)
            pool[x[1]] = pool[x[1]].intersection(s2)
            pool[x[2]] = pool[x[2]].intersection(s3)

    return prune_sort_CDS(CDS, pool)


def prune_sort_CDS(cds, pool):
    """
    :param cds:
    :type cds:
    
    :param pool:
    :type pool:
    
    :returns: 
    :rtype:
    """
    idx = np.argsort([len(x) for x in pool])
    p = [pool[i] for i in idx]

    ds = {}
    ds[0] = {}
    ds[0][0] = pool[idx[0]]

    for i in range(1, len(idx)):
        ds[i] = {}
        for j in range(i):
            if idx[j] > idx[i]:
                dd = invertCDSelement(cds[idx[j]][idx[i]])
            else:
                dd = cds[idx[i]][idx[j]]
            for e in pool[idx[i - 1]].intersection(dd.keys()):
                ds[i][e] = pool[idx[i]].intersection(dd[e])

    return ds, p, idx


def invertCDSelement(d_i):
    """
    :param d_i:
    :type d_i:
    
    :returns: 
    :rtype:
    """
    d = {}
    for e in d_i:
        for v in d_i[e]:
            if v in d:
                d[v].add(e)
            else:
                d[v] = set([e])
    return d


def length_d_paths(G, s, d):
    """
    Iterate over nodes in G reachable from s in exactly d steps
    
    :param G:
    :type G:
    
    :param s:
    :type s:
    
    :param d:
    :type d:
    
    :returns: 
    :rtype:
    """
    def recurse(G, s, d, path=[]):
        if d == 0:
            yield path
            return

        for u in G[s]:
            if G[s][u] == 2 or u in path:
                continue
            for v in recurse(G, u, d - 1, path + [u]):
                yield v

    for u in recurse(G, s, d, [s]):
        yield u


def edge_increment_ok(s, m, e, g, g2):
    """
    :param s: start
    :type s:
    
    :param m: middle
    :type m:
    
    :param e : end
    :type e:
    
    :param g:
    :type g:
    
    :param g2:
    :type g2:
    
    :returns: 
    :rtype:
    """
    dd = {1: True, 2: False, 3: True}
    bd = {1: False, 2: True, 3: True}
    # bidirected edges
    for u in g[s]:
        if u != m and not (m in g2[u] and bd[g2[u][m]]):
            return False

    # directed edges
    if s == e:
        if not (m in g2[m] and dd[g2[m][m]]):
            return False
        if not (s in g2[s] and dd[g2[s][s]]):
            return False
    for u in g[m]:
        if not (u in g2[s] and dd[g2[s][u]]):
            return False
        # bidirected edges
        if u != e and not (e in g2[u] and bd[g2[u][e]]):
            return False
    for u in g[e]:
        if not (u in g2[m] and dd[g2[m][u]]):
            return False

    for u in g:
        if s in g[u] and not (m in g2[u] and dd[g2[u][m]]):
            return False
        if m in g[u] and not (e in g2[u] and dd[g2[u][e]]):
            return False

    return True


def length_d_loopy_paths(G, s, dt, p):
    """
    Iterate over nodes in ``G`` reachable from ``s`` in exactly d steps
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :param s:
    :type s:
    
    :param dt:
    :type dt:
    
    :param p:
    :type p:
    
    :returns: 
    :rtype:
    """
    g1 = cloneempty(G)

    def recurse(g, g2, s, d, path=[]):

        if edge_increment_ok(p[-d - 2], s, p[-d - 1], g, g2):

            if d == 0:
                yield path
                return

            mask = add2edges(g, (p[-d - 2], p[-d - 1]), s)
            for u in g2[s]:
                if g2[s][u] == 2:
                    continue
                for v in recurse(g, g2, u, d - 1, path + [u]):
                    yield v
            del2edges(g, (p[-d - 2], p[-d - 1]), s, mask)

    for u in recurse(g1, G, s, dt - 1, [s]):
        yield u
