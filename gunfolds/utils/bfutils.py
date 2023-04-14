from gunfolds.conversions import g2num, ug2num, num2CG
import gunfolds.utils.zickle as zkl
import gunfolds.utils.graphkit as gk
import itertools
import numpy as np


# Undersampling functions
def is_sclique(G):
    """
    Tests if the graph is a superclique (all possible connections)

    :param G: ``gunfolds`` format graph
    :type G:  dictionary (``gunfolds`` graphs)

    :returns: True, if ``G`` is a super clique
    :rtype: boolean
    """
    n = len(G)
    row_sum = 3 * (n-1) + 1    # rows will have all connections plus a self loop
    for vertex in G:
        if G[vertex].get(vertex) != 1 or sum(G[vertex].values()) != row_sum:
            return False
    return True


def directed_inc(G, D):
    """ 
    A helper function for determining directed edges in an undersampled
    graph given a previously undersampled graph 
        
    :param G: ``gunfolds`` format graph G_1 - without undersampling
    :type G:  dictionary (``gunfolds`` graphs)
    
    :param D: ``gunfolds`` graph G_u a version of G_1 undersampled by u
    :type D:  dictionary (``gunfolds`` graphs)

    :returns:
    :rtype: dictionary (``gunfolds`` graphs)
    """
    G_un = {}
    # directed edges
    for vert1 in D:
        G_un[vert1] = {}
        for vert2 in D[vert1]:
            if G[vert2] and D[vert1][vert2] in (1, 3):
                for e in G[vert2]:
                    G_un[vert1][e] = 1
    return G_un


def bidirected_inc(G, D):
    """ 
    A helper function for determining bidirected edges in an undersampled
    graph given a previously undersampled graph 
    
    :param G: ``gunfolds`` format graph G_1 - without undersampling
    :type G:  dictionary (``gunfolds`` graphs)
    
    :param D: ``gunfolds`` graph G_u a version of G_1 undersampled by u
    :type D:  dictionary (``gunfolds`` graphs)

    :returns:
    :rtype:
    """
    for vert1 in G:
        # transfer old bidirected edges
        for vert2 in D[vert1]:
            if D[vert1][vert2] in (2, 3):
                G[vert1][vert2] = 2 if G[vert1].get(vert2, 2) == 2 else 3
        # new bidirected edges
        edges = [e for e in D[vert1] if D[vert1][e] in (1, 3)]
        for pair in itertools.permutations(edges, 2):
            G[pair[0]][pair[1]] = 2 if G[pair[0]].get(pair[1], 2) == 2 else 3
    return G


def increment_u(G_star, G_u):
    """ 
    Given graph `G_star` - without undersampling - and its undersampled version `G_u`, generate the `G_{u+1}` graph

    :param G_star: ``gunfolds`` format graph G_1 - without undersampling
    :type G_star: dictionary (``gunfolds`` graphs)
    
    :param G_u: ``gunfolds`` graph G_u a version of G_1 undersampled by u
    :type G_u: dictionary (``gunfolds`` graphs)
    
    :returns: 
    :rtype: 
    """
    # directed edges
    G_un = directed_inc(G_star, G_u)
    # bidirected edges
    G_un = bidirected_inc(G_un, G_u)
    return G_un


def pure_directed_inc(G, D):
    """ 
    :param G: ``gunfolds`` format graph G_1 - without undersampling
    :type G:  dictionary (``gunfolds`` graphs)
    
    :param D: ``gunfolds`` graph G_u a version of G_1 undersampled by u
    :type D:  dictionary (``gunfolds`` graphs)
    
    :returns: 
    :rtype: 
    """
    G_un = {}
    # directed edges
    for vert1 in D:
        G_un[vert1] = {}
        for prev_vert in D[vert1]:
            for vert2 in G[prev_vert]:
                G_un[vert1][vert2] = 1
    return G_un


def increment(G):
    """
    Undersample ``G`` by 2
    only works for G1 to G2 directed

    :param G: ``gunfolds`` format graph
    :type G:  dictionary (``gunfolds`` graphs)

    :returns: undersampled graph
    :rtype: dictionary (``gunfolds`` graphs)
    """
    G2 = {n: {} for n in G}

    for vert1 in G:
        for h in G[vert1]:
            for e in G[h]:
                if not (e in G2[vert1]):
                    G2[vert1][e] = 1

    for vert1 in G:
        for pair in itertools.combinations(G[vert1], 2):

            if pair[1] in G2[pair[0]]:
                G2[pair[0]][pair[1]] |= 2
            else:
                G2[pair[0]][pair[1]] = 2

            if pair[0] in G2[pair[1]]:
                G2[pair[1]][pair[0]] |= 2
            else:
                G2[pair[1]][pair[0]] = 2

    return G2


def dincrement_u(G_star, G_u):
    """
    :param G_star: ``gunfolds`` format graph
    :type G_star: dictionary (``gunfolds`` graphs)

    :param G_u:
    :type G_u:

    :returns:
    :rtype:
    """
    # directed edges
    G_un = pure_directed_inc(G_star, G_u)
    return G_un


def undersample(G, u):
    """
    Undersample graph ``G`` by rate ``u``

    :param G: ``gunfolds`` format graph
    :type G:  dictionary (``gunfolds`` graphs)

    :param u: undersampling rate
    :type u: integer

    :returns: undersampled graph
    :rtype: dictionary (``gunfolds`` graphs)
    """
    if u == 1:
        return G
    Gu = G
    for i in range(u-1):
        Gu = increment_u(G, Gu)
    return Gu


def all_undersamples(G_star):
    """ 
    Returns a list of all undersampled graphs (excluding superclique) 
 
    :param G_star: ``gunfolds`` format graph
    :type G_star: dictionary (``gunfolds`` graphs)

    :returns: list of all undersampled graphs (excluding superclique)
    :rtype: list of graphs
    """
    glist = [G_star]
    while True:
        g = increment_u(G_star, glist[-1])
        if is_sclique(g):
            return glist  # superclique convergence
        # this will (may be) capture DAGs and oscillations
        if g in glist:
            return glist
        glist.append(g)
    return glist


def call_undersamples(G_star):
    """ 
    Returns a list of all undersampled graphs (including superclique) 
    
    :param G_star: ``gunfolds`` format graph
    :type G_star: dictionary (``gunfolds`` graphs)

    :returns: list of all undersampled graphs (including superclique)
    :rtype: list of graphs
    """
    glist = [G_star]
    while True:
        g = increment_u(G_star, glist[-1])
        if g in glist:
            return glist
        glist.append(g)
    return glist


def compact_call_undersamples(G_star):
    """ 
    Returns a list of all undersampled graphs (including superclique)
    in binary encoded format 
    
    :param G_star: ``gunfolds`` format graph
    :type G_star: dictionary (``gunfolds`` graphs)

    :returns: a list of all undersampled graphs (including superclique)
              in binary encoded format
    :rtype: list of integers
    """
    glist = [ug2num(G_star)]
    lastgraph = G_star
    while True:
        g = increment_u(G_star, lastgraph)
        n = ug2num(g)
        if n in glist:
            return glist
        glist.append(n)
        lastgraph = g
    return glist


def cc_undersamples(G_star, steps=1):
    """
    Returns ``G_u`` for ``G_star`` with default value of ``u=2``.
    If the input graph converges before the requested ``u`` - an empty list is returned.

    :param G_star: ``gunfolds`` format graph
    :type G_star: dictionary (``gunfolds`` graphs)
    
    :param steps: undersampling rate - 1 for G_star
    :type steps: integer

    :returns:
    :rtype:
    """
    glist = [ug2num(G_star)]
    lastgraph = G_star
    for i in range(steps):
        g = increment_u(G_star, lastgraph)
        n = ug2num(g)
        if n in glist:
            return []
        glist.append(n)
        lastgraph = g
    return glist[-1]


# Misc graph functions

def overshoot(G_star, H):
    """
    Undersample ``G_star`` until it turns into a ``G_u`` for which ``H`` is an edge subset.
    Return ``True`` in this case.
    If this does not happen for either of the undersampling rates until teh convergence - return ``False``

    :param G_star: ``gunfolds`` format graph at ``u=1``
    :type G_star: dictionary (``gunfolds`` graphs)

    :param H: ``gunfolds`` format graph at some ``u``
    :type H: dictionary (``gunfolds`` graphs)

    :returns:
    :rtype: boolean
    """
    glist = [G_star]
    while True:
        g = increment_u(G_star, glist[-1])
        if is_sclique(g):
            return False
        if gk.isedgesubset(H, g):
            return True
        if g in glist:
            return False
        glist.append(g)
    return False


def forms_loop(G_star, loop):
    """
    :param G_star: ``gunfolds`` format graph
    :type G_star: dictionary (``gunfolds`` graphs)

    :param loop:
    :type loop:

    :returns:
    :rtype: boolean
    """
    glist = [G_star]
    while True:
        g = increment_u(G_star, glist[-1])
        if (g2num(gk.digonly(g)) & loop) == loop:
            return True
        if g in glist:
            return False
        glist.append(g)
    return False


def call_u_conflicts_d(G_star, H):
    """
    :param G_star: ``gunfolds`` format graph
    :type G_star: dictionary (``gunfolds`` graphs)

    :param H:
    :type H:

    :returns:
    :rtype: boolean
    """
    glist = [G_star]
    while True:
        g = dincrement_u(G_star, glist[-1])
        if gk.isedgesubset(g, H):
            return False
        if g in glist:
            return True
        glist.append(g)
    return True


def call_u_conflicts(G_star, H):
    """
    :param G_star: ``gunfolds`` format graph
    :type G_star: dictionary (``gunfolds`` graphs)

    :param H:
    :type H:

    :returns:
    :rtype: boolean
    """
    glist = [G_star]
    while True:
        # g = increment_u(G_star, glist[-1])
        g = directed_inc(G_star, glist[-1])
        if gk.isdedgesubset(g, H):
            return False
        g = bidirected_inc(g, glist[-1])
        if gk.isedgesubset(g, H):
            return False
        if g in glist:
            return True
        glist.append(g)
    return True


def call_u_conflicts2(G_star, H):
    """
    :param G_star: ``gunfolds`` format graph
    :type G_star: dictionary (``gunfolds`` graphs)

    :param H:
    :type H:

    :returns:
    :rtype:
    """
    glist = [G_star]
    while True:
        g = increment_u(G_star, glist[-1])
        if gk.isedgesubset(g, H):
            return False, glist
        if g in glist:
            return True, glist
        glist.append(g)
    return True, glist


def call_u_equals2(G_star, glist, H):
    """
    :param G_star: ``gunfolds`` format graph
    :type G_star: dictionary (``gunfolds`` graphs)

    :param glist: a list of graphs that are undersampled versions of
        the same system
    :type glist: list of dictionaries (``gunfolds`` graphs)

    :param H:
    :type H:

    :returns:
    :rtype: boolean
    """
    while True:
        g = increment_u(G_star, glist[-1])
        if g == H:
            return True
        if g in glist:
            return False
        glist.append(g)
    return False


def call_u_equals(G_star, H):
    """
    :param G_star: ``gunfolds`` format graph
    :type G_star: dictionary (``gunfolds`` graphs)

    :param H:
    :type H:

    :returns:
    :rtype: boolean
    """
    glist = [G_star]
    while True:
        g = increment_u(G_star, glist[-1])
        if g == H:
            return True
        if g in glist:
            return False
        glist.append(g)
    return False


def compatible(d1, d2):
    """
    :param d1:
    :type d1:

    :param d2:
    :type d2:

    :returns:
    :rtype:
    """
    idx = np.where(np.array([[r == l for l in d2] for r in d1]))
    return idx


def compat(G):
    """
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)

    :returns:
    :rtype:
    """
    n = len(G)
    # sample all the graph for gStar
    star_l = all_undersamples(G)
    hits = {}
    # brute force all graphs
    for i in range(0, 2 ** (n**2)):
        tmpG = num2CG(i, n)
        tmp_l = all_undersamples(tmpG)
        c = compatible(tmp_l, star_l)
        if len(sum(c)) > 0:
            hits[i] = c
    return hits


def icompat(i, nodes):
    """
    :param i:
    :type i:

    :param nodes:
    :type nodes:

    :returns:
    :rtype:
    """
    print(i)
    g = num2CG(i, nodes)
    return compat(g)


def ilength(i, nodes):
    """
    :param i:
    :type i:

    :param nodes:
    :type nodes:

    :returns:
    :rtype:
    """
    print(i)
    g = num2CG(i, nodes)
    return len(call_undersamples(g))


def iall(i, nodes):
    """
    :param i:
    :type i:

    :param nodes:
    :type nodes:

    :returns:
    :rtype:
    """
    print(i)
    g = num2CG(i, nodes)
    return compact_call_undersamples(g)


def cc_all(i, nodes, steps):
    """
    :param i:
    :type i:

    :param nodes:
    :type nodes:

    :param steps:
    :type steps:

    :returns:
    :rtype:
    """
    g = num2CG(i, nodes)
    return cc_undersamples(g, steps=steps)


def make_rect(l):
    """
    :param l:
    :type l:

    :returns:
    :rtype:
    """
    max_seq = max(map(len, l))
    nl = []
    for e in l:
        e += [e[-1]] * (max_seq - len(e))
        nl.append(e)
    return nl


def loadgraphs(fname):
    """
    Loads a graph from a zickle file

    :param fname: file name
    :type fname: string

    :returns: ``gunfolds`` graph
    :rtype: dictionary (``gunfolds`` graphs)
    """
    g = zkl.load(fname)
    return g


def savegraphs(l, fname):
    """
    Saves a graph into a zickle file format

    :param l: ``gunfolds`` graph
    :type l: dictionary (``gunfolds`` graphs)

    :param fname: file name
    :type fname: string
    """
    zkl.save(l, fname)


# talking about extra edges on top of the ring


def dens2edgenum(d, n=10):
    """
    Convert density into the number of extra edges needed for a ring
    graph to achieve that density

    :param d: density
    :type d: float

    :param n: number of nodes in the graph
    :type n: integer

    :returns: number of extra edges needed for a ring
              graph to achieve that density
    :rtype: integer
    """
    return max(int(d * n**2)-n, 1)


def edgenum2dens(e, n=10):
    """
    Convert number of edges excluding edges of the ring to density

    :param e: number of edges excluding edges of the ring
    :type e: integer

    :param n: number of nodes in the graph
    :type n: integer

    :returns: density
    :rtype: double
    """
    return np.double(e + n)/n**2


def check_conflict(H, G_test, au=None):
    """
    :param H:
    :type H:

    :param G_test:
    :type G_test:

    :param au:
    :type au:

    :returns:
    :rtype:
    """
    if not au:
        allundersamples = call_undersamples(G_test)
    else:
        allundersamples = au
    for graph in allundersamples:
        if gk.isedgesubset(graph, H):
            return False
    return True


def check_conflict_(Hnum, G_test, au=None):
    """
    :param Hnum:
    :type Hnum:

    :param G_test:
    :type G_test:

    :param au:
    :type au:

    :returns:
    :rtype:
    """
    if not au:
        allundersamples = call_undersamples(G_test)
    else:
        allundersamples = au
    # Hnum = ug2num(H)
    for graph in allundersamples:
        gnum = ug2num(graph)
        if gnum[0] & Hnum[0] == gnum[0] and gnum[1] & Hnum[1] == gnum[1]:
            return False
    return True


def check_equality(H, G_test, au=None):
    """
    :param H:
    :type H:

    :param G_test:
    :type G_test:

    :param au:
    :type au:

    :returns:
    :rtype:
    """
    if not au:
        allundersamples = call_undersamples(G_test)
    else:
        allundersamples = au
    for graph in allundersamples:
        if graph == H:
            return True
    return False
