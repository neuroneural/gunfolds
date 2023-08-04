# tools to construct (random) graphs
from gunfolds.conversions import nx2graph, nxbp2graph, graph2nx
from gunfolds.utils import ecj
from itertools import combinations
import networkx as nx
from networkx.utils.random_sequence import powerlaw_sequence
import numpy as np
import random


def edgelist(g):  # directed
    """
    Returns a list of tuples for edges of ``g``
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :returns: a list of tuples for edges of ``g``
    :rtype: list of tuples
    """
    l = []
    for n in g:
        l.extend([(n, e) for e in g[n] if g[n][e] in (1, 3)])
    return l


def inedgelist(g):  # missing directed iterator
    """
    Iterate over the list of tuples for edges of ``g``
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    """
    n = len(g)
    for v in g:
        for i in range(1, n + 1):
            if not (i in g[v]):
                yield v, i
            elif g[v][i] not in (1, 3):
                yield v, i


def inbedgelist(g):  # missing bidirected iterator
    """
    Iterate over the list of tuples for edges of ``g``
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    """
    for v in g:
        for w in g:
            if v != w:
                if not (w in g[v]):
                    yield v, w
                elif g[v][w] not in (2, 3):
                    yield v, w


def bedgelist(g):
    """ 
    Bidirected edge list with flips 
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :returns: a list of tuples for bidirected edges of ``g``
    :rtype: list of tuples
    """
    l = []
    for n in g:
        l.extend([tuple(sorted((n, e))) for e in g[n] if g[n][e] in (2, 3)])
    l = list(set(l))
    l = l + list(map(lambda x: (x[1], x[0]), l))
    return l


def undedgelist(g, exclude_bi=False):
    """
    Returns a list of tuples for undirected edges of ``g`` with nodes
    sorted in ascending order.
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :param exclude_bi: if True, only converts directed edges to undirected
        edges. If False, converts directed and bidirected edges to undirected edges
    :type exclude_bi: boolean
    
    :returns: a list of tuples for undirected edges of ``g`` with nodes
              sorted in ascending order
    :rtype: list
    """
    l = []
    if exclude_bi:
        for n in g:
            l.extend([(n, e) if n <= e else (e, n) for e in g[n] if g[n][e] in (1, 3)])
    else:
        for n in g:
            l.extend([(n, e) if n <= e else (e, n) for e in g[n] if g[n][e] in (1, 2, 3)])
    return set(l)


def superclique(n):
    """ 
    Returns a Graph with all possible edges
    
    :param n: number of nodes
    :type n: integer
    
    :returns: a Graph with all possible edges
    :rtype: dictionary (``gunfolds`` graph)
    """
    g = {}
    for i in range(n):
        g[i + 1] = {j + 1: 3 for j in range(n) if j != i}
        g[i + 1][i + 1] = 1
    return g


def fullyconnected(n):
    """
    Returns a graph with all possible directed edges

    :param n: number of nodes
    :type n: integer

    :returns: a graph with all possible directed edges
    :rtype: dictionary(``gunfolds`` graph)
    """
    g = {}
    for i in range(n):
        g[i + 1] = {j + 1: 1 for j in range(n)}
    return g


def complement(G):
    """ 
    Returns the complement of ``G``
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graph)
    
    :returns: the complement of ``G``
    :rtype: dictionary (``gunfolds`` graph)
    """
    n = len(G)
    sq = superclique(n)
    for v in G:
        for w in G[v]:
            sq[v][w] = sq[v][w] - G[v][w]
            if sq[v][w] == 0:
                del sq[v][w]
    return sq


def gtranspose(G):
    """ 
    Transpose (rev. edges of) ``G``
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graph)
    
    :returns: Transpose of given graph
    :rtype: dictionary (``gunfolds`` graph)
    """
    GT = {u: {} for u in G}
    for u in G:
        for v in G[u]:
            if G[u][v] in (1, 3):
                GT[v][u] = 1        # Add all reverse edges
    return GT


def scale_free(n, alpha=0.7, beta=0.25, delta_in=0.2, delta_out=0.2):
    """
    (Copied from scale_free function in networkx should i need to add any citation)

    :param n: number of nodes
    :type n: integer
    
    :param alpha: probability for adding a new node connected to an existing node
    :type alpha: float
    
    :param beta: probability for adding an edge between two existing nodes.
    :type beta: float
    
    :param delta_in: bias for choosing nodes from in-degree
    :type delta_in: float
    
    :param delta_out: bias for choosing nodes from out-degree distribution.
    :type delta_out: float
    
    :returns: 
    :rtype: 
    """
    g = nx.scale_free_graph(n, alpha=alpha,
                            beta=beta,
                            delta_in=delta_in, delta_out=delta_out)
    g = nx2graph(g)
    g = gtranspose(g)
    addAring(g)
    return g


def randH(n, d1, d2):
    """ 
    Generate a random H with ``n`` nodes 
    
    :param n: number of nodes
    :type n: integer
    
    :param d1: number of additional edges to the ring
    :type d1: integer
    
    :param d2: (ask) number of random permutations
    :type d2: integer
    
    :returns: a random graph with ``n`` nodes 
    :rtype: dictionary (``gunfolds`` graph)
    """
    g = ringmore(n, d1)
    pairs = [x for x in combinations(g.keys(), 2)]
    for p in np.random.permutation(pairs)[:d2]:
        g[p[0]][p[1]] = g[p[0]].get(p[1], 0) + 2
        g[p[1]][p[0]] = g[p[1]].get(p[0], 0) + 2
    return g


def ring(n):
    """
    Create a Ring Graph with ``n`` number of nodes

    :param n: number of nodes
    :type n: integer
    
    :returns: a Ring Graph with ``n`` number of nodes
    :rtype: dictionary (``gunfolds`` graph)
    """
    g = {}
    for i in range(1, n):
        g[i] = {i + 1: 1}
    g[n] = {1: 1}
    return g


def addAring(g):
    """
    Add a ring to ``g`` in place
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    """
    for i in range(1, len(g)):
        if g[i].get(i + 1) == 2:
            g[i][i + 1] = 3
        else:
            g[i][i + 1] = 1
    if g[i].get(1) == 2:
        g[i][1] = 3
    else:
        g[i][1] = 1


def upairs(n, k):
    """
    Returns ``n`` unique nonsequential pairs
    
    :param n: (ask)
    :type n: integer
    
    :param k: (ask)
    :type k: integer
    
    :returns: n unique nonsequential pairs
    :rtype: list
    """
    s = set()
    for p in np.random.randint(n, size=(3 * k, 2)):
        if p[1] - p[0] == 1:
            continue
        s.add(tuple(p))
    return list(s)[:k]


def ringarcs(g, n):
    """
    (Ask)

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :param n: (ask)
    :type n: integer
    
    :returns: 
    :rtype: dictionary (``gunfolds`` graph)
    """
    for edge in upairs(len(g), n):
        g[edge[0] + 1][edge[1] + 1] = 1
    return g


def ringmore(n, m):
    """
    Returns a ``n`` node ring graph with ``m`` additional edges

    :param n: number of nodes
    :type n: integer
    
    :param m: number of additional edges
    :type m: integer
    
    :returns: a ``n`` node ring graph with ``m`` additional edges
    :rtype: dictionary (``gunfolds`` graph)
    """
    return ringarcs(ring(n), m)


def digonly(H):
    """   
    Returns a subgraph of ``H`` contatining all directed edges of ``H``

    :param H: ``gunfolds`` graph
    :type H: dictionary (``gunfolds`` graph)
    
    :returns: a subgraph of ``H`` contatining all directed edges of ``H``
    :rtype: dictionary (``gunfolds`` graph)
    """

    g = {n: {} for n in H}
    for v in g:
        g[v] = {w: 1 for w in H[v] if not H[v][w] == 2}
    return g


def _OCE(g1, g2):
    """
    Omission/commision error of ``g1`` referenced to ``g2``
    
    :param g1: the graph to check
    :type g1: dictionary (``gunfolds`` graph)
    
    :param g2: the ground truth graph
    :type g2: dictionary (``gunfolds`` graph)
    
    :returns: Omission/commision error for directed and bidirected edges
    :rtype: dictionary
    """
    s1 = set(edgelist(g1))
    s2 = set(edgelist(g2))
    omitted = len(s2 - s1)
    comitted = len(s1 - s2)

    s1 = set(bedgelist(g1))
    s2 = set(bedgelist(g2))
    bomitted = len(s2 - s1)//2
    bcomitted = len(s1 - s2)//2

    return {'directed': (omitted, comitted),
            'bidirected': (bomitted, bcomitted),
            'total': (omitted+bomitted, comitted+bcomitted)}


def _normed_OCE(g1, g2):
    """
    Return omission and comission errors for directed and
    bidirected edges.

    Omission error is normalized by the number of edges present
    in the ground truth. Commision error is normalized by the
    number of possible edges minus the number of edges present
    in the ground truth.
    
    :param g1: the graph to check
    :type g1: dictionary (``gunfolds`` graph)
    
    :param g2: the ground truth graph
    :type g2: dictionary (``gunfolds`` graph)
    
    :returns: normalized omission and comission errors for directed and
              bidirected edges.
    :rtype: dictionary
    """
    def sdiv(x, y):
        if y < 1.:
            return 0.
        return x/y

    n = len(g2)
    gt_DEN = float(len(edgelist(g2)))  # number of d  edges in GT
    gt_BEN = 0.5 * len(bedgelist(g2))  # number of bi edges in GT
    DEN = n*n                          # all posible directed edges
    BEN = n*(n-1)/2                    # all possible bidirected edges
    err = OCE(g1, g2)
    nerr = {'directed': (sdiv(err['directed'][0], gt_DEN),
                         sdiv(err['directed'][1], (DEN - gt_DEN))),
            'bidirected': (sdiv(err['bidirected'][0], gt_BEN),
                           sdiv(err['bidirected'][1], (BEN - gt_BEN))),
            'total': (sdiv((err['directed'][0]+err['bidirected'][0]), (gt_DEN+gt_BEN)),
                      sdiv((err['directed'][1]+err['bidirected'][1]),
                           (DEN+BEN - gt_BEN - gt_DEN)))
            }
    return nerr


def _undirected_OCE(g1, g2):
    """
    Returns omission/commision error of ``g1`` referenced to ``g2``
    if both are undirected graphs.
    
    :param g1: the graph to check
    :type g1: dictionary (``gunfolds`` graph)
    
    :param g2: the ground truth graph
    :type g2: dictionary (``gunfolds`` graph)
    
    :returns: omission and comission errors for undirected edges
    :rtype: dictionary
    """
    
    s1 = set(undedgelist(g1))
    s2 = set(undedgelist(g2))
    omitted = len(s2 - s1)
    comitted = len(s1 - s2)
    return {'undirected': (omitted, comitted)}


def _normed_undirected_OCE(g1, g2):
    """
    Return omission and comission errors for undirected edges.

    Omission error is normalized by the number of edges present
    in the ground truth. Commision error is normalized by the
    number of possible edges minus the number of edges present
    in the ground truth.

    :param g1: the graph to check
    :type g1: dictionary (``gunfolds`` graph)
    
    :param g2: the ground truth graph
    :type g2: dictionary (``gunfolds`` graph)
    
    :returns: omission and comission errors for normalized undirected edges
    :rtype: dictionary
    """

    def sdiv(x, y):
        if y < 1.:
            return 0.
        return x / y

    n = len(g2)
    gt_DEN = float(len(undedgelist(g2)))  # number of undirected  edges in GT
    DEN = n * n + n  # all posible undirected edges
    err = _undirected_OCE(g1, g2)
    nerr = {'undirected': (err['undirected'][0]/gt_DEN,
            sdiv(err['undirected'][1], (DEN - gt_DEN)))}
    return nerr


def OCE(g1, g2, normalized=False, undirected=False):
    """
    Return omission and comission errors for graphs.

    :param g1: the graph to check
    :type g1: dictionary (``gunfolds`` graph)
    
    :param g2: the ground truth graph
    :type g2: dictionary (``gunfolds`` graph)
    
    :param normalized: If True, returns normalized error and vice versa
    :type normalized: boolean
    
    :param undirected: If True, returns undirected error and vice versa
    :type undirected: boolean
    
    :returns: omission and comission errors for graphs
    :rtype: dictionary
    """
    if normalized:
        if undirected:
            err = _normed_undirected_OCE(g1, g2)
        else:
            err = _normed_OCE(g1, g2)
    else:
        if undirected:
            err = _undirected_OCE(g1, g2)
        else:
            err = _OCE(g1, g2)
    return err


def clean_leaf_nodes(g):
    """
    Removes leaf_nodes of the given graph ``g``

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    """
    for v in g:
        g[v] = {w: g[v][w] for w in g[v] if g[v][w] > 0}


def cerror(d):
    """
    Returns normalized comission error

    :param d: calculated error
    :type d: dictionary
    
    :returns: normalized comission error
    :rtype: float
    """
    return d['OCE']['directed'][1] / np.double(len(d['gt']['graph']) ** 2 - len(edgelist(d['gt']['graph'])))


def oerror(d):
    """
    Returns normalized omission error

    :param d: calculated error
    :type d: dictionary
    
    :returns: normalized omission error
    :rtype: float
    """
    return d['OCE']['directed'][0] / np.double(len(edgelist(d['gt']['graph'])))


def bidirected_no_fork(g):
    """
    (Ask)

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :returns: 
    :rtype: boolean
    """
    be = bedgelist(g)
    T = gtranspose(g)
    for e in be:
        if not set(T[e[0]].keys()) & set(T[e[1]].keys()):
            return True
    return False


def no_parents(g):
    """
    Checks if there exists a node that has no parents in the given graph ``g``.

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :returns: True, if there exists a node with no parents. False, otherwise.
    :rtype: boolean
    """
    T = gtranspose(g)
    for n in T:
        if not T[n]:
            return True
    return False


def no_children(g):
    """
    Checks if there exists a node that has no children in the given graph ``g``.

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :returns: True, if there exists a node with no children. False, otherwise.
    :rtype: boolean
    """
    for n in g:
        if not g[n]:
            return True
    return False


def scc_unreachable(g):
    """
    Checks if there exists a strongly connected component in the given graph ``g`` that is unreachable.

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :returns: True, if there exists SCC that is unreachable. False, otherwise.
    :rtype: boolean
    """
    if bidirected_no_fork(g):
        return True
    if no_parents(g):
        return True
    if no_children(g):
        return True
    return False

# unlike functions from traversal package these do no checking


def addanedge(g, e):
    """
    Adds an edge ``e`` from the given graph ``g``

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :param e: edge to be added
    :type e: pair of integers
    """
    g[e[0]][e[1]] = 1


def delanedge(g, e):
    """
    Deletes an edge ``e`` from the given graph ``g``

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :param e: edge to be deleted
    :type e: pair of integers
    """
    g[e[0]].pop(e[1], None)


def addedges(g, es):
    """
    Adds the edges in the list ``es`` for given graph ``g``.

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :param es: list of edges to be added
    :type es: list of pairs of integers
    """
    for e in es:
        addanedge(g, e)


def deledges(g, es):
    """
    Deletes the edges in the list ``es`` for given graph ``g``.

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :param es: list of edges to be deleted
    :type es: list of pairs of integers
    """
    for e in es:
        delanedge(g, e)


def isdedgesubset(g2star, g2):
    """
    Checks if ``g2star`` directed edges are a subset of those of ``g2``
    
    :param g2star: ``gunfolds`` graph to be checked
    :type g2star: dictionary (``gunfolds`` graph)
    
    :param g2:  ``gunfolds`` graph
    :type g2: dictionary (``gunfolds`` graph)
    
    :returns: True, if ``g2star`` directed edges are a subset of those of ``g2`` and vice versa
    :rtype: boolean
    """
    for n in g2star:
        for h in g2star[n]:
            if h in g2[n]:
                # if g2star has a directed edge and g2 does not
                if g2star[n][h] in (1, 3) and g2[n][h] == 2:
                    return False
            else:
                return False
    return True


def isedgesubset(g2star, g2):
    """
    Checks if all ``g2star`` edges are a subset of those of ``g2``
    
    :param g2star: ``gunfolds`` graph to be checked
    :type g2star: dictionary (``gunfolds`` graph)
    
    :param g2:  ``gunfolds`` graph
    :type g2: dictionary (``gunfolds`` graph)
    
    :returns: True, if all ``g2star`` edges are a subset of those of ``g2`` and vice versa
    :rtype: boolean
    """
    for n in g2star:
        for h in g2star[n]:
            if h in g2[n]:
                # Everything is a subset of 3 (both edge types)
                if g2[n][h] != 3:
                    # Either they both should have a directed edge, or
                    # both should have a bidirected edge
                    if g2star[n][h] != g2[n][h]:
                        return False
            else:
                return False
    return True


def degree_ring(n, d):
    """
    Generate a ring graph with ``n`` nodes and average degree ``d``
    
    :param n: number of nodes
    :type n: integer
    
    :param d: degree
    :type d: integer
    
    :returns: a ring graph with ``n`` nodes and average degree ``d``
    :rtype: dictionary(``gunfolds`` graph)
    """
    g = nx.expected_degree_graph([d-1]*n)
    gn = nx2graph(g)
    addAring(gn)
    return gn


def density(g):
    """
    Returns the density of the directed part of the graph ``g``.

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :returns: the density of the directed part of the graph
    :rtype: float
    """
    return len(edgelist(g)) / np.double(len(g) ** 2)


def udensity(g):
    """
    Returns the density of a undersampled graph that simultaneously accounts for directed and bidirected edges of ``g``.

    :param g: `gunfolds` graph
    :type g: dictionary (`gunfolds` graph)
    
    :returns: the density of the given undersampled graph
    :rtype: float
    """
    return (len(edgelist(g))+len(bedgelist(g))/2) / np.double(len(g)**2 + len(g)*(len(g)-1)/2)


def mean_degree_graph(node_num, degree):
    """
    Generates a random graph with ``node_num``, nodes and mean outgoing degree of ``degree``.

    :param node_num: number of nodes
    :type node_num: integer
    
    :param degree: degree
    :type degree: integer
    
    :returns: a random graph with mean outgoing degree of the given graph
    :rtype: dictionary(``gunfolds`` graph)
    """
    g = nx.fast_gnp_random_graph(node_num, degree/node_num, directed=True)
    g = nx2graph(g)
    return g


def pow_degree_graph(node_num, degree):
    """
    Generates a graph by powerlaw sequence of ``degree`` constructed using the Havel-Hakimi algorithm.

    :param node_num: number of nodes
    :type node_num: integer
    
    :param degree: degree
    :type degree: integer
    
    :returns: a graph constructed using the Havel-Hakimi algorithm.
    :rtype: dictionary(``gunfolds`` graph)
    """
    while True:
        try:
            sequence = powerlaw_sequence(node_num, exponent=degree)
            g = nx.havel_hakimi_graph([int(x) for x in sequence])
            break
        except nx.NetworkXError:
            continue
    g = nx2graph(g)
    return g


def bp_mean_degree_graph(node_num, degree, seed=None):
    """
    Generates a random bipartite graph with ``node_num``, nodes and mean outgoing degree (``degree``)

    :param node_num: number of nodes
    :type node_num: integer
    
    :param degree: degree
    :type degree: float
    
    :param seed: random seed
    :type seed: integer
    
    :returns: a random bipartite graph with ``node_num``, nodes and mean outgoing degree of ``degree``
    :rtype: dictionary(``gunfolds`` graph)
    """
    G = nx.bipartite.random_graph(node_num, node_num, degree/node_num, seed=seed)
    g = nxbp2graph(G)
    return g


# this function does not work yet - WIP
def bp_pow_degree_graph(node_num, degree, prob=0.7):
    """
    Generates a bipartite graph by powerlaw sequence of ``degree`` constructed using the Havel-Hakimi algorithm.

    :param node_num: number of nodes
    :type node_num: integer
    
    :param degree: degree
    :type degree: integer
    
    :param prob: probability 
    :type prob: float
    
    :returns: a bipartite graph constructed using the Havel-Hakimi algorithm.
    :rtype: dictionary(``gunfolds`` graph)
    """
    while True:
        try:
            sequence = powerlaw_sequence(node_num, exponent=degree)
            g = nx.bipartite.preferential_attachment_graph([int(x) for x in sequence], prob)
            break
        except nx.NetworkXError:
            continue
    g = nxbp2graph(g)
    return g


def remove_tril_singletons(T):
    """
    Ensure that the DAG resulting from this matrix will not have
    singleton nodes not connected to anything.

    :param T: lower triangular matrix representing a DAG
    :type T: numpy array
    
    :returns: adjacency matrix where no singleton nodes are connected to anything
    :rtype: numpy array 
    """
    N = T.shape[0]
    neighbors = T.sum(0) + T.sum(1)
    idx = np.where(neighbors == 0)
    if idx:
        for i in idx[0]:
            v1 = i
            while i == v1:
                v1 = np.random.randint(0, N-1)
            if i > v1:
                T[i][v1] = 1
            elif i < v1:
                T[v1][i] = 1
    return T


def randomTRIL(N, degree=5, connected=False):
    """
    Generate a random triangular matrix

    https://stackoverflow.com/a/56514463
    
    :param N: size of the matrix
    :type N: integer
    
    :param degree: degree
    :type degree: integer
    
    :param connected: (Ask)
    :type connected: boolean
    
    :returns: a random triangular adjacency matrix
    :rtype: numpy array 
    """
    mat = [[0 for x in range(N)] for y in range(N)]
    for _ in range(N):
        for j in range(degree):
            v1 = np.random.randint(0, N-1)
            v2 = np.random.randint(0, N-1)
            if v1 > v2:
                mat[v1][v2] = 1
            elif v1 < v2:
                mat[v2][v1] = 1
    mat = np.asarray(mat, dtype=np.uint8)
    if connected:
        mat = remove_tril_singletons(mat)
    return mat


def randomDAG(N, degree=5, connected=True):
    """
    Generates a random DAG

    :param N: number of nodes
    :type N: integer
    
    :param degree: degree
    :type degree: integer
    
    :param connected: If true, returns a connected DAG
    :type connected: boolean
    
    :returns: a random DAG
    :rtype: NetworkX graph
    """
    adjacency_matrix = randomTRIL(N, degree=degree,
                                  connected=connected)
    rows, cols = np.where(adjacency_matrix == 1)
    edges = zip(rows.tolist(), cols.tolist())
    gr = nx.DiGraph()
    gr.add_edges_from(edges)
    if connected:
        components = [x for x in nx.algorithms.components.weakly_connected_components(gr)]
        if len(components) > 1:
            for component in components[1:]:
                v1 = random.choice(tuple(components[0]))
                v2 = random.choice(tuple(component))
                gr.add_edge(v1, v2)
    assert nx.is_directed_acyclic_graph(gr)
    assert nx.algorithms.components.is_weakly_connected(gr)
    return gr


def shift_labels(g, shift):
    """
    Returns same graph with shifted labels

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :param shift: number of vertices to be shifted
    :type shift: integer
    
    :returns: same graph with shifted labels
    :rtype: dictionary(``gunfolds`` graph)
    """
    new_g = {}
    for v in g:
        new_g[v+shift] = {}
        for w in g[v]:
            new_g[v+shift][w+shift] = g[v][w]
    return new_g


def shift_list_labels(glist):
    """
    Shifts the labels for a list of graphs

    :param glist: a list of graphs that are undersampled versions of
        the same system
    :type glist: list of dictionaries (``gunfolds`` graphs)
    
    :returns: shifted list of graphs
    :rtype: list
    """
    if len(glist) < 2:
        return glist
    components = [glist[0]]
    shift = len(glist[0])
    for i in range(1, len(glist)):
        components.append(shift_labels(glist[i], shift))
        shift += len(glist[i])
    return components


def merge_list(glist):
    """
    Merge the graphs in the list

    :param glist: a list of graphs that are undersampled versions of
        the same system
    :type glist: list of dictionaries (``gunfolds`` graphs)
    
    :returns: merged graph of a list
    :rtype: dictionary(``gunfolds`` graph)
    """
    g = {}
    for e in glist:
        g.update(e)
    return g


def subgraph(g, nodes):
    """
    Returns a subgraph of ``g`` that consists of ``nodes`` and their
    interconnections.

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)

    :param nodes: integer valued nodes to include
    :type nodes: list
    
    :returns: a subgraph of ``g`` that consists of ``nodes`` and their
              interconnections.
    :rtype: dictionary(``gunfolds`` graph)
    """
    nodes = set(nodes)
    sg = {}
    for node in nodes:
        sg[node] = {x: g[node][x] for x in g[node]}
    return sg


def gcd4scc(SCC):
    # first check if there is at least a single simple loop
    """
    Returns the greatest common divisor of the strongly connected component

    :param SCC: a strongly connected component
    :type SCC: dictionary (``gunfolds`` graph)
    
    :returns: the greatest common divisor of the strongly connected component
    :rtype: integer
    """
    x = np.sum([selfloop(_, SCC) for _ in SCC])
    if x > 0:
        return 1
    g = graph2nx(SCC)
    return ecj.listgcd([len(x) for x in nx.simple_cycles(g)])


def ensure_gcd1(scc):
    """
    If ``scc``'s loop structure is not ``gcd=1`` pick a random node and
    add a self-loop

    :param scc: a strongly connected component
    :type scc: dictionary (``gunfolds`` graph)
    
    :returns: a graph with ``gcd=1``
    :rtype: dictionary (``gunfolds`` graph)
    """
    if gcd4scc(scc) > 1:
        a = random.choice([*scc])
        scc[a][a] = 1
    return scc


def update_graph(g, g2):
    """
    Update ``g`` with connections (or nodes) from ``g2``. Both must be at ``u=1``
    
    :param g: graph to be updated
    :type g: dictionary (``gunfolds`` graph)
    
    :param g2: reference graph
    :type g2: dictionary (``gunfolds`` graph)
    
    :returns: updated ``g`` with respect to ``g2``
    :rtype: dictionary (``gunfolds`` graph)
    """
    for v in g2:
        if v in g:
            g[v].update(g2[v])
        else:
            g[v] = g2[v].copy()
    return g


def merge_graphs(glist):
    """
    Merge a list of graphs at ``u=1`` into a single new graph ``g``
    
    :param glist: a list of graphs that are undersampled versions of
        the same system
    :type glist: list of dictionaries (``gunfolds`` graphs)
    
    :returns: a new  graph by merging a list of graphs at ``u=1``
    :rtype: dictionary (``gunfolds`` graph)
    """
    g = {}
    for subg in glist:
        update_graph(g, subg)
    return g


def ensure_graph_gcd1(g, ignore_singletons=True):
    """
    This function takes any graph, breaks it into SCCs and make sure each SCC has a gcd of 1

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)

    :param ignore_singletons: ignores singleton SCCs when adding a self-loop to make gcd=1
    :type ignore_singletons: boolean

    :returns: a graph with ``gcd=1``
    :rtype: dictionary (``gunfolds`` graph)
    """
    G = graph2nx(g)
    if ignore_singletons:
        x = [ensure_gcd1(subgraph(g, c)) for c in nx.strongly_connected_components(G) if len(c) > 1]
    else:
        x = [ensure_gcd1(subgraph(g, c)) for c in nx.strongly_connected_components(G)]
    return merge_graphs([g] + x)


def gcd1_bp_mean_degree_graph(node_num, degree, seed=None):
    """
    Returns a random graph with ``gcd=1``

    :param node_num: number of nodes
    :type node_num: integer
    
    :param degree: degree
    :type degree: float
    
    :param seed: random seed
    :type seed: integer
    
    :returns: a random graph with ``gcd=1``
    :rtype: dictionary (``gunfolds`` graph)
    """
    g = bp_mean_degree_graph(node_num, degree, seed)
    return ensure_graph_gcd1(g)


def ring_sccs(num, num_sccs, dens=0.5, degree=3, max_cross_connections=3):
    """
    Generate a random graph with ``num_sccs`` SCCs, n-nodes each
    
    :param num: number of nodes in each sec
    :type num: integer
    
    :param num_sccs: number of secs
    :type num_sccs: integer
    
    :param dens: density of each sac
    :type dens: float
    
    :param degree: degree of the connecting DAG
    :type degree: integer
    
    :param max_cross_connections: maximum number of connections per each edge in DAG
    :type max_cross_connections: integer
    
    :returns: a random graph with ``num_sccs`` SCCs, n-nodes each
    :rtype: dictionary (``gunfolds`` graph)
    """
    dag = randomDAG(num_sccs, degree=degree)
    while nx.is_empty(dag):
        dag = randomDAG(num_sccs, degree=degree)
    if not nx.is_weakly_connected(dag):
        randTree = nx.random_tree(n=num_sccs, create_using=nx.DiGraph)
        dag = remove_loop(nx.compose(dag, randTree), dag=dag)
    ss = shift_list_labels([ensure_gcd1(ringmore(num, max(int(dens * num**2)-num, 1))) for i in range(num_sccs)])
    for v in dag:
        v_nodes = [x for x in ss[v].keys()]
        for w in dag[v]:
            w_nodes = [x for x in ss[w].keys()]
            for i in range(np.random.randint(low=1, high=max_cross_connections+1)):
                a = random.choice(v_nodes)
                b = random.choice(w_nodes)
                ss[v][a][b] = 1
    return merge_list(ss)


def selfloop(n, g):
    """
    Checks if ``g`` has a self loop at node n

    :param n: node
    :type n: integer
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :returns: True if ``g`` has a self loop at node n and vice versa
    :rtype: boolean
    """
    return n in g[n]


def remove_loop(G, dag):
    """  
    Removes the loops from ``G`` according to ``dag``

    :param G: graph
    :type G: NetworkX graph
    
    :param dag: DAG
    :type dag: NetworkX graph
    
    :returns: graph with removed loop according to DAG
    :rtype: NetworkX graph
    """
    try:
        while True:
            lis = nx.find_cycle(G)
            for edge in lis:
                if edge in list(dag.edges):
                    G.remove_edge(edge[0], edge[1])
                    break
    except nx.exception.NetworkXNoCycle:
        assert nx.is_directed_acyclic_graph(G)
        assert nx.is_weakly_connected(G)
        return G
