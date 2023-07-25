""" This module contains graph format conversion functions """
from __future__ import print_function
from networkx.algorithms.components import condensation, strongly_connected_components
import networkx as nx
import numpy as np
import igraph
import sys


def g2num(g):
    """ 
    Convert a graph into a long int 

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: unique number for each graph considering only directed edges
    :rtype: long integer
    """
    n = len(g)
    num = ['0']*n*n
    for v in range(1, n + 1):
        idx = (v-1)*n
        for w in g[v]:
            num[idx + (w-1)] = '1'

    return int(''.join(num), 2)


def ug2num(g):
    """
    Convert non-empty edges into a tuple of (directed, bidriected) in
    binary format
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: unique number for each graph considering directed and bidirected in binary format
    :rtype: a tuple of binary integer
    """
    n = len(g)
    n2 = n ** 2 + n
    num = 0
    mask = 0
    num2 = 0
    for v in g:
        for w in g[v]:
            if g[v][w] in (1, 3):
                mask = (1 << (n2 - v * n - w))
                num |= mask
            if g[v][w] in (2, 3):
                num2 |= mask
    return num, num2


def bg2num(g):
    """
    Convert bidirected edges into a binary format
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: unique number for each graph considering bidirected in binary format
    :rtype: a tuple of binary integer
    """
    n = len(g)
    n2 = n ** 2 + n
    num = 0
    for v in g:
        for w in g[v]:
            if g[v][w] in (2, 3):
                num = num | (1 << (n2 - v * n - w))
    return num


def graph2nx(G):
    """
    Convert a ``gunfolds`` graph to NetworkX format ignoring bidirected edges
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: NetworkX format graph
    :rtype: NetworkX graph
    """
    g = nx.DiGraph()
    for v in G:
        edges = [(v, x) for x in G[v] if G[v][x] in (1, 3)]
        if edges:
            g.add_edges_from(edges)
        else:
            g.add_node(v)
    return g


def graph2dot(g, filename):
    """
    Save the graph structure of `g` to a graphviz format dot file with the name `filename`

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)

    :param filename: name of the file
    :type filename: string
    """
    G = graph2nx(g)
    nx.drawing.nx_pydot.write_dot(G, filename)


def nx2graph(G):
    """
    Convert NetworkX format graph to ``gunfolds`` graph ignoring bidirected edges
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: ``gunfolds`` graph
    :rtype: dictionary (``gunfolds`` graphs)
    """
    g = {n: {} for n in G}
    for n in G:
        g[n] = {x: 1 for x in G[n]}
    return g


def num2CG(num, n):
    """
    Converts a number  whose binary representaion encodes edge
    presence/absence into a compressed graph representaion
    
    :param num: unique graph representation in numbers
    :type num: integer
    
    :param n: number of nodes
    :type n: integer
    
    :returns: ``gunfolds`` graph 
    :rtype: dictionary (``gunfolds`` graphs)
    """
    s = bin(num)[2:].zfill(n*n)
    g = {i+1: {} for i in range(n)}
    for v in g:
        for w in range(n):
            if s[(v-1)*n:(v-1)*n+n][w] == '1':
                g[v][w+1] = 1
    return g


def dict_format_converter(H):
    """ Convert a graph from the set style dictionary format to the integer style
    
        :param H: set style dictionary format
        :type H: dictionary 
        
        :returns: ``gunfolds`` graph 
        :rtype: dictionary (``gunfolds`` graphs)

        >>> test = {'1': {'1': {(0, 1)},
        ...   '2': {(0, 1), (2, 0)},
        ...   '3': {(0, 1), (2, 0)},
        ...   '4': {(2, 0)},
        ...   '5': {(0, 1)}},
        ...  '2': {'1': {(2, 0)}, '2': {(0, 1)}, '5': {(0, 1), (2, 0)}},
        ...  '3': {'1': {(0, 1), (2, 0)}, '2': {(0, 1)}, '5': {(0, 1)}},
        ...  '4': {'1': {(2, 0)},
        ...   '2': {(0, 1)},
        ...   '3': {(0, 1)},
        ...   '4': {(0, 1)},
        ...   '5': {(0, 1)}},
        ...  '5': {'1': {(0, 1)}, '2': {(0, 1), (2, 0)}, '5': {(0, 1)}}}
        >>> dict_format_converter(test)
        {1: {1: 1, 2: 3, 3: 3, 4: 2, 5: 1}, 2: {1: 2, 2: 1, 5: 3}, 3: {1: 3, 2: 1, 5: 1}, 4: {1: 2, 2: 1, 3: 1, 4: 1, 5: 1}, 5: {1: 1, 2: 3, 5: 1}}
        >>>
    """
    H_new = {}
    for vert_a in H:
        H_new[int(vert_a)] = {}
        for vert_b in H[vert_a]:
            edge_val = 0
            if (0, 1) in H[vert_a][vert_b]:
                edge_val = 1
            if (2, 0) in H[vert_a][vert_b]:
                edge_val = 2 if edge_val == 0 else 3
            if edge_val:
                H_new[int(vert_a)][int(vert_b)] = edge_val
    return H_new


def g2ian(g):
    """
    (Ask)

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: ``gunfolds`` graph 
    :rtype: dictionary (``gunfolds`` graphs)
    """
    return dict_format_converter(g)


def ian2g(g):
    """
    (Ask)

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: (Ask)
    :rtype: (Ask)
    """
    c = {1: {(0, 1)}, 2: {(2, 0)}, 3: {(0, 1), (2, 0)}}
    gg = {}
    for w in g:
        gg[str(w)] = {}
        for v in g[w]:
            gg[str(w)][str(v)] = c[g[w][v]]
    return gg


# Adjacency matrix functions

def graph2adj(G):
    """ 
    Convert the directed edges to an adjacency matrix 
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: graph adjacency matrix for directed edges
    :rtype: numpy matrix
    """
    n = len(G)
    A = np.zeros((n, n), dtype=np.int8)
    for v in G:
        A[int(v) - 1, [int(w)-1 for w in G[v] if G[v][w] in (1, 3)]] = 1
    return A


def graph2badj(G):
    """ 
    Convert the bidirected edges to an adjacency matrix 
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: graph adjacency matrix for bidirected edges
    :rtype: numpy matrix
    """
    n = len(G)
    A = np.zeros((n, n), dtype=np.int8)
    for v in G:
        A[int(v) - 1, [int(w)-1 for w in G[v] if G[v][w] in (2, 3)]] = 1
    return A


def adjs2graph(directed, bidirected):
    """ 
    Convert an adjacency matrix of directed and bidirected edges to a graph
    
    :param directed: graph adjacency matrix for directed edges
    :type directed: numpy matrix

    :param bidirected: graph adjacency matrix for bidirected edges
    :type bidirected: numpy matrix
    
    :returns: ``gunfolds`` format graph
    :rtype: dictionary (``gunfolds`` graphs)
    """
    G = {i: {} for i in range(1, directed.shape[0] + 1)}
    for i in range(directed.shape[0]):
        for j in np.where(directed[i, :] == 1)[0] + 1:
            G[i + 1][j] = 1

    for i in range(bidirected.shape[0]):
        for j in range(bidirected.shape[1]):
            if bidirected[i, j] and j != i:
                if j + 1 in G[i + 1]:
                    G[i + 1][j + 1] = 3
                else:
                    G[i + 1][j + 1] = 2
    return G


def g2vec(g):
    """
    Converts ``gunfolds`` graph to a vector

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: a vector representing a ``gunfolds`` graph 
    :rtype: numpy vector
    """
    A = graph2adj(g)
    B = graph2badj(g)
    return np.r_[A.flatten(), B[np.triu_indices(B.shape[0], k=1)]]


def vec2adj(v, n):
    """
    Converts a vector representation to adjacency matrix

    :param v: vector representation of ``gunfolds`` graph
    :type v: numpy vector
    
    :param n: number of nodes
    :type n: integer
    
    :returns: a tuple of Adjacency matrices
    :rtype: a tuple of numpy matrices
    """
    A = np.zeros((n, n))
    B = np.zeros((n, n))
    A[:] = v[:n ** 2].reshape(n, n)
    B[np.triu_indices(n, k=1)] = v[n ** 2:]
    B = B + B.T
    return A, B


def vec2g(v, n):
    """
    Converts a vector representation to ``gunfolds`` graph 

    :param v: vector representation of ``gunfolds`` graph
    :type v: numpy vector
    
    :param n: number of nodes
    :type n: integer
    
    :returns: ``gunfolds`` format graph
    :rtype: dictionary (``gunfolds`` graphs)
    """
    A, B = vec2adj(v, n)
    return adjs2graph(A, B)


def rate(u):
    """
    Converts under sampling rate to ``clingo`` predicate

    :param u: maximum under sampling rate
    :type u: integer
    
    :returns: ``clingo`` predicate for under sampling rate
    :rtype: string
    """
    s = "u(1.."+str(u)+")."
    return s


def clingo_preamble(g):
    """
    Converts number of nodes into a ``clingo`` predicate

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: ``clingo`` predicate
    :rtype: string 
    """
    s = ''
    n = len(g)
    s += '#const n = '+str(n)+'. '
    s += 'node(1..n). '
    return s


def g2clingo(g, directed='hdirected', bidirected='hbidirected', both_bidirected=False, preamble=True):
    """ Convert a graph to a string of grounded terms for clingo
    
        :param g: ``gunfolds`` graph
        :type g: dictionary (``gunfolds`` graphs)
        
        :param directed: name of the variable for directed edges in the observed graph
        :type directed: string

        :param bidirected: name of the variable for  bidirected edges in the observed graph
        :type bidirected: string
        
        :param both_bidirected: (Ask)
        :type both_bidirected: boolean
        
        :param preamble: (Ask)
        :type preamble: boolean
        
        :returns: ``clingo`` predicate
        :rtype: string 
        
        .. code-block:: 
        
           Example: {1:{3:1,4:2,5:3}}
           "1": node 1 has an edge with node 3 => edge(1,3).
           "2": node 1 has an conf with node 4 => conf(1,4).
           "3": node 1 has both edge and conf with node 5 => edge(1,5). conf(1,5).
         
     """
    s = ''
    if preamble:
        s += clingo_preamble(g)
    for v in g:
        for w in g[v]:
            if both_bidirected:
                direction = True
            else:
                direction = v < w
            if g[v][w] & 1:
                s += directed+'('+str(v)+','+str(w)+'). '
            if g[v][w] & 2 and direction:
                s += bidirected+'('+str(v)+','+str(w)+'). '
    return s


def numbered_g2clingo(g, n, directed='hdirected', bidirected='hbidirected'):
    """ Convert a graph to a string of grounded terms for clingo
    
        :param g: ``gunfolds`` graph
        :type g: dictionary (``gunfolds`` graphs)
        
        :param n: number of nodes
        :type n: integer
        
        :param directed: name of the variable for directed edges in the observed graph
        :type directed: string

        :param bidirected: name of the variable for  bidirected edges in the observed graph
        :type bidirected: string
        
        :returns: ``clingo`` predicate
        :rtype: string 
    
         .. code-block:: 

               Example: {1:{3:1,4:2,5:3}}
               "1": node 1 has an edge with node 3 => edge(1,3).
               "2": node 1 has an conf with node 4 => conf(1,4).
               "3": node 1 has both edge and conf with node 5 => edge(1,5). conf(1,5).
           
     """
    s = ''
    for v in g:
        for w in g[v]:
            if g[v][w] & 1:
                s += directed+'('+str(v)+','+str(w)+','+str(n)+'). '
            if g[v][w] & 2 and v < w:
                s += bidirected+'('+str(v)+','+str(w)+','+str(n)+'). '
    return s


def clingo_wedge(x, y, w, n, name='edge'):
    """
    Returns ``clingo`` predicate for weighted edge

    :param x: outgoing edge
    :type x: integer
    
    :param y: incoming edge
    :type y: integer
    
    :param w: weight
    :type w: integer
    
    :param n: number of nodes
    :type n: integer

    :param name: name of the variable for ``clingo``
    :type name: string
    
    :returns: ``clingo`` predicate for weighted edge
    :rtype: string
    """
    edge = name+'('+str(x)+', '+str(y)+', '+str(w)+', '+str(n)+'). '
    return edge


def numbered_g2wclingo(g, num, directed_weights_matrix=None, bidirected_weights_matrix=None,
                       directed='hdirected', bidirected='hbidirected'):
    """
    Convert a graph to a string of grounded terms for ``clingo``
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)

    :param num: index of the graph in the resulting clingo command
    :type num: integer
    
    :param directed_weights_matrix: directed weight matrix
    :type directed_weights_matrix: numpy matrices 
    
    :param bidirected_weights_matrix: bidirected weight matrix
    :type bidirected_weights_matrix: numpy matrices

    :param directed: name of the directed edges in the observed graph
    :type directed: string

    :param bidirected: name of the bidirected edges in the observed
        graph
    :type bidirected: string
    
    :returns: ``clingo`` predicate
    :rtype: string  
    """
    s = ''
    n = len(g)

    if directed_weights_matrix is None:
        directed_weights_matrix = np.ones((n, n))
    directed_weights_matrix = directed_weights_matrix.astype('int')

    if bidirected_weights_matrix is None:
        bidirected_weights_matrix = np.ones((n, n))
    bidirected_weights_matrix = bidirected_weights_matrix.astype('int')

    assert directed_weights_matrix.shape[0] == n
    assert directed_weights_matrix.shape[1] == n
    assert bidirected_weights_matrix.shape[0] == n
    assert bidirected_weights_matrix.shape[1] == n

    for v in range(1, n+1):
        for w in range(1, n+1):
            i = v-1
            j = w-1
            missing = [clingo_wedge(v, w, bidirected_weights_matrix[i, j], num, name='no_'+bidirected),
                       clingo_wedge(v, w, directed_weights_matrix[i, j], num, name='no_'+directed)]
            if w in g[v]:
                if g[v][w] & 1:
                    s += clingo_wedge(v, w, directed_weights_matrix[i, j], num, name=directed)
                    missing = [missing[0], '']
                if g[v][w] & 2:
                    s += clingo_wedge(v, w, bidirected_weights_matrix[i, j], num, name=bidirected)
                    missing = ['', missing[1]]
            s += ' '.join(missing)
    return s


def g2wclingo(g):
    """
    Convert a graph to a string of grounded terms for ``clingo``
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: ``clingo`` predicate
    :rtype: string  
    """
    s = ''
    n = len(g)
    s += 'node(1..'+str(n)+'). '
    for v in range(1, n+1):
        for w in range(1, n+1):
            missing = ['no_confh('+str(v)+','+str(w)+', 1).',
                       'no_edgeh('+str(v)+','+str(w)+', 1).']
            if w in g[v]:
                if g[v][w] & 1:
                    s += 'edgeh('+str(v)+','+str(w)+', 1). '
                    missing = [missing[0]]
                if g[v][w] & 2:
                    s += 'confh('+str(v)+','+str(w)+', 1). '
                    missing = [missing[1]]
            s += ' '.join(missing)
    return s


def clingo2num(value):
    """
    Converts the output of ``clingo`` into list of edges and under sampling rates for ``drasl``

    :param value: output of ``clingo``
    :type value: string
    
    :returns: list of edges and under sampling rates
    :rtype: a tuple of lists
    """
    a2edgetuple(value)


def rasl_a2edgetuple(answer):
    """
    Returns list of edges and the under sampling rate for ``rasl``

    :param answer: output of ``clingo``
    :type answer: string
    
    :returns: list of edges and the under sampling rate
    :rtype: a tuple of list and an integer
    """
    edges = [x for x in answer if 'edge' in x]
    u = [x for x in answer if 'min' in x]
    if not u:
        u = [x for x in answer if 'trueu' in x]
    u = u[0].split('(')[1].split(')')[0]
    return edges, int(u)


def a2edgetuple(answer):
    """
    Converts the output of ``clingo`` into list of edges and under sampling rates for ``drasl``
 
    :param answer: output of ``clingo``
    :type answer: string
    
    :returns: list of edges and under sampling rates
    :rtype: a tuple of lists
    """
    edges = [x for x in answer if 'edge1' in x]
    u = [x for x in answer if x[0] == 'u']
    return edges, u


def rasl_c2edgepairs(clist):
    """
    Converts ``clingo`` predicates to edge pairs for ``rasl``

    :param clist: ``clingo`` predicates
    :type clist: list of strings
    
    :returns: list of edge pairs
    :rtype: list
    """
    return [x[5:-1].split(',') for x in clist]


def c2edgepairs(clist):
    """
    Converts ``clingo`` predicates to edge pairs for ``drasl``

    :param clist: ``clingo`` predicates
    :type clist: list of strings
    
    :returns: list of edge pairs
    :rtype: list
    """
    return [x.strip(' ')[6:-1].split(',') for x in clist]


def nodenum(edgepairs):
    """
    Returns the number of nodes in the graph

    :param edgepairs: list of edge pairs
    :type edgepairs: list
    
    :returns: number of nodes in the graph
    :rtype: integer
    """
    nodes = 0
    for e in edgepairs:
        nodes = np.max([nodes, int(e[0]), int(e[1])])
    return nodes


def edgepairs2g(edgepairs):
    """
    Converts edge pairs to a ``gunfolds`` graph

    :param edgepairs: list of edge pairs
    :type edgepairs: list
    
    :returns: ``gunfolds`` graph
    :rtype: dictionary (``gunfolds`` graph)
    """
    n = nodenum(edgepairs)
    g = {x+1: {} for x in range(n)}
    for e in edgepairs:
        g[int(e[0])][int(e[1])] = 1
    return g


def msl_jclingo2g(output_g):
    """
    Converts the output of ``clingo`` to ``gunfolds`` graph for ``rasl_msl``

    :param output_g: the output of ``clingo`` for ``rasl_msl``
    :type output_g: string
    
    :returns: ``gunfolds`` graph
    :rtype: dictionary (``gunfolds`` graph)
    """
    l = a2edgetuple(output_g)
    l = (c2edgepairs(l[0]), l[1][0])
    l = (g2num(edgepairs2g(l[0])), int(l[1][2:-1]))
    return l


def rasl_jclingo2g(output_g):
    """
    Converts the output of ``clingo`` to ``gunfolds`` graph for ``rasl``

    :param output_g: the output of ``clingo`` for ``rasl``
    :type output_g: string
    
    :returns: ``gunfolds`` graph
    :rtype: dictionary (``gunfolds`` graph)
    """
    l = rasl_a2edgetuple(output_g)
    l = (rasl_c2edgepairs(l[0]), l[1])
    l = (g2num(edgepairs2g(l[0])), l[1])
    return l


def drasl_jclingo2g(output_g):
    """
    Converts the output of ``clingo`` to ``gunfolds`` graph for ``drasl``

    :param output_g: the output of ``clingo`` for ``drasl``
    :type output_g: string
    
    :returns: ``gunfolds`` graph
    :rtype: dictionary (``gunfolds`` graph)
    """
    l = a2edgetuple(output_g)
    l = (c2edgepairs(l[0]), tuple(np.sort([int(x.split(',')[0][2:]) for x in l[1]])))
    l = (g2num(edgepairs2g(l[0])), l[1])
    return l


def old_g2clingo(g, file=sys.stdout):
    """
    (Ask)

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param file: (Ask)
    :type file:
    """
    n = len(g)
    print('node(1..'+str(n)+').', file=file)
    for v in g:
        for w in g[v]:
            if g[v][w] == 1:
                print('edgeu('+str(v)+','+str(w)+').', file=file)
            if g[v][w] == 2:
                print('confu('+str(v)+','+str(w)+').', file=file)
            if g[v][w] == 3:
                print('edgeu('+str(v)+','+str(w)+').', file=file)
                print('confu('+str(v)+','+str(w)+').', file=file)


def g2ig(g):
    """
    Converts our graph representation to an igraph for plotting
    
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: igraph representation of ``gunfolds`` graph
    :rtype: igraph
    """
    t = np.where(graph2adj(g) == 1)
    l = zip(t[0], t[1])
    ig = igraph.Graph(l, directed=True)
    ig.vs["name"] = np.sort([u for u in g])
    ig.vs["label"] = ig.vs["name"]
    return ig


def nxbp2graph(G):
    """
    Ask 

    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: Ask
    :rtype: 
    """
    nodesnum = len(G)//2
    g = {n+1: {} for n in range(nodesnum)}
    for n in g:
        g[n] = {(x % nodesnum+1): 1 for x in G[n-1]}
    return g


def encode_sccs(g, idx, components=True, SCCS=None):
    """
    Encodes strongly connected components of ``gunfolds`` graph to ``clingo`` predicates 

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param idx: index of the graph
    :type idx: integer
    
    :param components: If True, encodes SCC components and memberships to ``clingo`` predicates
    :type components: boolean
    
    :param SCCS: SCC membership of nodes 
    :type SCCS: list
    
    :returns: ``clingo`` predicates 
    :rtype: string
    """
    G = graph2nx(g)
    if SCCS is None:
        SCCS = strongly_connected_components(G)
    CG = condensation(G, scc=SCCS)
    s = ''
    for v in CG:
        for w in CG[v]:
            s += 'dag(' + str(v) + ', ' + str(w) + ', ' + str(idx) + '). '
    if not components:
        return s
    for c, component in enumerate(SCCS):
        cl = len(component)
        if cl == 1:
            x = [x for x in component][0]
            if x in G[x]:
                cl = 2
        s += 'sccsize(' + str(c) + ', ' + str(cl) + '). '
        for node in component:
            s += 'scc(' + str(node) + ', ' + str(c) + '). '
    return s


def encode_list_sccs(glist, scc_members=None):
    """
    Encodes strongly connected components of a list of ``gunfolds`` graph to ``clingo`` predicates 

    :param glist: a list of graphs that are under sampled versions of
        the same system
    :type glist: list of dictionaries (``gunfolds`` graphs)

    :param scc_members: a list of dictionaries for nodes in each SCC
    :type scc_members: list
    
    :returns: ``clingo`` predicates 
    :rtype: string
    """
    s = ''
    first_graph = True
    for i, g in enumerate(glist):
        if first_graph:
            if scc_members is not None:
                SCCS = scc_members
            else:
                SCCS = [s for s in strongly_connected_components(graph2nx(g))]
        s += encode_sccs(g, i+1, components=first_graph, SCCS=SCCS)
        first_graph = False
    # if the generating graph has an edge between non singleton SCCs that are
    # not connected in any of the measured graphs - no go
    s += ':- edge1(X,Y), scc(X,K), scc(Y,L), K != L, sccsize(L,Z), Z > 1, not dag(K,L,_). '
    # if the produced graph has a cycle connecting 2 SCCs - no go
    s += ':- directed(X,Y,M), directed(Y,X,N), scc(X, K), scc(Y,L), K != L, M<=U, N<=U, M<=N, u(U,_).'
    # if there is an edge between SCCs in the produced graph and none in the measured for nonsingleton SCCs - no go
    s += ':- directed(X,Y,U), scc(X,K), scc(Y,L), K != L, sccsize(L,Z), Z > 1, not dag(K,L,N), u(U,N).'
    return s
