def walk(G, s, S=set()):
    """
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :param s:
    :type s:
    
    :param S:
    :type S:
    
    :returns: 
    :rtype: 
    """
    P, Q = dict(), set()
    P[s] = None
    Q.add(s)
    while Q:
        u = Q.pop()
        for v in G[u].difference(P, S):
            Q.add(v)
            P[v] = u
    return P


def dfs_topsort(G):
    """
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: 
    :rtype: 
    """
    S, res = set(), []

    def recurse(u):
        if u in S:
            return
        S.add(u)
        for v in G[u]:
            recurse(v)
        res.append(u)
    for n in G:
        recurse(n)
    res.reverse()
    return res


def tr(G):                      # Transpose (rev. edges of) G
    """
    Reverses all the edges in the graph

    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: transposed graph
    :rtype: dictionary (``gunfolds`` graphs)
    """
    GT = {}
    for u in G:
        GT[u] = set()   # Get all the nodes in there
    for u in G:
        for v in G[u]:
            GT[v].add(u)        # Add all reverse edges
    return GT


def scc(G):                   # Kosaraju's algorithm
    """
    Returns a list of strongly connected components using Kosaraju's algorithm

    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: a list of strongly connected components using Kosaraju's algorithm
    :rtype: list
    """
    GT = tr(G)                # Get the transposed graph
    sccs, seen = [], set()
    for u in dfs_topsort(G):   # DFS starting points
        if u in seen:
            continue  # Ignore covered nodes
        C = walk(GT, u, seen)  # Don't go "backward" (seen)
        seen.update(C)         # We've now seen C
        sccs.append(C)         # Another SCC found
    return sccs


def cloneBfree(G):
    """
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :returns: 
    :rtype: 
    """
    D = {}
    for v in G:
        D[v] = {}
        for u in G[v]:
            if G[v][u] in (1, 3):
                D[v][u] = 1
    return D


def gcd(a, b):
    """
    Returns greatest common divisor of ``a`` and ``b``

    :param a: first value
    :type a: integer
    
    :param b: second value
    :type b: integer
    
    :returns: greatest common divisor of ``a`` and ``b``
    :rtype: integer
    """
    while b != 0:
        a, b = b, a % b
    return a


def listgcd_r(l):
    """
    Returns greatest common divisor for a given list of numbers using recursion

    :param l: list of values
    :type l: list
    
    :returns: greatest common divisor for a given list of numbers
    :rtype: integer
    """
    if len(l) > 0:
        return gcd(l[0], listgcd_r(l[1:]))
    else:
        return 0


def listgcd(l):
    """
    Returns greatest common divisor for a given list of numbers 

    :param l: list of values
    :type l: list
    
    :returns: greatest common divisor for a given list of numbers
    :rtype: integer 
    """
    mygcd = 0
    if len(l) > 0:
        for x in l:
            mygcd = gcd(x, mygcd)
    return mygcd


def reachable(s, G, g):
    """
    :param s:
    :type s:
    
    :param G: (guess)``gunfolds`` format graph
    :type G: (guess)dictionary (``gunfolds`` graphs)
    
    :param g: (guess)``gunfolds`` graph
    :type g: (guess)dictionary (``gunfolds`` graphs)
    
    :returns: 
    :rtype: boolean
    """
    S, Q = set(), []
    Q.append(s)
    while Q:
        u = Q.pop()
        if u in S:
            continue
        if g in G[u]:
            return True
        S.add(u)
        Q.extend(G[u])
    return False


def has_unit_cycle(G, path):
    """ 
    Checks  if two  unequal  length  paths can  be  compensated by  their
    elementary cycles 
    
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graphs)
    
    :param path:
    :type path:
    
    :returns: True, if two  unequal  length  paths can  be  compensated by  their
              elementary cycles and vice versa
    :rtype: boolean
    """
    for v in path:
        if v in G[v]:
            return True
    return False
