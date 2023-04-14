""" This is where functions from old experiments and previous data formats have been moved
    to in order to clarify the current code.  These remain for reference and in case they
    become useful again.  No guarantee all imports are here to use these functions."""
import numpy as np
import scipy



""" from bfutils.py """

# tried mutable ctypes buffer - not faster :(
def graph2str(G):
    n = len(G)
    d = {((0, 1),): '1', ((2, 0),): '0', ((2, 0), (0, 1),): '1', ((0, 1), (2, 0),): '1'}
    A = ['0'] * (n * n)
    for v in G:
        for w in G[v]:
            A[n * (int(v, 10) - 1) + int(w, 10) - 1] = d[tuple(G[v][w])]
    return ''.join(A)


def graph2bstr(G):
    n = len(G)
    d = {((0, 1),): '0', ((2, 0),): '1', ((2, 0), (0, 1),): '1', ((0, 1), (2, 0),): '1'}
    A = ['0'] * (n * n)
    for v in G:
        for w in G[v]:
            A[n * (int(v, 10) - 1) + int(w, 10) - 1] = d[tuple(G[v][w])]
    return ''.join(A)


def adj2num(A):
    s = reduce(lambda y, x: y + str(x),
               A.flatten().tolist(), '')
    return int(s, 2)


def num2adj(num, n):
    l = list(bin(num)[2:])
    l = ['0' for i in range(0, n ** 2 - len(l))] + l
    return scipy.reshape(map(int, l), [n, n])


def add_bd_by_adj(G, adj):
    c = 0
    for e in adj:
        for v in range(len(e)):
            if e[v] == 1:
                try:
                    G[str(c + 1)][str(v + 1)].add((2, 0))
                except KeyError:
                    G[str(c + 1)][str(v + 1)] = set([(2, 0)])
        c += 1
    return G


def tuple2graph(t, n):
    g = num2CG(t[0], n)
    return add_bd_by_adj(g, num2adj(t[1], n))


def uniqseq(l):
    s = []
    ltr = map(lambda *a: list(a), *l)
    for i in range(len(ltr)):
        s.append(len(np.unique(ltr[i])))


def jason2graph(g):
    r = {}
    d = {1: set([(0, 1)]),
         2: set([(2, 0)]),
         3: set([(0, 1), (2, 0)])}
    for head in g:
        r[head] = {}
        for tail in g[head]:
            r[head][tail] = d[g[head][tail]]
    return r


def graph2jason(g):
    r = {}
    for head in g:
        r[head] = {}
        for tail in g[head]:
            if g[head][tail] == set([(0, 1)]):
                r[head][tail] = 1
            elif g[head][tail] == set([(2, 0)]):
                r[head][tail] = 2
            elif g[head][tail] == set([(0, 1), (2, 0)]):
                r[head][tail] = 3
    return r










""" from ecj.py """


def traverse(G, s, qtype=set):
    S, Q = set(), qtype()
    Q.add(s)
    while Q:
        u = Q.pop()
        if u in S:
            continue
        S.add(u)
        for v in G[u]:
            Q.add(v)
        yield u
        

def bfs_print_tree(tree, r):
    """
    A modified single list solution
    """
    Q = []
    idx = 1
    print str(idx) + ': ' + r
    Q.extend(tree[r])
    while Q:
        idx += 1
        print str(idx) + ':',
        for u in range(0, len(Q)):
            e = Q.pop(0)
            print e,
            Q.extend(tree[e])
        print ''


def bfs_dict(tree, r):
    """
    Bob's suggested dictionary based solution
    """
    D = {}
    idx = 1
    D[idx] = [r]
    while D[idx]:
        idx += 1
        D[idx] = []
        for u in D[idx - 1]:
            D[idx].extend(tree[u])
    D.pop(idx)  # the last dictionary element is empty - must go
    for idx in D:
        print str(idx) + ': ' + ' '.join(D[idx])


def clrbi(G):
    for v in G:
        d = []
        for u in G[v]:
            try:
                G[v][u].remove((edge_type['bidirected'], 0))
                if len(G[v][u]) == 0:
                    d.append(u)
            except KeyError:
                pass
        for e in d:
            G[v].pop(e)


def ecj(G, s, sccs=set()):           # elementary circuit by Johnson
    blocked = {v: False for v in G}  # unblock all
    B = {v: [] for v in G}
    stack = []

    def unblock(u):
        blocked[u] = False
        for w in B[u]:
            B[u].remove(w)
            if blocked[w]:
                unblock(w)

    def circuit(v, stack):
        f = False
        stack.append(v)
        blocked[v] = True
        for u in G[v]:
            if u == s:
                f = True
                # print stack
                sccs.add(len(stack))
            elif not blocked[u]:
                if circuit(u, stack):
                    f = True
        if f:
            unblock(v)
        else:
            for w in G[v]:
                if v not in B[w]:
                    B[w].append(v)
        stack.pop()
        return f
    circuit(s, stack)
    return sccs


def ecj_loops(G, s, sl=set()):           # elementary circuit by Johnson
    blocked = {v: False for v in G}  # unblock all
    B = {v: [] for v in G}
    stack = []

    def unblock(u):
        blocked[u] = False
        for w in B[u]:
            B[u].remove(w)
            if blocked[w]:
                unblock(w)

    def circuit(v, stack):
        f = False
        stack.append(v)
        blocked[v] = True
        for u in G[v]:
            if u == s:
                f = True
                # print scipy.sort(stack)
                sl.add(tuple(scipy.sort(stack)))
            elif not blocked[u]:
                if circuit(u, stack):
                    f = True
        if f:
            unblock(v)
        else:
            for w in G[v]:
                if v not in B[w]:
                    B[w].append(v)
        stack.pop()
        return f
    circuit(s, stack)
    return sl


def lcm(a, b):
    return a * b / gcd(a, b)


def chmatch(n, m, delta):
    m, n = scipy.sort([n, m])
    sq = scipy.mod(range(n, lcm(n, m) + 1, n), m)
    return scipy.mod(delta, m) in sq


def allpaths(G, s, g, S=[]):
    if S is None:
        S = []
    S.append(s)
    if s == g:
        print S
    else:
        for u in G[s]:
            if u in S:
                continue
            allpaths(G, u, g, S)
    S.remove(s)


def lz_ecj(G, s, sccs=set()):           # elementary circuit by Johnson
    blocked = {v: False for v in G}    # unblock all
    B = {v: set() for v in G}
    stack = []

    def unblock(u):
        blocked[u] = False
        for w in B[u]:
            if blocked[w]:
                unblock(w)
        B[u].clear()

    def circuit(v, stack):
        stack.append(v)
        blocked[v] = True
        for u in G[v]:
            if u == s:
                print 'bottom'
                unblock(v)
                yield len(stack)
            elif not blocked[u]:
                print 'recurse'
                for x in circuit(u, stack):
                    unblock(v)
                    yield x
            else:
                print 'unmet'
                for w in G[v]:
                    B[w].add(v)
        stack.pop()
#    circuit(s,stack)
    for v in circuit(s, stack):
        yield v


def exist_equal_paths(h, G, a, b):
    Sa, Sb, Dff = set(), set(), set()
    ag = iterate_allpaths(G, h, a, 0, [], True)
    bg = iterate_allpaths(G, h, b, 0, [], True)
    for v in izip_longest(ag, bg):
        print v
        Sa.add(v[0])
        Sb.add(v[1])
        if v[0] in Sb or v[1] in Sa:
            return True
    return False


def iexist_equal_paths(h, G, a, b):
    """ checks if there  exist exact length paths from the  head node to there
        nodes at question  by iterative deepining to avoid  oing through all
        paths """
    Sa, Sb = set(), set()
    Pa, Pb = [], []
    ag = iddfs(G, h, a)
    bg = iddfs(G, h, b)
    for v in izip(ag, bg):
        print v
        Sa.add(len(v[0]))
        Pa.append(v[0])
        Sb.add(len(v[1]))
        Pb.append(v[1])
        if len(v[0]) in Sb or len(v[1]) in Sa:
            return True
    return False


def iterate_allpaths(G, s, g, d=0, S=[], c=True):
    if S is None:
        S = []
    S.append(s)
    d += 1
    if s == g:
        if c:
            yield d - 1
        else:
            yield list(S)
    else:
        for u in G[s]:
            if u in S:
                continue
            for v in iterate_allpaths(G, u, g, d, S, c):
                yield v
    S.remove(s)


def iddfs(G, s, g):  # iterative depening DFS paths
    yielded = set()

    def recurse(G, s, g, d, S=None):
        if s not in yielded:
            yielded.add(s)
        if d == 0:
            return
        if S is None:
            S = []
        S.append(s)
        if s == g:
            yield list(S)
        else:
            for u in G[s]:
                if u in S:
                    continue
                for v in recurse(G, u, g, d - 1, S):
                    yield v
        S.remove(s)
    n = len(G)
    for d in range(n):
        # if len(yielded) == n: break
        for u in recurse(G, s, g, d):
            yield u


def ecj_compat(G, p1, p2):
    n = len(p1)
    m = len(p2)
    p2, p1 = [[p1, p2][i] for i in scipy.argsort([n, m])]
    m, n = scipy.sort([n, m])
    delta = n - m
    if not delta:
        return True  # redundant check for 0
    if has_unit_cycle(G, p2):
        return True  # equivalent
    # if the shorter path does not have cycles they are not compatible
    # if the  longer path does not  have cycles: check  if the shorter
    #                                            path    has    cycles
    #                                            divisible by delta

    # otherwise start checking
    print p1, p2, n, m, delta


def wc(n):
    n = n * 3
    a = {str(v): set([str(v + 1), str(v + 2), str(v + 3)]) for v in range(1, n, 3)}
    b = {str(v): set([str(v + 2)]) for v in range(2, n, 3)}
    c = {str(v): set([str(v + 1)]) for v in range(3, n, 3)}
    a.update(b)
    a.update(c)
    a.update({str(n): set()})
    return a


def residue_table(a):
    """ Frobenius number from here: http://cgi.gladman.plus.com/wp/?page_id=563 """
    n = [0] + [None] * (a[0] - 1)
    for i in range(1, len(a)):
        d = gcd(a[0], a[i])
        for r in range(d):
            try:
                nn = min(n[q] for q in range(r, a[0], d) if n[q] != None)
            except:
                continue
            if nn != None:
                for c in range(a[0] // d):
                    nn += a[i]
                    p = nn % a[0]
                    nn = min(nn, n[p]) if n[p] != None else nn
                    n[p] = nn
    return n


def frobenius_number(a):
    return max(residue_table(sorted(a))) - min(a)


def isJclique(G):
    """ Jianyu does not use bidirected edges """
    return (sum([len(G[w].keys()) for w in G]) == len(G) ** 2)


def sample_graph(graph_g, steps=5):
    graph_g_list = [graph_g]
    for i in range(0, steps):
        g = increment_u(graph_g, graph_g_list[-1])
        graph_g_list.append(g)
    return graph_g_list


def reached_at_step(G, s, d):
    """
    Iterate over nodes in G reachable from s in exactly d steps
    """
    yielded = set()

    def recurse(G, s, d, B=None):
        if d == 0:
            if s not in yielded:  # this avoids yielding duplicates
                yielded.add(s)
                yield s
            return
        if B is None:
            B = []  # black - backed out of this path
        for u in G[s]:
            # if u in B: continue
            if G[s][u] == (edge_type['bidirected'], 0):
                continue
            for v in recurse(G, u, d - 1, B):
                yield v
        B.append(s)
    for u in recurse(G, s, d):
        yield u


def d_trek(h, G, a, b, d):
    """
    Does there exist a trek with head h connecting a and b in d steps.
    """
    return set([a, b]).issubset(reached_at_step(G, h, d))


def d_biegde(G, a, b, d):
    """
    Do  a and  b  become connected  by  a bidirectional  edge after  d
    undersamples
    """
    for i in range(1, d + 1):
        for u in G:
            if d_trek(u, G, a, b, i):
                return True
    return False


def undersample(G, d, bid=True):
    """

    """
    N = {}
    for u in G:
        N.update({u: {v: set([(0, 1)]) for v in reached_at_step(G, u, d + 1)}})
    if bid:
        items = G.keys()
        for i in range(len(items)):
            for j in range(i + 1, len(items)):
                u, v = items[i], items[j]
                if d_biegde(G, u, v, d):
                    try:
                        N[u][v].add((edge_type['bidirected'], 0))
                    except KeyError:
                        N[u].update({v: set([(edge_type['bidirected'], 0)])})
                    try:
                        N[v][u].add((edge_type['bidirected'], 0))
                    except KeyError:
                        N[v].update({u: set([(edge_type['bidirected'], 0)])})
    return N







""" from graphkit.py """



def ibedgelist(g):  # directed iterator
    '''
    iterate over the list of tuples for edges of g
    '''
    for v in g:
        for w in g[v]:
            if (2, 0) in g[v][w]:
                yield (v, w)


def edgenumber(g):
    return sum([sum([len(g[y][x]) for x in g[y]]) for y in g])


def iedgelist(g):  # directed iterator
    '''
    iterate over the list of tuples for edges of g
    '''
    for v in g:
        for w in g[v]:
            if (0, 1) in g[v][w]:
                yield (v, w)


def list2dbn(l):
    """ convert list of edge presences/absences (0,1) to a DBN graph
    """
    n = scipy.sqrt(len(l))
    l = scipy.reshape(map(int, l), [n, n])
    G = ecj.adj2DBN(l)
    return G


def rnd_dbn(n):
    return list2dbn(rnd_edges(n))


def sp_rnd_dbn(n, maxindegree=3):
    '''
    a sparse random DBN graph
    '''
    l = sp_rnd_edges(n)
    return list2dbn(l)


def rnd_edges(n):
    """ generate a random uniformly distributed mask
    """
    rnum = std_random.getrandbits(n ** 2)
    l = list(bin(rnum)[2:])
    l = ['0' for i in range(0, n ** 2 - len(l))] + l
    return l


def rnd_adj(n, maxindegree=5):
    l = scipy.zeros([n, n])
    for u in range(0, n):
        cap = scipy.random.randint(min([n, maxindegree + 1]))
        idx = scipy.random.randint(n, size=cap)
        l[u, idx] = 1
    return l


def sp_rnd_edges(n, maxindegree=5):
    '''
    a sparse set of random edges
    '''
    l = rnd_adj(n, maxindegree=maxdegree)
    return scipy.reshape(l, n ** 2)


def list2CG(l):
    """ convert list of edge presences/absences (0,1) to a compressed
    graph (CG) representation
    """
    n = scipy.sqrt(len(l))
    l = scipy.reshape(map(int, l), [n, n])
    G = adj2graph(l)
    return G


def rnd_cg(n):
    return list2CG(rnd_edges(n))


def sp_rnd_CG(n, maxindegree=3, force_connected=False):
    l = sp_rnd_edges(n, maxindegree=maxindegree)
    cg = list2CG(l)
    if force_connected:
        while not connected(cg):
            cg = list2CG(sp_rnd_edges(n, maxindegree=maxindegree))
    return cg


def adj2graph(A):
    G = {str(i): {} for i in range(1, A.shape[0] + 1)}
    idx = np.where(A == 1)
    for i in range(len(idx[0])):
        G['%i' % (idx[0][i] + 1)]['%i' % (idx[1][i]+1)] = set([(0, 1)])
    return G


def emptyG(n):
    A = [[0 for j in range(n)] for i in range(n)]
    return adj2graph(np.asarray(A))


def fullG(n):
    A = [[1 for j in range(n)] for i in range(n)]
    return adj2graph(np.asarray(A))


def CG2uCG(cg):
    """
    convert to an undirected graph
    """
    G = {}
    for u in cg:
        G[u] = cg[u].copy()
    for u in cg:
        for v in cg[u]:
            G[v][u] = cg[u][v]
    return G


def connected(cg):
    n = len(cg)
    return sum(1 for _ in ecj.traverse(CG2uCG(cg), '1')) == n


def fork_mismatch(g):
    be = bedgelist(g)
    benum = len(be) / 2
    forknum = 0
    for v in g:
        fn = len([w for w in g[v] if (0, 1) in g[v][w]])
        forknum += fn * (fn - 1) / 2.
    if benum < len(g) * (len(g) - 1) / 2.:
        return (forknum - benum) > benum
    else:
        return False







""" from pc.py """

def addallb(g):
    n = len(g)
    for i in range(n):
        for j in range(n):
            if str(j+1) in g[str(i+1)]:
                g[str(i+1)][str(j+1)].add((2, 0))
            else:
                g[str(i+1)][str(j+1)] = set([(2, 0)])
    return g




""" from simpleloops.py """


def ul(l):
    """
    returns list elements that are present only once
    """
    u, r = set(), set()
    for e in l:
        if e not in u:
            u.add(e)
        else:
            r.add(e)
    return u.difference(r)





""" from traversal.py """

def next_or_none(it):
    try:
        n = it.next()
    except StopIteration:
        return None
    return n


def ok2addanedge_sub(s, e, g, g2, rate=1):
    mask = addanedge(g, (s, e))
    value = gk.isedgesubset(bfu.undersample(g, rate), g2)
    delanedge(g, (s, e), mask)
    return value


def ok2addanedge_(s, e, g, g2, rate=1):
    f = [ok2addanedge1, ok2addanedge_sub]
    return f[min([1, rate - 1])](s, e, g, g2, rate=rate)


def ok2addapath1(e, p, g, g2):
    for i in range(len(p)):
        if not edge_increment_ok(e[i + 1], p[i], e[i + 2], g, g2):
            return False
    return True


def rotate(l):
    return l[1:] + l[:1]  # rotate a list


def backtrackup2u(H, umax=2):
    s = set()
    for i in xrange(1, umax + 1):
        s = s | backtrack_more(H, rate=i)
    return s


def memo(func):
    cache = {}                        # Stored subproblem solutions

    @wraps(func)                      # Make wrap look like func
    def wrap(*args):                  # The memoized wrapper
        s = signature(args[0], args[2])  # Signature: g and edges
        if s not in cache:            # Not already computed?
            cache[s] = func(*args)    # Compute & cache the solution
        return cache[s]               # Return the cached solution
    return wrap


def memo1(func):
    cache = {}                        # Stored subproblem solutions

    @wraps(func)                      # Make wrap look like func
    def wrap(*args):                  # The memoized wrapper
        s = gsig(args[0])             # Signature: g
        if s not in cache:            # Not already computed?
            cache[s] = func(*args)    # Compute & cache the solution
        return cache[s]               # Return the cached solution
    return wrap


def eqsearch(g2, rate=1):
    '''Find  all  g  that are also in  the equivalence
    class with respect to g2 and the rate.
    '''

    s = set()
    noop = set()

    @memo1
    def addnodes(g, g2, edges):
        if edges:
            masks = []
            for e in edges:
                if ok2addanedge_(e[0], e[1], g, g2, rate=rate):
                    masks.append(True)
                else:
                    masks.append(False)
            nedges = [edges[i] for i in range(len(edges)) if masks[i]]
            n = len(nedges)
            if n:
                for i in range(n):
                    mask = addanedge(g, nedges[i])
                    if bfu.undersample(g, rate) == g2:
                        s.add(g2num(g))
                    addnodes(g, g2, nedges[:i] + nedges[i + 1:])
                    delanedge(g, nedges[i], mask)
                return s
            else:
                return noop
        else:
            return noop

    g = cloneempty(g2)
    edges = gk.edgelist(gk.complement(g))
    addnodes(g, g2, edges)
    return s


def g22g1(g2, capsize=None):
    '''
    computes all g1 that are in the equivalence class for g2
    '''
    if bfu.is_sclique(g2):
        print 'Superclique - any SCC with GCD = 1 fits'
        return set([-1])

    single_cache = {}

    @memo  # memoize the search
    def nodesearch(g, g2, edges, s):
        if edges:
            if bfu.increment(g) == g2:
                s.add(g2num(g))
                if capsize and len(s) > capsize:
                    raise ValueError('Too many elements')
                return g
            e = edges[0]
            for n in g2:

                if (n, e) in single_cache:
                    continue
                if not edge_increment_ok(e[0], n, e[1], g, g2):
                    continue

                mask = add2edges(g, e, n)
                r = nodesearch(g, g2, edges[1:], s)
                del2edges(g, e, n, mask)

        elif bfu.increment(g) == g2:
            s.add(g2num(g))
            if capsize and len(s) > capsize:
                raise ValueError('Too many elements in eqclass')
            return g

    # find all directed g1's not conflicting with g2
    n = len(g2)
    edges = gk.edgelist(g2)
    random.shuffle(edges)
    g = cloneempty(g2)

    for e in edges:
        for n in g2:

            mask = add2edges(g, e, n)
            if not gk.isedgesubset(bfu.increment(g), g2):
                single_cache[(n, e)] = False
            del2edges(g, e, n, mask)

    s = set()
    try:
        nodesearch(g, g2, edges, s)
    except ValueError:
        s.add(0)
    return s


def vg22g1(g2, capsize=None):
    '''
    computes all g1 that are in the equivalence class for g2
    '''
    if bfu.is_sclique(g2):
        print 'Superclique - any SCC with GCD = 1 fits'
        return set([-1])

    f = [(add2edges, del2edges),
         (addavedge, delavedge),
         (addacedge, delacedge),
         (addaAedge, delaAedge),
         (addapath, delapath)]
    c = [ok2add2edges,
         ok2addavedge,
         ok2addacedge,
         ok2addaAedge,
         ok2addapath]

    @memo2  # memoize the search
    def nodesearch(g, g2, edges, s):
        if edges:
            # key, checklist = edges.popitem()
            key = random.choice(edges.keys())
            checklist = edges.pop(key)
            adder, remover = f[edge_function_idx(key)]
            checks_ok = c[edge_function_idx(key)]
            for n in checklist:
                mask = adder(g, key, n)
                if gk.isedgesubset(bfu.increment(g), g2):
                    r = nodesearch(g, g2, edges, s)
                    if r and bfu.increment(r) == g2:
                        s.add(g2num(r))
                        if capsize and len(s) > capsize:
                            raise ValueError('Too many elements')
                remover(g, key, n, mask)
            edges[key] = checklist
        else:
            return g

    # find all directed g1's not conflicting with g2
    n = len(g2)
    chlist = checkable(g2)
    g = cloneempty(g2)

    s = set()
    try:
        nodesearch(g, g2, chlist, s)
    except ValueError:
        s.add(0)
    return s

    
def backtrack_more(g2, rate=1, capsize=None):
    '''
    computes all g1 that are in the equivalence class for g2
    '''
    if bfu.is_sclique(g2):
        print 'Superclique - any SCC with GCD = 1 fits'
        return set([-1])

    single_cache = {}
    if rate == 1:
        ln = [n for n in g2]
    else:
        ln = []
        for x in itertools.combinations_with_replacement(g2.keys(), rate):
            ln.extend(itertools.permutations(x, rate))
        ln = set(ln)

    @memo  # memoize the search
    def nodesearch(g, g2, edges, s):
        if edges:
            if bfu.undersample(g, rate) == g2:
                s.add(g2num(g))
                if capsize and len(s) > capsize:
                    raise ValueError('Too many elements')
                return g
            e = edges[0]
            for n in ln:

                if (n, e) in single_cache:
                    continue
                if not ok2addaVpath(e, n, g, g2, rate=rate):
                    continue

                mask = addaVpath(g, e, n)
                r = nodesearch(g, g2, edges[1:], s)
                delaVpath(g, e, n, mask)

        elif bfu.undersample(g, rate) == g2:
            s.add(g2num(g))
            if capsize and len(s) > capsize:
                raise ValueError('Too many elements in eqclass')
            return g

    # find all directed g1's not conflicting with g2
    n = len(g2)
    edges = gk.edgelist(g2)
    random.shuffle(edges)
    g = cloneempty(g2)

    for e in edges:
        for n in ln:

            mask = addaVpath(g, e, n)
            if not gk.isedgesubset(bfu.undersample(g, rate), g2):
                single_cache[(n, e)] = False
            delaVpath(g, e, n, mask)

    s = set()
    try:
        nodesearch(g, g2, edges, s)
    except ValueError:
        s.add(0)
    return sdef edge_backtrack2g1_directed(g2, capsize=None):
    '''
    computes all g1 that are in the equivalence class for g2
    '''
    if bfu.is_sclique(g2):
        print 'Superclique - any SCC with GCD = 1 fits'
        return set([-1])

    single_cache = {}

    def edgeset(g):
        return set(gk.edgelist(g))

    @memo  # memoize the search
    def nodesearch(g, g2, edges, s):
        if edges:
            e = edges.pop()
            ln = [n for n in g2]
            for n in ln:
                if (n, e) in single_cache:
                    continue
                mask = add2edges(g, e, n)
                if gk.isedgesubset(bfu.increment(g), g2):
                    r = nodesearch(g, g2, edges, s)
                    if r and edgeset(bfu.increment(r)) == edgeset(g2):
                        s.add(g2num(r))
                        if capsize and len(s) > capsize:
                            raise ValueError('Too many elements in eqclass')
                del2edges(g, e, n, mask)
            edges.append(e)
        else:
            return g
    # find all directed g1's not conflicting with g2
    n = len(g2)
    edges = gk.edgelist(g2)
    random.shuffle(edges)
    g = cloneempty(g2)

    for e in edges:
        for n in g2:
            mask = add2edges(g, e, n)
            if not gk.isedgesubset(bfu.increment(g), g2):
                single_cache[(n, e)] = False
            del2edges(g, e, n, mask)

    s = set()
    try:
        nodesearch(g, g2, edges, s)
    except ValueError:
        s.add(0)
    return s


def edge_backtrack2g1(g2, capsize=None):
    '''
    computes all g1 that are in the equivalence class for g2
    '''
    if bfu.is_sclique(g2):
        print 'Superclique - any SCC with GCD = 1 fits'
        return set([-1])

    single_cache = {}

    @memo  # memoize the search
    def nodesearch(g, g2, edges, s):
        if edges:
            e = edges.pop()
            ln = [n for n in g2]
            for n in ln:
                if (n, e) in single_cache:
                    continue
                mask = add2edges(g, e, n)
                if gk.isedgesubset(bfu.increment(g), g2):
                    r = nodesearch(g, g2, edges, s)
                    if r and bfu.increment(r) == g2:
                        s.add(g2num(r))
                        if capsize and len(s) > capsize:
                            raise ValueError('Too many elements in eqclass')
                del2edges(g, e, n, mask)
            edges.append(e)
        else:
            return g
    # find all directed g1's not conflicting with g2
    n = len(g2)
    edges = gk.edgelist(g2)
    random.shuffle(edges)
    g = cloneempty(g2)

    for e in edges:
        for n in g2:
            mask = add2edges(g, e, n)
            if not gk.isedgesubset(bfu.increment(g), g2):
                single_cache[(n, e)] = False
            del2edges(g, e, n, mask)

    s = set()
    try:
        nodesearch(g, g2, edges, s)
    except ValueError:
        s.add(0)
    return s


def backtrack_more2(g2, rate=2, capsize=None):
    '''
    computes all g1 that are in the equivalence class for g2
    '''
    if bfu.is_sclique(g2):
        print 'Superclique - any SCC with GCD = 1 fits'
        return set([-1])

    f = [(addaVpath, delaVpath, maskaVpath)]
    c = [ok2addaVpath]

    def predictive_check(g, g2, pool, checks_ok, key):
        s = set()
        for u in pool:
            if not checks_ok(key, u, g, g2, rate=rate):
                continue
            s.add(u)
        return s

    @memo2  # memoize the search
    def nodesearch(g, g2, order, inlist, s, cds, pool, pc):
        if order:
            if bfu.undersample(g, rate) == g2:
                s.add(g2num(g))
                if capsize and len(s) > capsize:
                    raise ValueError('Too many elements')
                s.update(supergraphs_in_eq(g, g2, rate=rate))
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
                if not checks_ok(key, n, g, g2, rate=rate):
                    continue
                masked = np.prod(masker(g, key, n))
                if masked:
                    nodesearch(g, g2, order[1:], [n] + inlist, s, cds, pool, pc)
                else:
                    mask = adder(g, key, n)
                    nodesearch(g, g2, order[1:], [n] + inlist, s, cds, pool, pc)
                    remover(g, key, n, mask)

        elif bfu.undersample(g, rate) == g2:
            s.add(g2num(g))
            if capsize and len(s) > capsize:
                raise ValueError('Too many elements')
            return g

    # find all directed g1's not conflicting with g2

    startTime = int(round(time.time() * 1000))
    ln = [x for x in itertools.permutations(g2.keys(), rate)] + \
         [(n, n) for n in g2]
    gg = {x: ln for x in gk.edgelist(g2)}
    keys = gg.keys()
    cds, order, idx = conformanceDS(g2, gg, gg.keys(), f=f, c=c)
    endTime = int(round(time.time() * 1000))
    print "precomputed in {:10} seconds".format(round((endTime - startTime) / 1000., 3))
    if 0 in [len(x) for x in order]:
        return set()
    g = cloneempty(g2)

    s = set()
    try:
        nodesearch(g, g2, [keys[i] for i in idx], ['0'], s, cds, order, set())
    except ValueError, e:
        print e
        s.add(0)
    return s


def unionpool(idx, cds):
    s = set()
    for u in cds[idx]:
        for v in cds[idx][u]:
            s = s.union(cds[idx][u][v])
    return s


def prune_modify_CDS(cds, pool):
    ds = {}
    ds[0] = {}
    ds[0]['0'] = pool[0]
    for i in range(1, len(pool)):
        ds[i] = {}
        for j in cds[i].keys():
            for e in pool[i - 1].intersection(cds[i][j].keys()):
                ds[i][e] = pool[i].intersection(cds[i][j][e])
    return ds, pool, range(len(pool))


def conformant(cds, inlist):

    if inlist[len(inlist) - 2] in cds[len(inlist) - 1][0]:
        s = cds[len(inlist) - 1][0][inlist[len(inlist) - 2]]
    else:
        return set()
    for i in range(1, len(inlist) - 1):
        if inlist[len(inlist) - i - 2] in cds[len(inlist) - 1][i]:
            s = s.intersection(cds[len(inlist) - 1][i][inlist[len(inlist) - i - 2]])
        else:
            return set()
    return s










""" from dbn2latex.py """


def unroll(G, steps):
    N = {}
    for i in range(0, steps):
        N.update({v + str(i): set([u + str(i + 1) for u in G[v]]) for v in G})
    N.update({v + str(steps): set() for v in G})
    return N


def unroll_undersample(G, steps):
    # does not provide isochronal bidirectional edges
    N = {}
    steps += 2
    U = unroll(G, steps)
    nodes = G.keys()
    for v in G:
        N.update(
            {v: set([nodes[k] for k in scipy.where([ecj.reachable(v + '0', U, u + str(steps - 1)) for u in G])[0]])})
    return N










""" from unknownrate.py """


def eqclass(H):
    '''
    Find all graphs in the same equivalence class with respect to
    graph H and any undesampling rate.
    '''
    g = {n: {} for n in H}
    s = set()

    @memo
    def addedges(g, H, edges):
        if edges:
            nedges = prune_conflicts(H, g, edges)
            n = len(nedges)

            if n == 0:
                return None

            for i in range(n):
                gk.addanedge(g, nedges[i])
                if bfu.call_u_equals(g, H):
                    s.add(g2num(g))
                addedges(g, H, nedges[:i] + nedges[i+1:])
                gk.delanedge(g, nedges[i])
    edges = gk.edgelist(gk.complement(g))
    addedges(g, H, edges)

    return s - set([None])

# these two functions come from this answer:
# http://stackoverflow.com/a/12174125

def set_bit(value, bit):
    return value | (1 << bit)


def clear_bit(value, bit):
    return value & ~(1 << bit)


def hashloop(l):
    t = [int(x) for x in l]
    idx = np.argmin(t)
    return tuple(l[idx:] + l[:idx])


def perm_circular_slow2(l):
    s = [tuple(l)]
    c = {}
    c[hashloop(l)] = True
    for e in permutations(l):
        if not hashloop(e) in c:
            s.append(e)
            c[hashloop(e)] = True
    return s


def perm_cyclic(l):
    return [tuple(l[i:] + l[:i]) for i in range(len(l))]


def perm_circular_slow(l):
    s = [tuple(l)]
    c = set(perm_cyclic(l))
    for e in permutations(l):
        if not e in c:
            s.append(e)
            c = c | set(perm_cyclic(e))
    return s


def may_be_true_selfloop(n, H):
    for v in H[n]:
        if v == n:
            continue
        if H[n][v] == 1:
            return False
    return True


def issingleloop(num):
    bl = gmp.bit_length(num)
    idx = [1 for i in xrange(bl) if num & (1 << i)]
    return len(idx) == 1


def nonbarren(H):
    for v in H:
        if H[v]:
            return v
    return False


def count_loops(n):
    s = 0
    for i in range(1, n + 1):
        s += comb(n, i) * math.factorial(i - 1)
    return s


def noverlap_loops(loops):
    d = {}
    for l in loops:
        el = []
        for k in loops:
            if not set(l) & set(k):
                el.append(tuple(k))
                # d.setdefault(tuple(l),set()).add(tuple(k))
        d[tuple(l)] = noverlap_loops(el)
    return d


def loop_combinations(loops):
    s = set()
    d = noverlap_loops(loops)

    def dfs_traverse(d, gs):
        if d:
            for e in d:
                dfs_traverse(d[e], gs | set([e]))
        else:
            s.add(frozenset(gs))
    for e in d:
        dfs_traverse(d[e], set([e]))
    return list(s)


def sorted_loops(g):
    l = [x for x in sl.simple_loops(g, 0)]
    s = {}
    for e in l:
        s.setdefault(len(e), []).append(e)
    return s


def loopgroups(g, n=None):
    d = sorted_loops(g)
    if n:
        return loop_combinations(d[n])
    else:
        l = []
        for key in d:
            l.append(loop_combinations(d[key]))
        return l


def rotate(l, n):
    return l[n:] + l[:n]


def get_perm(loop1, loop2, n=None):
    if not n:
        n = len(loop1)
    basel = [i for i in xrange(1, n + 1)]
    diff1 = set(basel) - set(loop1)
    diff2 = set(basel) - set(loop2)
    if loop1[0] in loop2:
        l2 = rotate(loop2, loop2.index(loop1[0]))
    else:
        l2 = loop2
    mp = {}
    for x, y in zip(loop1 + list(diff1), l2+list(diff2)):
        mp[x] = y
    return mp


def permute(g, perm):
    gn = {x: {} for x in g}
    for e in g:
        gn[perm[e]] = {perm[x]: g[e][x] for x in g[e]}
    return gn


def permuteAset(s, perm):
    n = len(perm)
    ns = set()
    for e in s:
        ns.add(g2num(permute(num2CG(e, n), perm)))
    return ns


def set_loop(loop, graph):
    for i in range(0, len(loop) - 1):
        graph[loop[i]][loop[i + 1]] = 1
    graph[loop[-1]][loop[0]] = 1


def add2set(gset, elist, H):
    n = len(H)

    s = set()
    ss = set()

    eremove = {e: True for e in elist}

    for gnum in gset:
        g = num2CG(gnum, n)
        for e in elist:
            if not e[1] in g[e[0]]:
                gk.addanedge(g, e)
                num = g2num(g)
                if not num in s:
                    au = bfu.call_undersamples(g)
                    if not bfu.check_conflict(H, g, au=au):
                        eremove[e] = False
                        s.add(num)
                        if bfu.check_equality(H, g, au=au):
                            ss.add(num)
                gk.delanedge(g, e)

    for e in eremove:
        if eremove[e]:
            elist.remove(e)

    return s, ss, elist


def eqclass_list(H):
    '''
    Find all graphs in the same equivalence class with respect to
    graph H and any undesampling rate.
    '''
    g = {n: {} for n in H}
    s = set()

    edges = gk.edgelist(gk.complement(g))
    # edges = prune_conflicts(H, g, edges)

    gset = set([g2num(g)])
    for i in range(len(H) ** 2):
        print i
        gset, ss, edges = add2set(gset, edges, H)
        s = s | ss
        if not edges:
            break

    return s


def quadmerge_(glist, H, ds):
    n = len(H)
    gl = set()
    ss = set()
    conflicts = set()
    for gi in combinations(glist, 2):
        if gi[0] & gi[1]:
            continue
        # if skip_conflict(gi[0], gi[1], ds): continue
        gnum = gi[0] | gi[1]
        if gnum in conflicts:
            continue
        if skip_conflictors(gnum, ds):
            conflicts.add(gnum)
            continue
        if gnum in gl:
            continue
        g = num2CG(gnum, n)
        if not bfu.call_u_conflicts(g, H):
            gl.add(gnum)
            if bfu.call_u_equals(g, H):
                ss.add(gnum)
        else:
            conflicts.add(gnum)
    return gl, ss


def edgemask(gl, H, cds):
    """given a list of encoded graphs and observed undersampled graph
    H returns a matrix with -1 on diagonal, 0 at the conflicting graph
    combination and encoded graph at non-conflicted
    positions. Furthermore, returns a set of graphs that are in the
    equivalence class of H

    Arguments:
    - `gl`: list of integer encoded graphs
    - `H`: the observed undersampled graph
    """
    n = len(H)
    nl = len(gl)
    s = set()
    mask = np.zeros((nl, nl), 'int')
    np.fill_diagonal(mask, -1)

    for i in xrange(nl):
        for j in xrange(i + 1, nl):

            if gl[i] & gl[j]:
                continue
            if skip_conflict(gl[i], gl[j], cds):
                continue

            gnum = gl[i] | gl[j]
            g = num2CG(gnum, n)
            if not bfu.call_u_conflicts(g, H):
                if bfu.call_u_equals(g, H):
                    s.add(gnum)
                mask[i, j] = gnum
                mask[j, i] = gnum
    return mask, s


def ledgemask(gl, H, cds):
    """given a list of encoded graphs and observed undersampled graph
    H returns a matrix with -1 on diagonal, 0 at the conflicting graph
    combination and encoded graph at non-conflicted
    positions. Furthermore, returns a set of graphs that are in the
    equivalence class of H

    Arguments:
    - `gl`: list of integer encoded graphs
    - `H`: the observed undersampled graph
    """
    n = len(H)
    nl = len(gl)
    s = set()
    mask = np.zeros((nl, nl), 'int')
    np.fill_diagonal(mask, -1)

    for i in xrange(nl):
        for j in xrange(i + 1, nl):

            if gl[i] & gl[j]:
                continue
            gnum = gl[i] | gl[j]
            if skip_conflictors(gnum, cds):
                continue
            g = num2CG(gnum, n)
            if not bfu.call_u_conflicts(g, H):
                if bfu.call_u_equals(g, H):
                    s.add(gnum)
                mask[i, j] = gnum
                mask[j, i] = gnum
    return mask, s


def edgeds(mask):
    """construct an edge dictionary from the mask matrix

    Arguments:
    - `mask`:
    """
    ds = {}
    nl = mask.shape[0]
    idx = np.triu_indices(nl, 1)
    for i, j in zip(idx[0], idx[1]):
        if mask[i, j]:
            ds[(i, j)] = set()
            conf = set([i, j])
            conf = conf.union(np.where(mask[i,:] == 0)[0])
            conf = conf.union(np.where(mask[j,:] == 0)[0])
            for k, m in zip(idx[0], idx[1]):
                if not mask[k, m]:
                    continue
                if k in conf:
                    continue
                if m in conf:
                    continue
                if not (k, m) in ds:
                    ds[(i, j)].add(mask[k, m])
            if not ds[(i, j)]:
                ds.pop((i, j))
    return ds


def edgedsg(mask):
    """construct an edge dictionary from the mask matrix

    Arguments:
    - `mask`:
    """
    ds = {}
    nl = mask.shape[0]
    idx = np.triu_indices(nl, 1)
    for i, j in zip(idx[0], idx[1]):
        if mask[i, j]:
            ds[mask[i, j]] = set()
            conf = set([i, j])
            conf = conf.union(np.where(mask[i,:] == 0)[0])
            conf = conf.union(np.where(mask[j,:] == 0)[0])
            for k, m in zip(idx[0], idx[1]):
                if not mask[k, m]:
                    continue
                if k in conf:
                    continue
                if m in conf:
                    continue
                if not (k, m) in ds:
                    ds[mask[i, j]].add(mask[k, m])
            if not ds[mask[i, j]]:
                ds.pop(mask[i, j])
    return ds


def quadlister(glist, H, cds):
    n = len(H)
    s = set()
    cache = {}

    def edgemask(gl, H, cds):
        nl = len(gl)
        ss = set()
        mask = np.zeros((nl, nl), 'int')
        np.fill_diagonal(mask, -1)
        idx = np.triu_indices(nl, 1)
        for i, j in zip(idx[0], idx[1]):
            if gl[i] & gl[j]:
                mask[i, j] = -1
                mask[j, i] = -1
                continue
            if skip_conflict(gl[i], gl[j], cds):
                gnum = gl[i] | gl[j]
                cache[gnum] = False
                continue

            gnum = gl[i] | gl[j]
            if gnum in cache:
                if cache[gnum]:
                    mask[i, j] = gnum
                    mask[j, i] = gnum
            else:
                cache[gnum] = False
                g = num2CG(gnum, n)
                if not bfu.call_u_conflicts(g, H):
                    if bfu.call_u_equals(g, H):
                        ss.add(gnum)
                    mask[i, j] = gnum
                    mask[j, i] = gnum
                    cache[gnum] = True
        return mask, ss

    def quadmerger(gl, H, cds):
        mask, ss = edgemask(gl, H, cds)
        ds = edgeds(mask)
        # ipdb.set_trace()
        return [[mask[x]] + list(ds[x]) for x in ds], ss

    l = []
    for gl in glist:
        ll, ss = quadmerger(gl, H, cds)
        l.extend(ll)
        s = s | ss

    return l, s


def dceqc(H):
    """Find all graphs in the same equivalence class with respect to H

    Arguments:
    - `H`: an undersampled graph
    """
    if bfu.is_sclique(H):
        print 'not running on superclique'
        return set()
    n = len(H)
    s = set()
    cds = confpairs(H)

    glist =  [2 ** np.arange(n**2)]
    i = 1
    # for i in range(int(np.log2(n**2))):
    while glist != []:
        print i, np.max(map(len, glist)), len(glist)
        glist, ss = quadlister(glist, H, cds)
        s = s | ss
        i += 1
    return s


def quadmerge(gl, H, cds):
    n = len(H)
    l = set()
    s = set()
    mask, ss = edgemask(gl, H, cds)
    s = s | ss
    ds = edgeds(mask)

    # pp = pprint.PrettyPrinter(indent=1)
    # pp.pprint(ds)

    for idx in ds:
        for gn in ds[idx]:
            if mask[idx] & gn:
                continue
            if skip_conflict(mask[idx], gn, cds):
                continue
            gnum = mask[idx] | gn
            if gnum in l or gnum in ss:
                continue
            g = num2CG(gnum, n)
            if not bfu.call_u_conflicts(g, H):
                l.add(gnum)
                if bfu.call_u_equals(g, H):
                    s.add(gnum)

    return list(l), s


def edgemask2(gl, H, cds):
    n = len(H)
    nl = len(gl)
    s = set()
    o = set()
    mask = np.zeros((nl, nl), 'int')
    np.fill_diagonal(mask, -1)
    for i in xrange(nl):
        for j in xrange(i + 1, nl):
            if gl[i] & gl[j]:
                continue
            gnum = gl[i] | gl[j]
            if skip_conflictors(gnum, cds):
                continue
            g = num2CG(gnum, n)
            if not bfu.call_u_conflicts(g, H):
                if bfu.call_u_equals(g, H):
                    s.add(gnum)
                mask[i, j] = gnum
                mask[j, i] = gnum
            elif bfu.overshoot(g, H):
                o.add(gnum)
    return mask, s, o  # mask, found eqc members, overshoots


def ecmerge(H):
    """Find all graphs in the same equivalence class with respect to H

    Arguments:
    - `H`: an undersampled graph
    """
    if bfu.is_sclique(H):
        print 'not running on superclique'
        return None
    n = len(H)
    s = set()
    ds = confpairs(H)
    ccf = conflictors(H)
    cset = set()
    for e in ccf:
        cset = cset.union(ccf[e])

    glist =  np.r_[[0], 2 ** np.arange(n**2)]
    # glist =  2**np.arange(n**2)

    # glist, ss = quadmerge(glist,H)

    for i in range(int(2 * np.log2(n))):
        print i, len(glist)
        glist, ss = quadmerge_(glist, H, cset)
        s = s | ss
    return s


def getrates(g, H):
    n = len(H)
    au = bfu.call_undersamples(g)
    return list(np.where(map(lambda x: x == H, au))[0])


def withrates(s, H):
    n = len(H)
    d = {g: set() for g in s}
    for g in s:
        d[g] = getrates(num2CG(g, n), H)
    return d


def patchmerge(ds, H, cds):
    n = len(H)
    l = set()
    s = set()
    o = set()
    for gkey in ds:
        for num in ds[gkey]:
            if gkey & num:
                continue
            gnum = gkey | num
            if gnum is s:
                continue
            if skip_conflictors(gnum, cds):
                continue
            g = num2CG(gnum, n)
            if not bfu.call_u_conflicts(g, H):
                l.add(gnum)
                if bfu.call_u_equals(g, H):
                    s.add(gnum)
            elif not gnum in o and bfu.overshoot(g, H):
                o.add(gnum)
    return l, s, o


def quadmerge2(gl, H, cds):
    n = len(H)

    mask, s, o = edgemask2(gl, H, cds)
    # ipdb.set_trace()
    ds = edgedsg(mask)
    l, ss, oo = patchmerge(ds, H, cds)

    o = o | oo
    s = s | ss

    print 'overshoots: ', len(o)

    return list(l), s


def quadmerge21(gl, H, cds):
    n = len(H)
    l = set()

    mask, ss, o = edgemask2(gl, H, cds)
    idx = np.triu_indices(mask.shape[0], 1)
    print len(o)
    for i in range(len(idx[0])):
        if mask[idx[0][i], idx[1][i]]:
            l.add(mask[idx[0][i], idx[1][i]])

    return list(l), ss


def dceqclass2(H):
    """Find all graphs in the same equivalence class with respect to H

    Arguments:
    - `H`: an undersampled graph
    """
    if bfu.is_sclique(H):
        print 'not running on superclique'
        return set()
    n = len(H)
    s = set()
    cp = confpairs(H)
    confs = conflictor_set(H)
    ccf = conflictors(H)

    def prune_loops(gl, H):
        l = []
        for e in gl:
            if e[0] == e[1] and not (e[1] in H[e[0]] and H[e[0]][e[1]] in (1,3)):
                continue
            l.append(e)
        return l
    edges = gk.edgelist(gk.complement(num2CG(0, n)))
    edges = prune_loops(edges, H)
    glist = map(lambda x: e2num(x, n), edges)

    # glist =  list(2**np.arange(n**2))
    i = 0
    while glist != []:
        print 2 ** i, len(glist)
        glist_prev = glist
        glist, ss = quadmerge21(glist, H, confs)
        s = s | ss
        i += 1

    ds = {x: edges for x in glist_prev}

    for j in range(i, len(H) ** 2):
        ds, ss = add2set_(ds, H, cp, ccf, iter=j, verbose=True)
        s = s | ss
        if not ds:
            break

    return s


def dceqclass(H):
    """Find all graphs in the same equivalence class with respect to H

    Arguments:
    - `H`: an undersampled graph
    """
    if bfu.is_sclique(H):
        print 'not running on superclique'
        return set()
    n = len(H)
    s = set()
    cds = confpairs(H)

    glist =  [0] + list(2**np.arange(n**2))
    i = 1
    while glist != []:
        print i, len(glist)
        glist, ss = quadmerge(glist, H, cds)
        s = s | ss
        i += 1
    return s


def ldceqclass(H, asl=None):
    """Find all graphs in the same equivalence class with respect to H

    Arguments:
    - `H`: an undersampled graph
    """
    if bfu.is_sclique(H):
        print 'not running on superclique'
        return set()
    n = len(H)
    s = set()
    cds = lconfpairs(H)
    if asl:
        sloops = asl
    else:
        sloops = prune_loops(allsloops(len(H)), H)

    glist = sloops
    i = 1
    while glist != []:
        print i, len(glist)
        glist, ss = lquadmerge(glist, H, cds)
        s = s | ss
        i += 1
    return s


def lquadmerge(gl, H, cds):
    n = len(H)
    l = set()
    s = set()
    mask, ss = ledgemask(gl, H, cds)
    s = s | ss
    ds = edgeds(mask)

    # pp = pprint.PrettyPrinter(indent=1)
    # pp.pprint(ds)

    for idx in ds:
        for gn in ds[idx]:
            if mask[idx] & gn:
                continue
            if skip_conflictors(mask[idx], gn, cds):
                continue
            gnum = mask[idx] | gn
            if gnum in l or gnum in ss:
                continue
            g = num2CG(gnum, n)
            if not bfu.call_u_conflicts(g, H):
                l.add(gnum)
                if bfu.call_u_equals(g, H):
                    s.add(gnum)

    return list(l), s