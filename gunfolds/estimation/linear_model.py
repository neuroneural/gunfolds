from gunfolds.utils import ecj
from gunfolds.utils import bfutils as bfu
from gunfolds.conversions import graph2adj, adjs2graph
from gunfolds.utils import graphkit as gk
import numpy as np
from progressbar import ProgressBar, Percentage
from scipy import linalg, optimize
import scipy.sparse as sp
from scipy.sparse.linalg import eigs
from statsmodels.tsa.api import VAR
from sympy.matrices import SparseMatrix


def symchol(M):  # symbolic Cholesky
    """
    :param M:
    :type M:
    
    :returns: 
    :rtype: 
    """
    B = SparseMatrix(M)
    t = B.row_structure_symbolic_cholesky()
    B = np.asarray(B)*0
    for i in range(B.shape[0]):
        B[i, t[i]] = 1
    return B


def G2SVAR(G):
    """
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graph)
    
    :returns: 
    :rtype: 
    """
    n = len(G)
    A, B = npG2SVAR(G)
    P, L, U = linalg.lu(B)
    A = linalg.inv(L).tolist()
    B = B.tolist()
    A = listplace(A, 0.0, 0.0)
    for i in range(0, n):
        A[i][i] = 1
    B = listplace(B, 0.0, 'e')
    for i in range(0, n):
        B[i][i] = 'e'
    return A, B, P


def G2AH(G):
    """
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graph)
    
    :returns: 
    :rtype: 
    """
    n = len(G)
    A, B = npG2SVAR(G)
    P, L, U = linalg.lu(B)
    A = linalg.inv(L).tolist()
    B = B.tolist()
    A = listplace(A, 0.0, 0.0)
    for i in range(0, n):
        A[i][i] = 1
    B = listplace(B, 0.0, 'e')
    for i in range(0, n):
        B[i][i] = 'e'
    return A, B, P


def bnf2CG(fname):
    """
    :param fname:
    :type fname:
    
    :returns: 
    :rtype: 
    """
    d = eval(open(fname).read())
    G = {}
    for v in d:
        G[v] = {u: 1 for u in d[v]['pars']}
    G = ecj.tr(G)
    for v in G:
        ld = {u: 1 for u in G[v]}
        G[v] = ld
    return G


def npG2SVAR(G):
    """
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graph)
    
    :returns: 
    :rtype: 
    """
    n = len(G)
    A = [[0]*n]*n
    B = [[0]*n]*n
    for i in range(n):
        B[i][i] = 1

    for v in G:
        for w in G[v]:
            if G[v][w] in (1, 3):
                A[w-1][v-1] = 1
            if G[v][w] in (2, 3):
                B[w-1][v-1] = 1
    A = np.asarray(A)
    B = symchol(B)
    return A, B


def x2M(x, A, B, aidx, bidx):
    """
    :param x:
    :type x:
    
    :param A:
    :type A:
    
    :param B:
    :type B:
    
    :param aidx:
    :type aidx:
    
    :param bidx:
    :type bidx:
    
    :returns: 
    :rtype: 
    """
    A[aidx] = x[:len(aidx[0])]
    B[bidx] = x[len(aidx[0]):]
    # B[(bidx[1],bidx[0])] = x[len(aidx[0]):]
    return A, B


def nllf(x, A, B, Y, aidx, bidx):  # negative log likelihood
    """
    :param x:
    :type x:
    
    :param A:
    :type A:
    
    :param B:
    :type B:
    
    :param Y:
    :type Y:
    
    :param aidx:
    :type aidx:
    
    :param bidx:
    :type bidx:
    
    :returns: 
    :rtype: 
    """
    A, B = x2M(x, A, B, aidx, bidx)
    T = Y.shape[1]
    X = Y[:, 1:] - np.dot(A, Y[:, :-1])
    ldB = T*np.log(abs(1./linalg.det(B)))
    return ldB + 0.5*np.trace(np.dot(np.dot(B.T, B), np.dot(X, X.T)))


def nllf2(x, A, B, YY, XX, YX, T, aidx, bidx):  # negative log likelihood
    """
    :param x:
    :type x:
    
    :param A:
    :type A:
    
    :param B:
    :type B:
    
    :param YY:
    :type YY:
    
    :param XX:
    :type XX:
    
    :param YX:
    :type YX:
    
    :param T:
    :type T:
    
    :param aidx:
    :type aidx:
    
    :param bidx:
    :type bidx:
    
    :returns: 
    :rtype: 
    """
    A, B = x2M(x, A, B, aidx, bidx)
    AYX = np.dot(A, YX.T)
    S = YY - AYX - AYX.T + np.dot(np.dot(A, XX), A.T)
    ldB = T*np.log(abs(1./linalg.det(B)))
    return 0.5*np.dot(np.dot(B.T, B).T.flat, S.flat) + ldB
    # return ldB + 0.5*np.trace( np.dot(np.dot(B.T, B), S))


def VARbic(nllf, K, T):
    """
    :param nllf:
    :type nllf:
    
    :param K:
    :type K:
    
    :param T:
    :type T:
    
    :returns: 
    :rtype: 
    """
    return 2*nllf + K*np.log(T)


def listplace(l, a, b):
    """
    :param l:
    :type l:
    
    :param a:
    :type a:
    
    :param b:
    :type b:
    
    :returns: 
    :rtype: 
    """
    return [listplace(x, a, b) if not np.isscalar(x) else b if x != a else x for x in l]

# -------------------------------------------------------------------
# data generation
# -------------------------------------------------------------------


def randweights(n, c=0.1, factor=9):
    """
    :param n:
    :type n:
    
    :param c:
    :type c: float
    
    :param factor:
    :type factor: (guess)integer
    
    :returns: 
    :rtype: 
    """
    rw = np.random.randn(n)
    idx = np.where(abs(rw) < factor*c)
    if idx:
        rw[idx] = rw[idx]+np.sign(rw[idx])*c*factor
    return rw


def transitionMatrix(cg, minstrength=0.1):
    """
    :param cg:
    :type cg:
    
    :param minstrength:
    :type minstrength: float
    
    :returns: 
    :rtype: 
    """
    A = graph2adj(cg)
    edges = np.where(A == 1)
    A[edges] = randweights(edges[0].shape[0], c=minstrength)
    l = linalg.eig(A)[0]
    c = 0
    pbar = ProgressBar(widgets=['Searching for weights: ', Percentage(), ' '], maxval=10000).start()
    while max(l*np.conj(l)) > 1:
        A[edges] = randweights(edges[0].shape[0], c=c)
        c += 1
        l = linalg.eig(A)[0]
        pbar.update(c)
    pbar.finish()
    return A


def sampleWeights(n, minstrength=0.1):
    """
    :param n:
    :type n:
    
    :param minstrength:
    :type minstrength: float
    
    :returns: 
    :rtype: 
    """
    r = np.randn(n)
    s = minstrength/np.min(np.abs(r))
    r = s*r
    return r


def transitionMatrix2(cg, minstrength=0.1):
    """
    :param cg:
    :type cg:
    
    :param minstrength:
    :type minstrength: float
    
    :returns: 
    :rtype: 
    """
    A = graph2adj(cg)
    edges = np.where(A == 1)
    A[edges] = sampleWeights(edges[0].shape[0], minstrength=minstrength)
    l = linalg.eig(A)[0]
    c = 0
    pbar = ProgressBar(widgets=['Searching for weights: ', Percentage(), ' '], maxval=10000).start()
    while max(l*np.conj(l)) > 1:
        A[edges] = sampleWeights(edges[0].shape[0], minstrength=minstrength)
        c += 1
        l = linalg.eig(A)[0]
        if c > pbar.maxval:
            raise ValueError
        pbar.update(c)
    pbar.finish()
    return A


def transitionMatrix3(cg, x0=None, minstrength=0.1):
    """
    :param cg:
    :type cg:
    
    :param x0:
    :type x0:
    
    :param minstrength:
    :type minstrength: float
    
    :returns: 
    :rtype: 
    """
    A = graph2adj(cg)
    edges = np.where(A == 1)

    try:
        x = x0
    except AttributeError:
        A = initRandomMatrix(A, edges)
        x = A[edges]

    def objective(x):
        A[edges] = np.real(x)
        l = linalg.eig(A)[0]
        m = np.max(np.real(l*np.conj(l)))-0.99
        n = np.min(np.min(np.abs(x)), minstrength)-minstrength
        return m*m + 0.1*n*n

    o = np.zeros(len(edges))
    while np.min(np.abs(o[0])) < 0.8*minstrength:
        rpt = True
        while rpt:
            try:
                try:
                    o = optimize.fmin_bfgs(objective, x,
                                           gtol=1e-10, maxiter=100,
                                           disp=False, full_output=True)
                    A[edges] = np.real(o[0])
                    l = linalg.eig(A)[0]
                    if np.max(np.real(l*np.conj(l))) < 1:
                        rpt = False

                except:
                    rpt = True
            except Warning:
                x = np.randn(len(edges[0]))
                rpt = True
    A[edges] = np.real(o[0])
    return A


def initRandomMatrix(A, edges, distribution='beta'):
    """
    possible distributions:
    flat
    flatsigned
    beta
    normal
    uniform

    :param A:
    :type A:
    
    :param edges:
    :type edges:
    
    :param distribution: (GUESS)distribution from which to sample the weights. Available 
     options are flat, flatsigned, beta, normal, uniform
    :type distribution: string

    :returns: 
    :rtype: 
    """
    def init():
        if distribution == 'flat':
            x = np.ones(len(edges[0]))
        elif distribution == 'flatsigned':
            x = np.sign(np.random.randn(len(edges[0]))) * np.ones(len(edges[0]))
        elif distribution == 'beta':
            x = np.random.beta(0.5, 0.5, len(edges[0]))*3-1.5
        elif distribution == 'normal':
            x = np.random.randn(len(edges[0]))
        elif distribution == 'uniform':
            x = np.sign(np.randn(len(edges[0])))*np.rand(len(edges[0]))
        else:
            raise ValueError('Wrong option!')
        return x

    def eigenvalue(A):
        l = linalg.eig(A)[0]
        s = np.max(np.real(l*np.conj(l)))
        return s

    x = init()
    A[edges] = x
    s = eigenvalue(A)
    alpha = np.random.rand()*(0.99-0.8)+0.8
    A = A/(alpha*s)
    return A


def transitionMatrix4(g, minstrength=0.1, distribution='normal', maxtries=1000):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graph)
    
    :param minstrength:
    :type minstrength: float
    
    :param distribution: (GUESS)distribution from which to sample the weights. Available 
     options are flat, flatsigned, beta, normal, uniform
    :type distribution: string
    
    :param maxtries:
    :type maxtries: (guess)integer
    
    :returns: 
    :rtype: 
    """
    A = graph2adj(g)
    edges = np.where(A == 1)
    s = 2.0
    c = 0
    pbar = ProgressBar(widgets=['Searching for weights: ',
                                Percentage(), ' '],
                       maxval=maxtries).start()
    while s > 1.0:
        minstrength -= 0.001
        A = initRandomMatrix(A, edges, distribution=distribution)
        x = A[edges]
        delta = minstrength/np.min(np.abs(x))
        A[edges] = delta*x
        l = linalg.eig(A)[0]
        s = np.max(np.real(l*np.conj(l)))
        c += 1
        if c > maxtries:
            return None
        pbar.update(c)
    pbar.finish()

    return A


def drawsamplesLG(A, nstd=0.1, samples=100):
    """
    :param A:
    :type A:
    
    :param nstd:
    :type nstd: float
    
    :param samples:
    :type samples: integer
    
    :returns: 
    :rtype: 
    """
    n = A.shape[0]
    data = np.zeros([n, samples])
    data[:, 0] = nstd*np.random.randn(A.shape[0])
    for i in range(1, samples):
        data[:, i] = A @ data[:, i-1] + nstd*np.random.randn(A.shape[0])
    return data


def drawsamplesMA(A, nstd=0.1, samples=100, order=5):
    """
    :param A:
    :type A:
    
    :param nstd:
    :type nstd: float
    
    :param samples:
    :type samples: integer
    
    :param order:
    :type order: integer
    
    :returns: 
    :rtype: 
    """
    n = A.shape[0]
    data = np.zeros([n, samples])
    data[:, 0] = nstd*np.random.randn(A.shape[0])
    for i in range(1, samples):
        if i > order:
            result = 0
            for j in range(order):
                result += np.dot(1/(j+1)*A, data[:, i-1-j]) \
                  + nstd*np.dot(1/(j+1)*A, np.random.randn(A.shape[0]))
            data[:, i] = result
        else:
            data[:, i] = A @ data[:, i-1] \
              + nstd*np.random.randn(A.shape[0])
    return data


def getAgraph(n, mp=2, st=0.5, verbose=True):
    """
    :param n:
    :type n:
    
    :param mp:
    :type mp: (guess)integer
    
    :param st:
    :type st: float
    
    :param verbose:
    :type verbose: boolean
    
    :returns: 
    :rtype: 
    """
    keeptrying = True
    while keeptrying:
        G = gk.rnd_CG(n, maxindegree=mp, force_connected=True)
        try:
            A = transitionMatrix2(G, minstrength=st)
            keeptrying = False
        except ValueError as e:
            if verbose:
                print("!!! Unable to find strong links for a stable matrix !!!")
                print("*** trying a different graph")
    return {'graph':      G,
            'transition': A,
            'converges':  len(bfu.call_undersamples(G))}


def getAring(n, density=0.1, st=0.5, verbose=True, dist='flatsigned'):
    """
    :param n:
    :type n:
    
    :param density: (guess)ratio of total nodes to n^2 possible nodes
    :type density: float
    
    :param st:
    :type st: float
    
    :param verbose:
    :type verbose: boolean
    
    :param dist:
    :type dist: string
    
    :returns: 
    :rtype: 
    """
    keeptrying = True
    plusedges = bfu.dens2edgenum(density, n)
    while keeptrying:
        G = gk.ringmore(n, plusedges)
        try:
            A = transitionMatrix4(G, minstrength=st, distribution=dist)
            try:
                s = A.shape
                keeptrying = False
            except AttributeError:
                keeptrying = True
        except ValueError:
            if verbose:
                print("!!! Unable to find strong links for a stable matrix !!!")
                print("*** trying a different graph")
    return {'graph':      G,
            'transition': A,
            'converges':  len(bfu.call_undersamples(G))}


# -------------------------------------------------------------------
# estimation
# -------------------------------------------------------------------

def scoreAGraph(G, data, x0=None):
    """
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graph)
    
    :param data:
    :type data:
    
    :param x0:
    :type x0:
    
    :returns: 
    :rtype: 
    """
    A, B = npG2SVAR(G)
    K = np.sum(abs(A)+abs(B))
    a_idx = np.where(A != 0)
    b_idx = np.where(B != 0)
    if x0:
        o = optimize.fmin_bfgs(nllf, x0, args=(A, B, data, a_idx, b_idx),
                               disp=False, full_output=True)
    else:
        o = optimize.fmin_bfgs(nllf, np.randn(K),
                               args=(np.double(A), np.double(B),
                                     data, a_idx, b_idx),
                               disp=False, full_output=True)
    VARbic(o[1], K, data.shape[1])


def estimateG(G, YY, XX, YX, T, x0=None):
    """
    :param G: ``gunfolds`` format graph
    :type G: dictionary (``gunfolds`` graph)
    
    :param YY:
    :type YY:
    
    :param XX:
    :type XX:
    
    :param YX:
    :type YX:
    
    :param T:
    :type T:
    
    :param x0:
    :type x0:
    
    :returns: 
    :rtype: 
    """
    A, B = npG2SVAR(G)
    K = np.sum(abs(A)+abs(B))
    a_idx = np.where(A != 0)
    b_idx = np.where(B != 0)
    try:
        s = x0.shape
        x = x0
    except AttributeError:
        x = np.randn(K)
    o = optimize.fmin_bfgs(nllf2, x,
                           args=(np.double(A), np.double(B),
                                 YY, XX, YX, T, a_idx, b_idx),
                           disp=False, full_output=True)
    A, B = x2M(o[0], np.double(A), np.double(B), a_idx, b_idx)
    return A, B


def data2AB(data, x0=None):
    """
    :param data:
    :type data:
    
    :param x0:
    :type x0:
    
    :returns: 
    :rtype: 
    """
    n = data.shape[0]
    T = data.shape[1]
    YY = np.dot(data[:, 1:], data[:, 1:].T)
    XX = np.dot(data[:, :-1], data[:, :-1].T)
    YX = np.dot(data[:, 1:], data[:, :-1].T)

    model = VAR(data.T)
    r = model.fit(1)
    A = r.coefs[0, :, :]

    # A = np.ones((n,n))
    B = np.ones((n, n))
    np.fill_diagonal(B, 0)
    B[np.triu_indices(n)] = 0
    K = np.sum(abs(B)).astype(int)  # abs(A)+abs(B)))

    a_idx = np.where(A != 0)
    b_idx = np.where(B != 0)
    np.fill_diagonal(B, 1)

    try:
        s = x0.shape
        x = x0
    except AttributeError:
        x = np.r_[A.flatten(), 0.1*np.random.randn(K)]
    o = optimize.fmin_bfgs(nllf2, x,
                           args=(np.double(A), np.double(B),
                                 YY, XX, YX, T, a_idx, b_idx),
                           gtol=1e-12, maxiter=500,
                           disp=False, full_output=True)
    A, B = x2M(o[0], np.double(A), np.double(B), a_idx, b_idx)
    B = B+B.T
    return A, B


def amap(f, a):
    """
    :param f:
    :type f:
    
    :param a:
    :type a:
    
    :returns: 
    :rtype: 
    """
    v = np.vectorize(f)
    return v(a)


def AB2intAB(A, B, th=0.09):
    """
    :param A:
    :type A:
    
    :param B:
    :type B:
    
    :param th: (GUESS)threshold for discarding edges in A and B
    :type th: float
    
    :returns: 
    :rtype: 
    """
    A[amap(lambda x: abs(x) > th, A)] = 1
    A[amap(lambda x: abs(x) < 1, A)] = 0
    B[amap(lambda x: abs(x) > th, B)] = 1
    B[amap(lambda x: np.abs(x) < 1, B)] = 0
    np.fill_diagonal(B, 0)
    return A, B


def data2graph(data, x0=None, th=0.0):
    """
    :param data:
    :type data:
    
    :param x0:
    :type x0:
    
    :returns: 
    :rtype: 
    """
    A, B = data2AB(data, x0=x0)
    Ab = A.copy()
    Bb = B.copy()
    A, B = AB2intAB(A, B, th=th)
    return adjs2graph(A, B), Ab, Bb


def data2VARgraph(data, pval=0.05):
    """
    :param data:
    :type data:
    
    :param pval:
    :type pval: float
    
    :returns: 
    :rtype: 
    """
    model = VAR(data.T)
    r = model.fit(1)
    A = r.coefs[0, :, :]
    n = A.shape[0]
    g = {i: {} for i in range(1, n+1)}

    for i in range(n):
        for j in range(n):
            if np.abs(A[j, i]) > pval:
                g[i+1][j+1] = 1
    return g


# this is for the SAT solver project
def stableVAR(n, density=0.1, dist='beta'):
    """
    This function keeps trying to create a random graph and a random
    corresponding transition matrix until it succeeds.

    :param n: number of nodes in the graph
    :type n: (guess)integer
    
    :param density: ratio of total nodes to n^2 possible nodes
    :type density: (guess)float
        
    :param dist: distribution from which to sample the weights. Available 
     options are flat, flatsigned, beta, normal, uniform
    :type dist: (guess)string
    
    :returns: 
    :rtype: 
    """
    np.random.seed()
    sst = 0.9
    r = None
    while not r:
        r = getAring(n, density, sst, False, dist=dist)
        if sst < 0.03:
            sst -= 0.001
        else:
            sst -= 0.01
        if sst < 0:
            sst = 0.02
    return r['graph'], r['transition']


def genData(n, rate=2, density=0.1, burnin=100, ssize=2000, noise=0.1, dist='beta'):
    """
    Given a number of nodes this function randomly generates a ring
    SCC and the corresponding stable transition matrix. It tries until
    succeeds and for some graph densities and parameters of the
    distribution of transition matrix values it may take
    forever. Please play with the dist parameter to stableVAR. Then
    using this transition matrix it generates `ssize` samples of data
    and undersamples them by `rate` discarding the `burnin` number of
    samples at the beginning.

    :param n: number of nodes in the desired graph
    :type n: (guess)integer
    
    :param rate: undersampling rate (1 - no undersampling)
    :type rate: integer
    
    :param density: density of the graph to be generted
    :type density: (guess) float
    
    :param burnin: number of samples to discard since the beginning of VAR sampling
    :type burnin: integer
    
    :param ssize: how many samples to keep at the causal sampling rate
    :type ssize: (guess)integer
    
    :param noise: noise standard deviation for the VAR model
    :type noise: (guess)float
    
    :param dist: (GUESS)distribution from which to sample the weights. Available 
     options are flat, flatsigned, beta, normal, uniform
    :type dist: (guess)string
    
    :returns: 
    :rtype: 
    """
    g, Agt = stableVAR(n, density=density, dist=dist)
    data = drawsamplesMA(Agt, samples=burnin + ssize * 2, nstd=noise)
    data = data[:, burnin:]
    return g, Agt, data[:, ::rate]


def estimateSVAR(data, th=0.09):
    """
    :param data:
    :type data:
    
    :param th: (GUESS)threshold for discarding edges in A and B
    :type th: (guess)float
    
    :returns: 
    :rtype: 
    """
    A, B = data2AB(data)
    A, B = AB2intAB(A, B, th=th)
    return A, B


# option #1
def randomSVAR(n, rate=2, density=0.1, th=0.09, burnin=100,
               ssize=2000, noise=0.1, dist='beta'):
    """
    Given a number of nodes this function randomly generates a ring
    SCC and the corresponding stable transition matrix. It tries until
    succeeds and for some graph densities and parameters of the
    distribution of transition matrix values it may take
    forever. Please play with the dist parameter to stableVAR. Then
    using this transition matrix it generates `ssize` samples of data
    and undersamples them by `rate` discarding the `burnin` number of
    samples at the beginning. For these data the funcion solves the
    SVAR estimation maximizing log likelihood and returns the A and B
    matrices.

    :param n: number of nodes in the desired graph
    :type n: (guess)integer
    
    :param rate: undersampling rate (1 - no undersampling)
    :type rate: integer
    
    :param density: density of the graph to be generted
    :type density: (guess)float
        
    :param th: threshold for discarding edges in A and B
    :type th: (guess)float
    
    :param burnin: number of samples to discard since the beginning of VAR sampling
    :type burnin: (guess)integer
    
    :param ssize: how many samples to keep at the causal sampling rate
    :type ssize: (guess)integer
    
    :param noise: noise standard deviation for the VAR model
    :type noise: (guess)float
    
    :param dist: (GUESS)distribution from which to sample the weights. Available 
     options are flat, flatsigned, beta, normal, uniform
    :type dist: (guess)string
    
    :returns: 
    :rtype: 
    """
    g, Agt, data = genData(n, rate=rate, density=density,
                           burnin=burnin, ssize=ssize, noise=noise, dist=dist)
    A, B = estimateSVAR(data, th=th)
    return {'graph': g,
            'rate': rate,
            'graph@rate': bfu.undersample(g, rate-1),
            'transition': Agt,
            'estimate': adjs2graph(A, B),
            'directed': A,
            'bidirected': B
            }


# option #2
def noiseData(data, noise=0.1):
    """
    :param data:
    :type data:
    
    :param noise: (GUESS)noise standard deviation for the VAR model
    :type noise: (guess)float
    
    :returns: 
    :rtype: 
    """
    h, w = data.shape
    return data + np.random.randn(h, w)*noise


def decide_absences(As):
    """
    Given a list of binary matrices returns a binary mask for absence
    and presence of edges

    :param As: a list of binary matrices
    :type As: 
    
    :returns: 
    :rtype: 
    """
    M = np.zeros(As[0].shape).astype('int')
    M[np.where(np.sum(As, axis=0) > len(As)/2.0)] = 1
    return M


def presence_probs(As):
    """
    Given a list of binary matrices returns a frequency of edge
    presence

    :param As: a list of binary matrices
    :type As:
    
    :returns: 
    :rtype: 
    """
    n = len(As)
    M = np.sum([np.zeros(As[0].shape), np.ones(As[0].shape)]+As, axis=0)
    return M/(n+2.0)


def weight_and_mask(As):
    """
    Given a list o fbinary matrices returns a weight matrix for
    presences and absences and a mask to identify which are which

    :param As: a list of binary matrices
    :type As:
    
    :returns: 
    :rtype: 
    """
    M = decide_absences(As)
    W = presence_probs(As)
    A = np.ones(M.shape) - W  # ansence probs
    A[np.where(M == 1)] = W[np.where(M == 1)]
    return (1000 * (np.log(A) - np.log(1-A))).astype('int'), M


def randomSVARs(n, repeats=100, rate=2, density=0.1, th=0.09,
                burnin=100, ssize=2000, noise=0.1, strap_noise=0.1):
    """
    does what requested - help is on the way

    :param n: number of nodes in the desired graph
    :type n: integer
    
    :param repeats: how many times to add noise and re-estiamte
    :type repeats: integer
    
    :param rate: undersampling rate (1 - no undersampling)
    :type rate: integer
    
    :param density: density of the graph to be generted
    :type density: (guess)float
    
    :param th: threshold for discarding edges in A and B
    :type th: (guess)float
    
    :param burnin: number of samples to discard since the beginning of
          VAR sampling
    :type burnin: integer
    
    :param ssize: how many samples to keep at the causal sampling rate
    :type ssize: integer
    
    :param noise: noise standard deviation for the VAR model
    :type noise: float
    
    :param strap_noise: amount of noise for bootstrapping
    :type strap_noise: float
    
    :returns: 
    :rtype: 
    """
    g, Agt, data = genData(n, rate=rate, density=density,
                           burnin=burnin, ssize=ssize, noise=noise)

    As = []
    Bs = []
    A, B = estimateSVAR(data, th=th)
    As.append(A)
    Bs.append(B)

    for i in range(repeats-1):
        A, B = estimateSVAR(noiseData(data, noise=strap_noise), th=th)
        As.append(A)
        Bs.append(B)

    A = weight_and_mask(As)
    B = weight_and_mask(Bs)

    return {'graph': g,
            'rate': rate,
            'graph@rate': bfu.undersample(g, rate-1),
            'transition': Agt,
            'directed': A,
            'bidirected': B
            }

def check_matrix_powers(W, powers, threshold):
    """
     Check if the powers of a matrix W preserve a threshold for non-zero entries.

     :param W: The input matrix.
     :type W: array_like

     :param powers: List of powers to check.
     :type powers: iterable

     :param threshold: Threshold for non-zero entries.
     :type threshold: float

     :return: True if all powers of the matrix preserve the threshold for non-zero entries, False otherwise.
     :rtype: bool
     """
    for n in powers:
        W_n = np.linalg.matrix_power(W, n)
        non_zero_indices = np.nonzero(W_n)
        if (np.abs(W_n[non_zero_indices]) < threshold).any():
            return False
    return True


def create_stable_weighted_matrix(
    A,
    threshold=0.1,
    powers=[1, 2, 3, 4],
    max_attempts=1000,
    damping_factor=0.99,
    random_state=None,
):
    """
    Create a stable weighted matrix with a specified spectral radius.

    :param A: The input matrix.
    :type A: array_like

    :param threshold: Threshold for non-zero entries preservation. Default is 0.1.
    :type threshold: float, optional

    :param powers: List of powers to check in the stability condition. Default is [1, 2, 3, 4].
    :type powers: iterable, optional

    :param max_attempts: Maximum attempts to create a stable matrix. Default is 1000.
    :type max_attempts: int, optional

    :param damping_factor: Damping factor for scaling the matrix. Default is 0.99.
    :type damping_factor: float, optional

    :param random_state: Random seed for reproducibility. Default is None.
    :type random_state: int or None, optional

    :return: A stable weighted matrix.
    :rtype: array_like

    :raises ValueError: If unable to create a stable matrix after the maximum attempts.
    """
    np.random.seed(
        random_state
    )  # Set random seed for reproducibility if provided
    attempts = 0

    while attempts < max_attempts:
        # Generate a random matrix with the same sparsity pattern as A
        random_weights = np.random.randn(*A.shape)
        weighted_matrix = A * random_weights

        # Convert to sparse format for efficient eigenvalue computation
        weighted_sparse = sp.csr_matrix(weighted_matrix)

        # Compute the largest eigenvalue in magnitude
        eigenvalues, _ = eigs(weighted_sparse, k=1, which="LM")
        max_eigenvalue = np.abs(eigenvalues[0])

        # Scale the matrix so that the spectral radius is slightly less than 1
        if max_eigenvalue > 0:
            weighted_matrix *= damping_factor / max_eigenvalue
            # Check if the powers of the matrix preserve the threshold for non-zero entries of A
            if check_matrix_powers(weighted_matrix, powers, threshold):
                return weighted_matrix

        attempts += 1

    raise ValueError(
        f"Unable to create a matrix satisfying the condition after {max_attempts} attempts."
    )

# option #3
