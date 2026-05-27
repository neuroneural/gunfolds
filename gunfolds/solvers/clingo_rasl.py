"""This module contains clingo interaction functions"""
from __future__ import print_function
from string import Template
from gunfolds.utils.clingo import clingo
from gunfolds.utils.calc_procs import get_process_count
from gunfolds.conversions import g2clingo, rate, rasl_jclingo2g,\
     drasl_jclingo2g, clingo_preamble,\
     numbered_g2clingo, numbered_g2wclingo, encode_list_sccs

CLINGO_LIMIT = 64
PNUM = min(CLINGO_LIMIT, get_process_count(1))
CAPSIZE = 1

all_u_rasl_program = """
{edge(X,Y)} :- node(X), node(Y).
directed(X,Y,1) :- edge(X,Y).
directed(X,Y,L) :- directed(X,Z,L-1), edge(Z,Y), L <= U, u(U).
bidirected(X,Y,U) :- directed(Z,X,L), directed(Z,Y,L), node(X;Y;Z), X < Y, L < U, u(U).
countdirh(C):- C = #count { hdirected(X, Y): hdirected(X, Y), node(X), node(Y)}.
countbidirh(C):- C = #count { hbidirected(X, Y): hbidirected(X, Y), node(X), node(Y)}.
equald(L):- { directed(X,Y,L): hdirected(X,Y), node(X), node(Y) } == C, countdirh(C),u(L).
equalb(L):- { bidirected(X,Y,L): hbidirected(X,Y), node(X), node(Y) } == C, countbidirh(C),u(L).
equal(L) :- equald(L), equalb(L).
{trueu(L)} :- equal(L).
equaltest(M) :- 1 < {equal(_)}, equal(M).
min(M):- #min {MM:equaltest(MM)}=M, equaltest(_).
repeat(N):- min(M), equal(N), M<N.
:- directed(X, Y, L), not hdirected(X, Y), node(X), node(Y), trueu(L).
:- not directed(X, Y, L) , hdirected(X, Y), trueu(L).
:- bidirected(X, Y, L), not hbidirected(X, Y), node(X), node(Y), X < Y, trueu(L).
:- not bidirected(X, Y, L), hbidirected(X, Y), X < Y, trueu(L).
:- not trueu(_).
:- min(M), trueu(N), M<N.
    """

# The ASP formulation that does the heavylifting of the RASL encoding
# -------------------------------------------------------------------
# Generate powerset of all possible edges and for given edges produce directed
# edges up to the current undersampling and bidirected edges at the current
# undersampling.
drasl_program = """
    {edge1(X,Y)} :- node(X), node(Y).
    directed(X, Y, 1) :- edge1(X, Y).
    directed(X, Y, L) :- directed(X, Z, L-1), edge1(Z, Y), L <= U, u(U, _).
    bidirected(X, Y, U) :- directed(Z, X, L), directed(Z, Y, L), node(X;Y;Z), X < Y, L < U, u(U, _).
    """
# A set of no-go rules that compare the produces undersampled graph from the
# current element of the edge powerset to the input measurement timescae
# graph(s) and get rid of solutions that have a mismatch.
drasl_program += """
    :- directed(X, Y, L), not hdirected(X, Y, K), node(X;Y), u(L, K).
    :- bidirected(X, Y, L), not hbidirected(X, Y, K), node(X;Y), u(L, K), X < Y.
    :- not directed(X, Y, L), hdirected(X, Y, K), node(X;Y), u(L, K).
    :- not bidirected(X, Y, L), hbidirected(X, Y, K), node(X;Y), u(L, K), X < Y.
    """
# Filer out graphs that have already converged
# ----------------------------------------
# The next two sections are connected and are there to make sure that we do not
# count graphs more than once. Withoutthese sections if a converged or
# oscillating graph matches the measured graph H at multiple undersampling
# rates, all of them will be listed as unique solutions. The main idea is to
# first define a notequal operator, that works by checking individual edges,
# and then filter out solutions where this operator does not hold for at least
# one pair of undersampling rates.

# Turns out that the generating rules above produce all directed edges for each
# undersampling rate from 1 to u in each answer set, but only one set of
# bidirected edges that correspond to the current u in this answer. Often, it
# suffices to check repeatition of all directed edges to determine that a graph
# has converged. The main reason is persistence of bidirected edges that stay
# present once they appear at some undersampling rate. However, in some cases
# the directed edges converge before the bidirected edges do. This happens when
# the final set of directed edges generates at least one bidirected edge at the
# next step.

# The following two lines take advantage of the present history of all directed
# edges across undersampling rates up to u, and build a workaround for the
# absent history of bidirected edges. The first line is true if there was a
# fork with X and Y at the ends before the last step of undersampling. This
# would mean that the bidirected adge, if it exists at the current
# undersampling step, was already there at the last step as well. The second
# line sets notequal to true if at the current undersampling rate L a new
# bidirected edge appeared.
drasl_program += """
    pastfork(X,Y,L) :- directed(Z, X, K), directed(Z, Y, K), node(X;Y;Z), X < Y, K < L-1, uk(K), u(L, _).
    notequal(L-1,L) :- bidirected(X,Y,L), not pastfork(X,Y,L), node(X;Y), X < Y, u(L, _).
    """

# The following lines set the notequal condition for the cases when there is a
# mismatch in directed edges between rate K lower than L and L - the current
# undersampling rate.
drasl_program += """
    notequal(K,L) :- directed(X, Y, K), not directed(X, Y, L), node(X;Y), K<L, uk(K), u(L,_).
    notequal(K,L) :- not directed(X, Y, K), directed(X, Y, L), node(X;Y), K<L, uk(K), u(L,_).
    """

# This line filters out all solutions for which there is a K lower than the
# current undersampling rate and the graph at rate K equals to the one at rate
# L. This is a way to bypass our inability to specify the foroll quantifier
# directly in clingo.
drasl_program += """
    :- not notequal(K,L), K<L, uk(K), u(L,_).
    """

# The following section refuces to handle graphs if their compressed
# representation is a DAGs withuot forks. Too many options that are
# uninformative anyways and thre is no need to waste computation on them.
drasl_program += """
    nonempty(L) :- directed(X, Y, L), u(L,_).
    nonempty(L) :- bidirected(X, Y, L), u(L,_).
    :- not nonempty(L), u(L,_).
    """


def weighted_drasl_program(directed, bidirected):
    """
    Adjusts the optimization code based on the directed and bidirected priority

    :param directed: priority of directed edges in optimization
    :type directed: integer

    :param bidirected: priority of bidirected edges in optimization
        graph
    :type bidirected: integer

    :returns: optimization part of the ``clingo`` code
    :rtype: string
    """
    t = Template("""
    {edge1(X,Y)} :- node(X), node(Y).
    directed(X, Y, 1) :- edge1(X, Y).
    directed(X, Y, L) :- directed(X, Z, L-1), edge1(Z, Y), L <= U, u(U, _).
    bidirected(X, Y, U) :- directed(Z, X, L), directed(Z, Y, L), node(X;Y;Z), X < Y, L < U, u(U, _).

    :~ not directed(X, Y, L), hdirected(X, Y, W, K), node(X;Y), u(L, K). [W@$directed,X,Y]
    :~ not bidirected(X, Y, L), hbidirected(X, Y, W, K), node(X;Y), u(L, K), X < Y. [W@$bidirected,X,Y]
    :~ directed(X, Y, L), no_hdirected(X, Y, W, K), node(X;Y), u(L, K). [W@$directed,X,Y]
    :~ bidirected(X, Y, L), no_hbidirected(X, Y, W, K), node(X;Y), u(L, K), X < Y. [W@$bidirected,X,Y]

    pastfork(X,Y,L) :- directed(Z, X, K), directed(Z, Y, K), node(X;Y;Z), X < Y, K < L-1, uk(K), u(L, _).
    notequal(L-1,L) :- bidirected(X,Y,L), not pastfork(X,Y,L), node(X;Y), X < Y, u(L, _).
    notequal(K,L) :- directed(X, Y, K), not directed(X, Y, L), node(X;Y), K<L, uk(K), u(L,_).
    notequal(K,L) :- not directed(X, Y, K), directed(X, Y, L), node(X;Y), K<L, uk(K), u(L,_).
    :- not notequal(K,L), K<L, uk(K), u(L,_).
    nonempty(L) :- directed(X, Y, L), u(L,_).
    nonempty(L) :- bidirected(X, Y, L), u(L,_).
    :- not nonempty(L), u(L,_).
    """)
    return t.substitute(directed=directed, bidirected=bidirected)


def rate(u, uname='u'):
    """
    Adds under sampling rate to ``clingo`` code

    :param u: maximum under sampling rate
    :type u: integer

    :param uname: name of the parameter
    :type uname: string

    :returns: predicate for under sampling rate
    :rtype: string
    """
    s = "1 {" + uname + "(1.."+str(u)+")} 1."
    return s


def drate(u, gnum, weighted=False):
    """
    Replaces ``rate`` if there are multiple under sampled inputs

    :param u: maximum under sampling rate
    :type u: integer

    :param gnum: number of under sampled inputs
    :type gnum: integer

    :param weighted: whether the input graphs are weighted or
        precize.  If `True` but no weight matrices are provided -
        all weights are set to `1`
    :type weighted: boolean

    :returns: ``clingo`` code for under sampling with multiple under sampled inputs
    :rtype: string
    """
    s = f"1 {{u({int(weighted)+1}..{u}, {gnum})}} 1."
    return s


def rasl_command(g, urate=0):
    """
    Given a graph generates ``clingo`` code

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)

    :param urate: maximum undersampling rate to consider
    :type urate: integer

    :returns: ``clingo`` code
    :rtype: string
    """
    if not urate:
        urate = 1+3*len(g)
    command = g2clingo(g) + ' ' + rate(urate) + ' '
    command += '{edge(X,Y)} :- node(X), node(Y). ' + all_u_rasl_program + ' '
    command += "#show edge/2. "
    command += "#show trueu/1. "
    command += "#show min/1."
    command = command.encode().replace(b"\n", b" ")
    return command


def glist2str(g_list, weighted=False, dm=None, bdm=None):
    """
    Converts list of graphs into ``clingo`` predicates

    :param g_list: a list of graphs that are undersampled versions of
        the same system
    :type g_list: list of dictionaries (``gunfolds`` graphs)

    :param weighted: whether the input graphs are weighted or
        precize.  If `True` but no weight matrices are provided -
        all weights are set to `1`
    :type weighted: boolean

    :param dm: a list of n-by-n 2-d square matrix of the weights for
        directed edges of each input n-node graph
    :type dm: list of numpy arrays

    :param bdm: a list of *symmetric* n-by-n 2-d square matrix of the
        weights for bidirected edges of each input n-node graph
    :type bdm: list of numpy arrays

    :returns: ``clingo`` predicates as a string
    :rtype: string
    """
    if dm is None:
        dm = [None]*len(g_list)
    else:
        dm = [nd.astype('int') for nd in dm]
    if bdm is None:
        bdm = [None]*len(g_list)
    else:
        bdm = [nd.astype('int') for nd in bdm]
    s = ''
    for count, (g, D, B) in enumerate(zip(g_list, dm, bdm)):
        if weighted:
            s += numbered_g2wclingo(g, count+1, directed_weights_matrix=D, bidirected_weights_matrix=B) + ' '
        else:
            s += numbered_g2clingo(g, count+1) + ' '
    return s


def drasl_command(g_list, max_urate=0, weighted=False, scc=False, scc_members=None, dm=None, bdm=None, edge_weights=(1, 1)):
    """
    Given a list of graphs generates ``clingo`` codes

    :param g_list: a list of graphs that are undersampled versions of
        the same system
    :type g_list: list of dictionaries (``gunfolds`` graphs)

    :param max_urate: maximum under sampling rate
    :type max_urate: integer

    :param weighted: whether the input graphs are weighted or
        precize.  If ``True`` but no weight matrices are provided -
        all weights are set to ``1``
    :type weighted: boolean

    :param scc: whether to assume that each SCC in the input graph is
        either a singleton or have ``gcd=1``.  If `True` a much more
        efficient algorithm is employed.
    :type scc: (GUESS)boolean

    :param scc_members: a list of sets for nodes in each SCC
    :type scc_members: list

    :param dm: a list of n-by-n 2-d square matrix of the weights for
        directed edges of each input n-node graph
    :type dm: list of numpy arrays

    :param bdm: a list of *symmetric* n-by-n 2-d square matrix of the
        weights for bidirected edges of each input n-node graph
    :type bdm: list of numpy arrays

    :param edge_weights: a tuple of 2 values, the first is importance of matching
        directed weights when solving optimization problem and the second is for bidirected.
    :type edge_weights: tuple with 2 elements

    :returns: clingo code as a string
    :rtype: string
    """
    if dm is not None:
        dm = [nd.astype('int') for nd in dm]
    if bdm is not None:
        bdm = [nd.astype('int') for nd in bdm]

    assert len({len(g) for g in g_list}) == 1, "Input graphs have variable number of nodes!"

    if not max_urate:
        max_urate = 1+3*len(g_list[0])
    n = len(g_list)
    command = clingo_preamble(g_list[0])
    if scc:
        command += encode_list_sccs(g_list, scc_members)
    command += f"dagl({len(g_list[0])-1}). "
    command += glist2str(g_list, weighted=weighted, dm=dm, bdm=bdm) + ' '   # generate all graphs
    command += 'uk(1..'+str(max_urate)+').' + ' '
    command += ' '.join([drate(max_urate, i+1, weighted=weighted) for i in range(n)]) + ' '
    command += weighted_drasl_program(edge_weights[0], edge_weights[1]) if weighted else drasl_program
    command += f":- M = N, {{u(M, 1..{n}); u(N, 1..{n})}} == 2, u(M, _), u(N, _). "
    command += "#show edge1/2. "
    command += "#show u/2."
    command = command.encode().replace(b"\n", b" ")
    return command


def drasl(glist, capsize=CAPSIZE, timeout=0, urate=0, weighted=False, scc=False, scc_members=None, dm=None,
          bdm=None, pnum=PNUM, edge_weights=(1, 1), configuration="crafty", optim='optN'):
    """
    Compute all candidate causal time-scale graphs that could have
    generated all undersampled graphs at all possible undersampling
    rates up to ``urate`` in ``glist`` each at an unknown undersampling
    rate.

    :param glist: a list of graphs that are undersampled versions of
        the same system
    :type glist: list of dictionaries (``gunfolds`` graphs)

    :param capsize: maximum number of candidates to return
    :type capsize: integer

    :param timeout: timeout in seconds after which to interrupt
        computation (0 - no limit)
    :type timeout: integer

    :param urate: maximum undersampling rate to consider
    :type urate: integer

    :param weighted: whether the input graphs are weighted or
        imprecize.  If ``True`` but no weight matrices are provided -
        all weights are set to ``1``
    :type weighted: boolean

    :param scc: whether to assume that each SCC in the input graph is
        either a singleton or have ``gcd=1``.  If ``True`` a much more
        efficient algorithm is employed.
    :type scc: boolean

    :param scc_members: a list of sets for nodes in each SCC
    :type scc_members: list

    :param dm: a list of n-by-n 2-d square matrix of the weights for
        directed edges of each input n-node graph
    :type dm: list of numpy arrays

    :param bdm: a list of *symmetric* n-by-n 2-d square matrix of the
        weights for bidirected edges of each input n-node graph
    :type bdm: list of numpy arrays

    :param pnum: number of parallel threads to run ``clingo`` on
    :type pnum: integer

    :param edge_weights: a tuple of 2 values, the first is importance
        of matching directed weights when solving optimization problem and the second is for bidirected.
    :type edge_weights: tuple with 2 elements

    :param configuration: Select configuration based on problem type

        - ``frumpy`` : Use conservative defaults
        - ``jumpy`` : Use aggressive defaults
        - ``tweety`` : Use defaults geared towards asp problems
        - ``handy`` : Use defaults geared towards large problems
        - ``crafty`` : Use defaults geared towards crafted problems
        - ``trendy`` : Use defaults geared towards industrial problems
    :type configuration: string

    :param optim: a comma separated string containing configuration for optimization algorithm and optionally a bound [<arg>[, <bound>]]
        
        - <arg> : <mode {opt|enum|optN|ignore}>
            - ``opt`` : Find optimal model
            - ``enum`` : Find models with costs <= <bound>
            - ``optN`` : Find optimum, then enumerate optimal models
            - ``ignore`` : Ignore optimize statements
        - <bound> : Set initial bound for objective function(s)
    :type optim: string
    
    :returns: results of parsed equivalent class
    :rtype: dictionary
    """
    if dm is not None:
        dm = [nd.astype('int') for nd in dm]
    if bdm is not None:
        bdm = [nd.astype('int') for nd in bdm]
    if not isinstance(glist, list):
        glist = [glist]
    return clingo(drasl_command(glist, max_urate=urate, weighted=weighted,
                                scc=scc, scc_members=scc_members, dm=dm, bdm=bdm, edge_weights=edge_weights),
                  capsize=capsize, convert=drasl_jclingo2g, configuration=configuration,
                  timeout=timeout, exact=not weighted, pnum=pnum, optim=optim)


def rasl(g, capsize, timeout=0, urate=0, pnum=None, configuration="tweety"):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)

    :param capsize: maximum number of candidates to return
    :type capsize: integer

    :param timeout: timeout in seconds after which to interrupt
        computation (0 - no limit)
    :type timeout: integer

    :param urate: maximum undersampling rate to consider
    :type urate: integer

    :param pnum: number of parallel threads to run ``clingo`` on
    :type pnum: integer

    :param configuration: Select configuration based on problem type

        - ``frumpy`` : Use conservative defaults
        - ``jumpy`` : Use aggressive defaults
        - ``tweety`` : Use defaults geared towards asp problems
        - ``handy`` : Use defaults geared towards large problems
        - ``crafty`` : Use defaults geared towards crafted problems
        - ``trendy`` : Use defaults geared towards industrial problems
    :type configuration: string

    :returns: results of parsed equivalent class
    :rtype: dictionary
    """
    return clingo(rasl_command(g, urate=urate), capsize=capsize, configuration=configuration, convert=rasl_jclingo2g, timeout=timeout, pnum=pnum)
