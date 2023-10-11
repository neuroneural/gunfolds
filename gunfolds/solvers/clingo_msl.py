""" This module contains clingo interaction functions """
from __future__ import print_function
from gunfolds.utils.clingo import clingo
from gunfolds.conversions import g2clingo, g2wclingo, msl_jclingo2g
import time


uprogram = """
{ edge1(X,Y) } :- node(X),node(Y).
edge(X,Y,1) :- edge1(X,Y).
edge(X,Y,L) :- edge(X,Z,L-1),edge1(Z,Y), L <= U, u(U).
derived_edgeu(X,Y) :- edge(X,Y,L), u(L).
derived_confu(X,Y) :- edge(Z,X,L), edge(Z,Y,L),node(X),node(Y),node(Z),X < Y, L < U, u(U).
:- edgeu(X,Y), not derived_edgeu(X,Y),node(X),node(Y).
:- not edgeu(X,Y), derived_edgeu(X,Y),node(X),node(Y).
:- derived_confu(X,Y), not confu(X,Y),node(X),node(Y), X < Y.
:- not derived_confu(X,Y), confu(X,Y),node(X),node(Y), X < Y.
    """
wuprogram = """
    { edge1(X,Y) } :- node(X), node(Y).
    path(X,Y,1) :- edge1(X,Y).
    path(X,Y,L) :- path(X,Z,L-1), edge1(Z,Y), L <= U, u(U).
    edgeu(X,Y) :- path(X,Y,L), u(L).
    confu(X,Y) :- path(Z,X,L), path(Z,Y,L), node(X;Y;Z), X < Y, L < U, u(U).
    :~ edgeh(X,Y,W), not edgeu(X,Y). [W,X,Y,1]
    :~ no_edgeh(X,Y,W), edgeu(X,Y). [W,X,Y,1]
    :~ confh(X,Y,W), not confu(X,Y). [W,X,Y,2]
    :~ no_confh(X,Y,W), confu(X,Y). [W,X,Y,2]
    """


def rate(u):
    """
    :param u: maximum under sampling rate 
    :type u: integer
    
    :returns: predicate for under sampling rate
    :rtype: string
    """
    s = "u("+str(u)+")."
    return s


def g2clingo_msl(g):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :returns: 
    :rtype: 
    """
    return g2clingo(g, directed='edgeu', bidirected='confu', both_bidirected=True)


def msl_command(g, urate=2, exact=True):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
  
    :param urate: maximum undersampling rate to consider
    :type urate: integer
    
    :param exact: If true, run clingo in exact mode. If false, run clingo in optimization mode 
    :type exact: boolean
    
    :returns: 
    :rtype: 
    """
    if exact:
        command = rate(urate) + ' ' + g2clingo_msl(g) + ' ' + uprogram
    else:
        command = rate(urate) + ' ' + g2wclingo(g) + ' ' + wuprogram
    command = command.encode().replace(b"\n", b" ")
    return command


def msl(g, capsize, exact=True, configuration="tweety", urate=2, timeout=0, pnum=None):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param capsize: maximum number of candidates to return
    :type capsize: integer
    
    :param exact: If true, run clingo in exact mode. If false, run clingo in optimization mode 
    :type exact: boolean
    
    :param urate: maximum undersampling rate to consider
    :type urate: integer
    
    :param timeout: timeout in seconds after which to interrupt
        computation (0 - no limit)
    :type timeout: integer

    :param configuration: Select configuration based on problem type

        - ``frumpy`` : Use conservative defaults
        - ``jumpy`` : Use aggressive defaults
        - ``tweety`` : Use defaults geared towards asp problems
        - ``handy`` : Use defaults geared towards large problems
        - ``crafty`` : Use defaults geared towards crafted problems
        - ``trendy`` : Use defaults geared towards industrial problems
    :type configuration: string
    
    :param pnum: number of parallel threads to run ``clingo`` on
    :type pnum: integer
    
    :returns: 
    :rtype: 
    """
    return clingo(msl_command(g, urate=urate, exact=True),
                  capsize=capsize, convert=msl_jclingo2g, timeout=timeout, configuration=configuration, pnum=pnum)


def rasl_msl(g, capsize, urate=2, timeout=0, pnum=None):
    """
    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)
    
    :param capsize: maximum number of candidates to return
    :type capsize: integer
    
    :param urate: maximum undersampling rate to consider
    :type urate: integer
    
    :param timeout: timeout in seconds after which to interrupt
        computation (0 - no limit)
    :type timeout: integer
    
    :param pnum: number of parallel threads to run ``clingo`` on
    :type pnum: integer
    
    :returns: 
    :rtype: 
    """
    r = set()
    remain_time = timeout
    for i in range(2, urate + 1):
        startTime = int(round(time.time()))
        k = msl(g, capsize, urate=i, pnum=pnum, timeout=remain_time, exact=True)
        endTime = int(round(time.time()))
        remain_time = max(0, remain_time - (endTime - startTime))
        r = r | {(x[0], i) for x in k}
        capsize = max(0, capsize - len(r))
        if not (capsize and remain_time):
            break
    return r
