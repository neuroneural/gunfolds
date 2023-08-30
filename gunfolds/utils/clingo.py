""" This module contains clingo interaction functions """
from __future__ import print_function
from gunfolds.conversions import msl_jclingo2g
import clingo as clngo
import json
from gunfolds.utils.calc_procs import get_process_count

CAPSIZE = 1000
CLINGO_LIMIT = 64
PNUM = min(CLINGO_LIMIT, get_process_count(1))


def run_clingo(command,
               exact=True,
               timeout=0,
               capsize=CAPSIZE,
               pnum=None):
    """
    Open sub-process and run clingo

    :param command: Completed clingo code
    :type command: string

    :param exact: If true, run clingo in exact mode. If false, run clingo in optimization mode
    :type exact: boolean

    :param timeout: timeout in seconds after which to interrupt
        computation (0 - no limit)
    :type timeout: integer

    :param capsize: maximum number of candidates to return
    :type capsize: integer

    :param cpath: clingo path
    :type cpath: string

    :param pnum: number of parallel threads to run clingo on
    :type pnum: integer

    :returns: results of equivalent class
    :rtype: dictionary
    """
    if pnum is None:
        pnum = PNUM
    ctrl = clngo.Control(
        ["--warn=no-atom-undefined", "--configuration=tweety", "-t", str(int(pnum)) + ",split", "-n", str(capsize)])
    if not exact:
        ctrl.configuration.solve.opt_mode = "opt"
    ctrl.add("base", [], command.decode())
    ctrl.ground([("base", [])])
    models = []
    with ctrl.solve(yield_=True, async_=True) as handle:
        while not handle.wait(timeout): pass
        for model in handle:
            models.append([str(atom) for atom in model.symbols(shown=True)])
    cost = ctrl.statistics["summary"]["costs"]
    num_opt = ctrl.statistics["summary"]["models"]["optimal"]
    if not exact:
        if num_opt == 0.0:
            return {}, cost
        else:
            return models, cost
    return models, cost


def clingo(command, exact=True,
           convert=msl_jclingo2g,
           timeout=0,
           capsize=CAPSIZE,
           pnum=None):
    """
    Runs ``run_clingo`` and returns parsed equivalent class

    :param command: Completed clingo code
    :type command: string

    :param exact: If true, run clingo in exact mode. If false, run clingo in optimization mode
    :type exact: boolean

    :param convert: result parsing protocol
    :type convert: function

    :param timeout: timeout in seconds after which to interrupt
        computation (0 - no limit)
    :type timeout: integer

    :param capsize: maximum number of candidates to return
    :type capsize: integer

    :param cpath: clingo path
    :type cpath: string

    :param pnum: number of parallel threads to run clingo on
    :type pnum: integer

    :returns: results of parsed equivalent class
    :rtype: dictionary
    """
    result = run_clingo(command,
                        exact=exact,
                        timeout=timeout,
                        capsize=capsize,
                        pnum=pnum)
    if result[0] == {} or result[0] == []:
        return {}
    else:
        if not exact:
            r = (convert(result[0][-1]), result[1])
        else:
            r = {convert(value) for value in result[0]}
    return r
