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
               configuration="tweety",
               pnum=None,
               optim=None):
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

    :param configuration: Select configuration based on problem type
        frumpy: Use conservative defaults
        jumpy : Use aggressive defaults
        tweety: Use defaults geared towards asp problems
        handy : Use defaults geared towards large problems
        crafty: Use defaults geared towards crafted problems
        trendy: Use defaults geared towards industrial problems
    :type configuration: string

    :param cpath: clingo path
    :type cpath: string

    :param pnum: number of parallel threads to run clingo on
    :type pnum: integer

    :param optim: a list containing configuration for optimization algorithm and optionally a bound  [<arg>, <bound>]
        <arg>: <mode {opt|enum|optN|ignore}>[,<bound>...]
        opt   : Find optimal model
        enum  : Find models with costs <= <bound>
        optN  : Find optimum, then enumerate optimal models
        ignore: Ignore optimize statements
        <bound> : Set initial bound for objective function(s)
    :type optim: list

    :returns: results of equivalent class
    :rtype: dictionary
    """
    if optim is None:
        optim = ['optN']
    if pnum is None:
        pnum = PNUM
    if len(optim) == 1:
        optim_option = optim[0]
    elif len(optim) == 2:
        optim_option = f"{optim[0]}, {optim[1]}"
    else:
        raise ValueError("The 'optim' list must have either one or two elements.")

    ctrl = clngo.Control(["--warn=no-atom-undefined","--configuration=", configuration, "-t", str(int(pnum)) + ",split", "-n", str(capsize)])
    if not exact:
        ctrl.configuration.solve.opt_mode = optim_option
    ctrl.add("base", [], command.decode())
    ctrl.ground([("base", [])])
    models = []
    with ctrl.solve(yield_=True, async_=True) as handle:
        for model in handle:
            models.append(([str(atom) for atom in model.symbols(shown=True)], model.cost))
    cost = ctrl.statistics["summary"]["costs"]
    num_opt = ctrl.statistics["summary"]["models"]["optimal"]
    if not exact:
        if num_opt == 0.0:
            return {}, cost
        else:
            return models, cost 
    return models, cost
    

def  clingo(command, exact=True,
           convert=msl_jclingo2g,
           timeout=0,
           capsize=CAPSIZE,
           configuration="crafty",
           optim=None,
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

    :param configuration: Select configuration based on problem type
        frumpy: Use conservative defaults
        jumpy : Use aggressive defaults
        tweety: Use defaults geared towards asp problems
        handy : Use defaults geared towards large problems
        crafty: Use defaults geared towards crafted problems
        trendy: Use defaults geared towards industrial problems
    :type configuration: string

    :param optim: a list containing configuration for optimization algorithm and optionally a bound  [<arg>, <bound>]
        <arg>: <mode {opt|enum|optN|ignore}>[,<bound>...]
        opt   : Find optimal model
        enum  : Find models with costs <= <bound>
        optN  : Find optimum, then enumerate optimal models
        ignore: Ignore optimize statements
        <bound> : Set initial bound for objective function(s)
    :type optim: list

    :param cpath: clingo path
    :type cpath: string

    :param pnum: number of parallel threads to run clingo on
    :type pnum: integer

    :returns: results of parsed equivalent class
    :rtype: dictionary
    """
    if optim is None:
        optim = ['optN']
    result = run_clingo(command,
                        exact=exact,
                        timeout=timeout,
                        capsize=capsize,
                        configuration=configuration,
                        pnum=pnum,
                        optim=optim)
    if result[0]=={} or result[0]==[]:
        return {}
    else:
        if not exact:
            r = {(convert(value[0]), value[1][0]) for value in result[0]}
        else:
            r = {convert(value[0]) for value in result[0]}
    return r
