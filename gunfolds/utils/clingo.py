""" This module contains clingo interaction functions """
from __future__ import print_function
from gunfolds.conversions import drasl_jclingo2g
import clingo as clngo
from gunfolds.utils.calc_procs import get_process_count

CLINGO_LIMIT = 64
PNUM = min(CLINGO_LIMIT, get_process_count(1))
CAPSIZE = 1

def run_clingo(command,
               exact=True,
               timeout=0,
               capsize=CAPSIZE,
               configuration="tweety",
               pnum=PNUM,
               optim='optN'):
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

        - ``frumpy`` : Use conservative defaults
        - ``jumpy`` : Use aggressive defaults
        - ``tweety`` : Use defaults geared towards asp problems
        - ``handy`` : Use defaults geared towards large problems
        - ``crafty`` : Use defaults geared towards crafted problems
        - ``trendy`` : Use defaults geared towards industrial problems
    :type configuration: string


    :param pnum: number of parallel threads to run clingo on
    :type pnum: integer

    :param optim: a comma separated string containing configuration for optimization algorithm and optionally a bound [<arg>[, <bound>]]
        
        - <arg> : <mode {opt|enum|optN|ignore}>
            - ``opt`` : Find optimal model
            - ``enum`` : Find models with costs <= <bound>
            - ``optN`` : Find optimum, then enumerate optimal models
            - ``ignore`` : Ignore optimize statements
        - <bound> : Set initial bound for objective function(s)
    :type optim: string

    :returns: results of equivalent class
    :rtype: dictionary
    """
    assert len(optim.split(',')) < 3, "optim option only takes 1 or 2 comma-separated parameters"

    clingo_control = ["--warn=no-atom-undefined", "--configuration=", configuration, "-t", str(int(pnum)) + ",split", "-n", str(capsize)]
    ctrl = clngo.Control(clingo_control)
    if not exact:
        ctrl.configuration.solve.opt_mode = optim
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
    

def clingo(command, exact=True,
           convert=drasl_jclingo2g,
           timeout=0,
           capsize=CAPSIZE,
           configuration="crafty",
           optim='optN',
           pnum=PNUM):
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

    :param pnum: number of parallel threads to run clingo on
    :type pnum: integer

    :returns: results of parsed equivalent class
    :rtype: dictionary
    """
    result = run_clingo(command,
                        exact=exact,
                        timeout=timeout,
                        capsize=capsize,
                        configuration=configuration,
                        pnum=pnum,
                        optim=optim)
    if result[0] == {} or result[0] == []:
        return {}
    else:
        if not exact:
            r = {(convert(value[0]), sum(value[1])) for value in result[0]}
        else:
            r = {convert(value[0]) for value in result[0]}
    return r
