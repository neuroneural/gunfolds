""" This module contains clingo interaction functions """
from __future__ import print_function
from gunfolds.conversions import msl_jclingo2g
import clingo as clngo
import json
from gunfolds.utils.calc_procs import get_process_count

CAPSIZE = 1000
CLINGO_LIMIT = 64
PNUM = min(CLINGO_LIMIT, get_process_count(1))


import subprocess
import os
from gunfolds.conversions import rasl_jclingo2g
import json
from gunfolds.utils.calc_procs import get_process_count
CLINGOPATH=''

def clingo_high_version(cpath=CLINGOPATH):
    """
    (Ask it is not used)

    :param cpath: clingo path 
    :type cpath: string
    
    :returns: 
    :rtype: 
    """
    v = os.popen(cpath+"clingo --version").read()[15:20]
    #v = subprocess.check_output(['clingo', '--version'])[15:20]
    return int(v.split('.')[1]) >= 5

def run_clingo_old(command,
               exact=True,
               timeout=0,
               capsize=CAPSIZE,
               cpath=CLINGOPATH,
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
    if pnum is None: pnum = PNUM
    clg_start = 'clingo -W no-atom-undefined --configuration=tweety '
    clingo_command = cpath+clg_start+'-t '+str(int(pnum))\
        +',split --outf=2 --time-limit='+str(timeout)\
        +' -n '+str(capsize)+' '

    if not exact:
        clingo_command += ' --opt-mode=opt '
    try:
        p = subprocess.Popen(clingo_command.split(),
                             stdin=subprocess.PIPE,
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             close_fds=True)
    except:
        return {}
    (output, err) = p.communicate(command)

    if not err:
        result = json.loads(output.decode())
    else:
        if not any([x in err for x in [b'*** Warn', b'*** Info']]):
            print(err)
            return {}
        else:
            result = json.loads(output.decode())
    return result


def run_clingo(command,
               exact=True,
               timeout=0,
               capsize=CAPSIZE,
               configuration="tweety",
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

    :returns: results of equivalent class
    :rtype: dictionary
    """
    if pnum is None:
        pnum = PNUM
    ctrl = clngo.Control(["--warn=no-atom-undefined","--configuration=", configuration, "-t", str(int(pnum)) + ",split", "-n", str(capsize)])
    if not exact:
        ctrl.configuration.solve.opt_mode = "opt"
    ctrl.add("base", [], command.decode())
    ctrl.ground([("base", [])])
    models = []
    with ctrl.solve(yield_=True, async_=True) as handle:
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

def clingo_old(command, exact=True,
           convert=msl_jclingo2g,
           timeout=0,
           capsize=CAPSIZE,
           cpath=CLINGOPATH,
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

    exp_result = 'SATISFIABLE'
    if not exact:
        exp_result = 'OPTIMUM FOUND'

    result = run_clingo_old(command,
        exact=exact,
        timeout=timeout,
        capsize=capsize,
        cpath=cpath,
        pnum=pnum)

    if result['Result'] == exp_result:
        if exact:
            r = {convert(value['Value']) for value in result['Call'][0]['Witnesses']}
        else:
            r = convert(result['Call'][0]['Witnesses'][-1]['Value'])
        return r
    return {}

def clingo(command, exact=True,
           convert=msl_jclingo2g,
           timeout=0,
           capsize=CAPSIZE,
           configuration="tweety",
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
                        configuration=configuration,
                        pnum=pnum)
    if result[0]=={} or result[0]==[]:
        return {}
    else:
        if not exact:
            r = (convert(result[0][-1]), result[1])
        else:
            r = {convert(value) for value in result[0]}
    return r
    