from multiprocessing import cpu_count
import socket


def get_process_count(starts):
    """ 
    Using the number of starts per graph and machine info, calculate
    the number of processes to spawn 
    
    :param starts:
    :type starts:
    
    :returns: 
    :rtype: 
    """
    if socket.gethostname().split('.')[0] == 'leibnitz':
        num_processes = 30
        num_processes = max((1, num_processes / starts))
    elif socket.gethostname().split('.')[0] == 'mars':
        num_processes = 21
        num_processes = max((1, num_processes / starts))
    elif socket.gethostname().split('.')[0] == 'saturn':
        num_processes = 12
        num_processes = max((1, num_processes / starts))
    elif socket.gethostname().split('.')[0] == 'hooke':
        num_processes = 22
        num_processes = max((1, num_processes / starts))
    else:
        # Setting the number  of parallel running processes  to the number
        # of cores minus 7% for breathing room
        num_processes = cpu_count() - int(0.07 * cpu_count())
        num_processes = max((1, num_processes / starts))
        
    return num_processes
