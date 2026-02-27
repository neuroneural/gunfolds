from collections import Counter
import functools
from gunfolds.utils.bfutils import *

nodes = 4  # number of nodes in the graph
PNUM = 75  # number of processes to use

template = "{0:9}{1:9}{2:9}{3:9}{4:9}{5:10}"


def wrapper_c(fold):
    return icompat(fold, nodes)


def wrapper_l(fold):
    return ilength(fold, nodes)


def wrapper_list(fold):
    return iall(fold, nodes)


def wrapper_u(fold, steps):
    return cc_all(fold, nodes, steps)


def make_rect(l):
    max_seq = max(map(len, l))
    for e in l:
        e += [e[-1]] * (max_seq - len(e))
    return l


def wrapper_unique(fold):
    counter = 0
    for i in results[number]:
        if i not in unique_appeared_graph[counter]:
            unique_appeared_graph[counter].append(i)
        counter += 1
    return 1


def wrapper_non_eliminate(fold):
    return tuple(results[fold][counter])

if __name__ == '__main__':
    resultsmp = True
    results = []
    pool = Pool(processes=PNUM)

    print(template.format(*('        u', ' all_uniq', '   unique', '  seen_Gu', ' converge',  ' uconverge')))
    cumset = set()
    clen = 0
    for s in range(23):

        results = pool.map(functools.partial(wrapper_u, steps=s),
                           range(2 ** (nodes ** 2)))

        converged = len([e for e in results if e == []])
        notconverged = len(results) - converged

        results = filter(None, results)
        r = set(results)
        d = r.difference(cumset)
        cumset = cumset.union(r)

        cl = 2 ** (nodes ** 2) - len(results) - clen
        clen += cl

        print (template.format(*(s, len(r), len(d), len(cumset),
                                converged, notconverged)))

    pool.close()
    pool.join()
