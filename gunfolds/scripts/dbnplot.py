""" This is a use-case of the tools in the tools directory. The
    example defines a graph and shows how to generate a figure that
    shows the graph at different undersampling rates. Running the file
    in python (python dbnplot.py) generates a figure in figures
    folder: shipfig.pdf
"""
import numpy as np
import os, sys
import gunfolds.utils.zickle as zkl
import gunfolds.viz.dbn2latex as d2l


def listplot(fname, mname='JJ', stl='', width=5):
    l = zkl.load(fname)
    y = min(width, len(l))
    x = np.int(np.ceil(len(l) / float(y)))
    d2l.matrix_list(l, y, x, R=2, w_gap=1, h_gap=2, mname=mname, stl=stl)

if __name__ == '__main__':

    g = {1: {2: 1, 7: 1},
         2: {3: 1, 4: 1, 6: 1, 7: 1},
         3: {4: 1},
         4: {1: 1, 4: 1, 5: 1},
         5: {5: 1, 6: 1, 8: 1, 9: 1, 10: 1},
         6: {2: 1, 7: 1},
         7: {8: 1},
         8: {4: 1, 7: 1, 8: 1, 9: 1},
         9: {1: 1, 2: 1, 6: 1, 7: 1, 10: 1},
         10: {1: 1, 5: 1, 9: 1}}

    # output file
    foo = open('gunfolds/figures/shipfig_figure.tex', 'wb')
    sys.stdout = foo

    # generation of the output
    g = {1: {2: 1}, 2: {1: 1, 3: 1}, 3: {4: 1}, 4: {1: 1}}

    # d2l.matrix_unfold(l[0],2,1,R=5, w_gap=1, h_gap=2, mname='TT1')
    listplot('list.zkl', width=5)

    sys.stdout = sys.__stdout__ # remember to reset sys.stdout!
    foo.flush()
    foo.close()
    PPP = os.getcwd()
    os.chdir('figures')
    os.system('pdflatex --shell-escape shipfig.tex 2>&1 > /dev/null')
    os.chdir(PPP)
