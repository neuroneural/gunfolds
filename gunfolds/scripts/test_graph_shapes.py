from gunfolds.viz import dbn2latex as latex
from gunfolds.utils import zickle as zkl
import sys
d= zkl.load('graph_scc_node_50_2_25.zkl')
g = d[0]['gt']

sys.stdout = open("shipfig_figure.tex", "w")

latex.foldplot(g, steps=1, gap=5, R=10)
sys.stdout.close()
