import graph_tool as gt
from graph_tool import draw as gtd
from graph_tool.draw.cairo_draw import position_parallel_edges
import gunfolds.utils.graphkit as gk
import gunfolds.utils.bfutils as bfu
import numpy as np
from gi.repository import Gdk
from gi import require_version
require_version("Gdk", "3.0")


def getscreensize():
    """
    Returns a tuple ``(w,h)`` with the largest width ``w`` and height ``h`` among all connected monitors.

    :returns: a tuple ``(w,h)``
    :rtype: a tuple with integer values
    """
    # from here https://stackoverflow.com/a/65894179
    screen = Gdk.Display.get_default()
    w = 0
    h = 0
    for x in range(0, screen.get_n_monitors()):
        xw = screen.get_monitor(x).get_geometry().width
        xh = screen.get_monitor(x).get_geometry().height
        if w < xw:
            w = xw
        if h < xh:
            h = xh
    return w, h


def circ_position(gg):
    """
    Generates node positions for graph-tool graph that arrange them in a ring

    :param gg: ``graph-tool`` graph object
    :type gg: a ``graph-tool`` object of class Graph
    """
    pos = gg.new_vertex_property("vector<double>")
    n = gg.num_vertices()
    s = 2.0*np.pi/n
    for v in range(n):
        idx = int(gg.vertex_properties['label'][gg.vertex(v)]) - 1
        pos[gg.vertex(v)] = (n * np.cos(s * idx),
                             n * np.sin(s * idx))
    gg.vertex_properties["pos"] = pos


def every_edge_control_points(n):
    """
    Generate a graph-tool Graph class object with all possible edges (a superclicue).

    The function generates a graph and populates all parameters for all edges.
    It first does not include the bidirected adges to compute the control_points for
    directed edges so that they all curve. Subsequently it adds bidirected (red) edges
    and resets their control points to have all bidirected edges be rendered as straight lines.

    :param n: number of nodes
    :type n: integer

    :returns: ``graph-tool`` graph
    :rtype: a ``graph-tool`` object of class Graph
    """

    g = gk.fullyconnected(n)

    gr = gt.Graph()

    vlabel = gr.new_vertex_property("string")
    ecolor = gr.new_edge_property("string")
    ep_width = gr.new_edge_property("float")
    em_size = gr.new_edge_property("float")

    verts = {}
    edges = {}

    for v in g:
        verts[v] = gr.add_vertex()
        vlabel[verts[v]] = str(v)
    gr.vertex_properties["label"] = vlabel
    circ_position(gr)

    for v in g:
        for w in g[v]:
            edges[(v, w)] = gr.add_edge(verts[v], verts[w])
            ecolor[edges[(v, w)]] = 'k'
            ep_width[edges[(v, w)]] = 1
            em_size[edges[(v, w)]] = 15

    control = position_parallel_edges(gr, gr.vertex_properties["pos"])

    for v in range(1, n+1):
        for w in range(v, n+1):
            edges[(v, w)] = gr.add_edge(verts[v], verts[w])
            ecolor[edges[(v, w)]] = 'r'
            ep_width[edges[(v, w)]] = 1
            em_size[edges[(v, w)]] = 1
            control[edges[(v, w)]] = []

    gr.edge_properties["color"] = ecolor
    gr.edge_properties["pen_width"] = ep_width
    gr.edge_properties["marker_size"] = em_size
    gr.edge_properties["control"] = control

    return gr


def colorcomponents(gr):
    """
    Assigns a color to each vertex to ensure that vertices from the same strongly connected component
    have the same color.

    By default 12 distinct colors from colorbrewer2 quantitative palette are used and anything beyond
    that is assigned a white color. In the future that behaviour may change to assigning random colors
    or to assigning useful semantics to color temperature or shade (such as SCC density or size).

    :param gr: ``graph-tool`` graph
    :type gr: a ``graph-tool`` object of class Graph
    """

    colors = ["#9feb3dff",
              '#a6cee3ff', '#1f78b4ff', '#b2df8aff',
              '#33a02cff', '#fb9a99ff', '#e31a1cff',
              '#fdbf6fff', '#ff7f00ff', '#cab2d6ff',
              '#6a3d9aff', '#ffff99ff', '#b15928ff']
    ecolor = gr.edge_properties["color"]
    vcolor = gr.new_vertex_property("string")

    is_true_directed = gr.new_edge_property("bool")
    for edge in gr.edges():
        is_true_directed[edge] = ecolor[edge] == 'k'
    g = gt.GraphView(gr, efilt=is_true_directed)
    comp, hist = gt.topology.label_components(g)
    for i, vertex in enumerate(gr.vertices()):
        try:
            vcolor[vertex] = colors[comp[i]]
        except IndexError:
            vcolor[vertex] = '#00000000'
    gr.vertex_properties["color"] = vcolor


def g2gt(g):
    """
    Converts a ``gunfolds`` graph to an object of class Graph of ``graph-tool`` package.
    This includes setting all parameters of edges and vertices but vertex colors.

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)

    :returns: ``graph-tool`` graph
    :rtype: a ``graph-tool`` object of class Graph
    """

    gr = every_edge_control_points(len(g))
    color = gr.edge_properties["color"]

    edge_removal = []
    for edge in gr.edges():
        v = gr.vertex_index[edge.source()]+1
        w = gr.vertex_index[edge.target()]+1
        if color[edge] == 'r' and w in g[v] and g[v][w] in {2, 3}:
            continue
        if color[edge] == 'k' and w in g[v] and g[v][w] in {1, 3}:
            continue
        edge_removal.append(edge)

    for edge in edge_removal:
        gr.remove_edge(edge)

    return gr


def gt2g(gr, dir_c='k', bidir_c='r'):
    """
    Converts a ``graph-tool`` object of class Graph to ``gunfolds`` graph

    :param gr: ``gunfolds`` graph
    :type gr: dictionary(``gunfolds`` graph)

    :returns: ``gunfolds`` graph
    :rtype: dictionary (``gunfolds`` graphs)
    """
    # check if the graph-tool graph has edge colors
    try:
        color = gr.edge_properties["color"]
    except KeyError:
        color = []

    g = {}
    for v in gr.vertices():
        g[int(str(v))+1] = {}
        if v.out_edges():
            if color:
                for x in v.out_edges():
                    if color[x] == dir_c:
                        g[int(str(v))+1][int(str(x.target()))+1] = 1
                for x in v.out_edges():
                    if color[x] == bidir_c:
                        if int(str(x.target()))+1 in g[int(str(v))+1]:
                            g[int(str(v))+1][int(str(x.target()))+1] = 3
                        else:
                            g[int(str(v))+1][int(str(x.target()))+1] = 2
                for x in v.in_edges():
                    if color[x] == bidir_c:
                        if int(str(x.source()))+1 in g[int(str(v))+1]:
                            g[int(str(v))+1][int(str(x.source()))+1] = 3
                        else:
                            g[int(str(v))+1][int(str(x.source()))+1] = 2
            else:
                g[int(str(v))+1] = {int(str(x.target()))+1: 1 for x in v.out_edges()}
    return g


def hshift(g, shift):
    """
    Horizontally shift positions of all nodes of ``graph-tool`` graph ``g`` by the value of ``shift``

    :param g: ``graph-tool`` graph
    :type g: a ``graph-tool`` object of class Graph

    :param shift: shift value
    :type shift: float
    """
    pos = g.vertex_properties["pos"]
    for x in g.vertices():
        pos[x][0] = pos[x][0] + shift


def linegraph(glist, sccs=True):
    """
    Takes a list of ``gunfolds`` graphs and merges them into a single
    Graph class ``graph-tool`` object taking care of node positions of
    each graph to ensure that the constituent graphs are arranged in a single raw when plotted.

    :param glist: a list of ``gunfold`` graphs
    :type glist: list of dictionaries (``gunfolds`` graphs)

    :param sccs: whether to distinguish SCCs by color
    :type sccs: boolean

    :returns: ``graph-tool`` graph
    :rtype: a ``graph-tool`` object of class Graph
    """

    gr = gt.Graph()

    shift = 0
    for i, g in enumerate(glist):
        g_i = g2gt(g)
        if sccs:
            colorcomponents(g_i)
        hshift(g_i, i*shift)
        points = np.asarray([x for x in g_i.vertex_properties["pos"]])
        shift = 1.5*(points[:, 0].max() - points[:, 0].min())
        gt.generation.graph_union(gr, g_i,
                                  internal_props=True,
                                  include=True)

    return gr


def plotg(g, sccs=True, output=None, fmt='auto'):
    """
    Given a ``gunfolds`` graph, plots it in an interactive window

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)

    :param sccs: whether to distinguish SCCs by color
    :type sccs: boolean

    :param output: Output file name (or object). If not given, the graph will be displayed via interactive_window().
    :type output: string or file object (optional, default: None)

    :param fmt: Output file format. Possible values are ``"auto"``, ``"ps"``, ``"pdf"``, ``"svg"``, and ``"png"``. If the value is ``"auto"``, the format is guessed from the output parameter.
    :type fmt: string or file object (optional, default: None)
    """

    gg = g2gt(g)

    vcolors = [0.62109375, 0.875, 0.23828125, 1]
    if sccs:
        colorcomponents(gg)
        vcolors = gg.vertex_properties["color"]

    gtd.graph_draw(gg, gg.vertex_properties["pos"],
                   vertex_text=gg.vertex_properties['label'],
                   edge_pen_width=gg.edge_properties['pen_width'],
                   edge_marker_size=gg.edge_properties['marker_size'],
                   edge_control_points=gg.edge_properties['control'],
                   vertex_pen_width=1,
                   edge_color=gg.edge_properties['color'],
                   vertex_fill_color=vcolors, output=output, fmt=fmt)


def plotgunfolds(g, sccs=True, output=None, fmt='auto'):
    """
    Given a ``gunfolds`` graph plots all of its undersamples versions arranged sequentially in a horizontal line.

    :param g: ``gunfolds`` graph
    :type g: dictionary (``gunfolds`` graphs)

    :param interactive: whether to make the window interactive
    :type interactive: boolean

    :param sccs: whether to distinguish SCCs by color
    :type sccs: boolean

    :param output: Output file name (or object). If not given, the graph will be displayed via interactive_window().
    :type output: string or file object (optional, default: None)

    :param fmt: Output file format. Possible values are ``"auto"``, ``"ps"``, ``"pdf"``, ``"svg"``, and ``"png"``. If the value is ``"auto"``, the format is guessed from the output parameter.
    :type fmt: string or file object (optional, default: None)
    """

    x = bfu.all_undersamples(g)
    gg = linegraph(x, sccs=sccs)

    points = np.asarray([x for x in gg.vertex_properties["pos"]])
    width = points[:, 0].max() - points[:, 0].min()
    height = points[:, 1].max() - points[:, 1].min()
    f_w = getscreensize()[0]
    f_h = height/width * f_w
    hshift(gg, -width)

    vcolors = [0.62109375, 0.875, 0.23828125, 1]
    if sccs:
        vcolors = gg.vertex_properties["color"]

    gtd.graph_draw(gg, gg.vertex_properties["pos"],
                   vertex_text=gg.vertex_properties['label'],
                   edge_pen_width=gg.edge_properties['pen_width'],
                   edge_marker_size=gg.edge_properties['marker_size'],
                   edge_control_points=gg.edge_properties['control'],
                   vertex_pen_width=1,
                   adjust_aspect=True,
                   fit_view=True,
                   fit_view_ink=True,
                   output_size=(f_w, f_h),
                   edge_color=gg.edge_properties['color'],
                   vertex_fill_color=vcolors,
                   output=output, fmt=fmt
                   )
