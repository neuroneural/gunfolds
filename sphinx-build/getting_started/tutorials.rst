
Tutorials
==========

Before we begin, make sure you have installed the gunfolds package. If not, you can follow the steps mentioned in the `Installation Guide <installation.html>`_ to use the gunfolds package.

Graph Format
------------

In the gunfolds package, graphs are represented as a dictionary of dictionaries, where each node is associated with its neighboring nodes through their respective edges. This data structure 
efficiently captures the relationships between nodes and allows for easy manipulation and analysis of graph data. Let's explore this representation in more detail:

For example:

.. code-block::

    >>> g = {1: {2: 1}, 2: {3: 2}, 3: {1: 3}}

| Here, g is a 3 node graph in which
      
      | 1: {2: 1} means node 1 is connected to node 2 with a directed edge
      | 2: {3: 2} means node 2 is connected to node 3 with a bidirected edge
      | 3: {1: 3} means node 3 is connected to node 1 with both directed and bidirected edge 


graphkit Usage
--------------
graphkit is a powerful utility tool in the gunfolds package designed for creating, analyzing, and visualizing graphs. 
It simplifies the process of handling graphs and provides essential functionalities to work with nodes, edges, and graph properties.

To import the graphkit module in your Python script or interactive environment:

.. code-block::

    from gunfolds.utils import graphkit


How to create a ring graph with some added edges?
""""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block::

    >>> g = graphkit.ringmore(n = 4, m = 3)
    >>> g
    {1: {2: 1, 1: 1}, 2: {3: 1}, 3: {4: 1, 1: 1}, 4: {1: 1, 2: 1}}


In the gunfolds package, the ``graphkit.ringmore()`` function is used to generate a graph 
that extends the concept of a ring graph by adding additional edges between nodes, resulting 
in a more connected structure.

Here,  
      | n is the number of nodes
      | m is the number of additional edges to be added

How to create a random graph?
"""""""""""""""""""""""""""""""""""

.. code-block::

    >>> g = graphkit.bp_mean_degree_graph(node_num = 4, degree = 3)
    >>> g
    {1: {1: 1, 2: 1, 3: 1, 4: 1}, 2: {3: 1, 4: 1}, 3: {1: 1, 2: 1, 3: 1}, 4: {3: 1, 4: 1}}

The function ``graphkit.bp_mean_degree_graph()`` generates a random graph with a specified number 
of nodes and a target mean degree.

How to create a DAG connecting multiple scc rings?
""""""""""""""""""""""""""""""""""""""""""""""""""""

.. code-block::

    >>> g = graphkit.ring_sccs(num=2, num_sccs=3, dens=0.5, degree=1, max_cross_connections=2)
    >>> g
    {1: {2: 1}, 2: {1: 1, 2: 1}, 3: {4: 1, 2: 1}, 4: {3: 1, 4: 1}, 5: {6: 1, 5: 1, 4: 1, 3: 1}, 6: {5: 1}}

gtool Usage
------------
gtool is a powerful visualization tool 

To import the gtool module in your Python script or interactive environment:

.. code-block::

    from gunfolds.utils import gtool

How to use gtool to plot graphs?
""""""""""""""""""""""""""""""""

bfutils Usage
---------------

To import the bfutils module in your Python script or interactive environment:

.. code-block::

    from gunfolds.utils import bfutils

How to to undersample a graph by one?
""""""""""""""""""""""""""""""""""""""
.. code-block::

    >>> g1 = {1: {2: 1}, 2: {3: 1, 2: 1}, 3: {1: 1}, 4: {5: 1, 3: 1, 2: 1}, 5: {6: 1}, 6: {4: 1, 5: 1}}
    >>> g2 = bfutils.increment(g1)
    >>> g2
    {1: {3: 1, 2: 1}, 2: {1: 1, 3: 3, 2: 1, 5: 2}, 3: {2: 3, 5: 2}, 4: {6: 1, 1: 1, 3: 1, 2: 1, 5: 2}, 5: {4: 3, 5: 1, 3: 2, 2: 2}, 6: {5: 1, 3: 1, 2: 1, 6: 1}}

How to get all undersamplings of a given graph g?
""""""""""""""""""""""""""""""""""""""""""""""""""
.. code-block::

    >>> g = {1: {2: 1}, 2: {3: 1, 2: 1}, 3: {1: 1}, 4: {1: 1, 3: 1, 2: 1}}
    >>> list_of_all_underssamplings = bfutils.all_undersamples(g)
    >>> list_of_all_underssamplings
    [{1: {2: 1}, 2: {3: 1, 2: 1}, 3: {1: 1}, 4: {1: 1, 3: 1, 2: 1}}, {1: {3: 3, 2: 3}, 2: {1: 3, 3: 3, 2: 1}, 3: {2: 3, 1: 2}, 4: {2: 1, 1: 1, 3: 1}}, {1: {1: 1, 3: 3, 2: 3}, 2: {2: 1, 1: 3, 3: 3}, 3: {3: 1, 2: 3, 1: 2}, 4: {3: 1, 2: 1, 1: 1}}, {1: {2: 3, 1: 1, 3: 3}, 2: {3: 3, 2: 1, 1: 3}, 3: {1: 3, 3: 1, 2: 3}, 4: {1: 1, 3: 1, 2: 1}}]

How to run a rasl algorithm?
""""""""""""""""""""""""""""
