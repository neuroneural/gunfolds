from copy import deepcopy
from gunfolds.utils import bfutils
from gunfolds import conversions
from gunfolds.utils import graphkit
from gunfolds.solvers import traversal
from gunfolds.solvers import unknownrate as ur
from gunfolds.utils import zickle as zkl
import numpy as np
import os
import unittest


class TestBFUtilsFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._G = {1: {1: 1, 2: 1, 3: 1},
                 2: {2: 1, 3: 1, 4: 1},
                 3: {1: 1, 2: 1, 3: 1, 4: 1},
                 4: {1: 1, 2: 1, 3: 1, 5: 1},
                 5: {1: 1}}

    def setUp(self):
        # Copy these before each test incase a test modifies them
        self.G = deepcopy(self._G)

    def test__call_undersamples(self):
        gs = [{1: {1: 1, 2: 1, 3: 1},
              2: {2: 1, 3: 1, 4: 1},
              3: {1: 1, 2: 1, 3: 1, 4: 1},
              4: {1: 1, 2: 1, 3: 1, 5: 1},
              5: {1: 1}},
             {1: {1: 1, 2: 3, 3: 3, 4: 3, 5: 2},
              2: {1: 3, 2: 1, 3: 3, 4: 3, 5: 3},
              3: {1: 3, 2: 3, 3: 1, 4: 3, 5: 3},
              4: {1: 3, 2: 3, 3: 3, 4: 1},
              5: {1: 3, 2: 3, 3: 3}},
             {1: {1: 1, 2: 3, 3: 3, 4: 3, 5: 3},
              2: {1: 3, 2: 1, 3: 3, 4: 3, 5: 3},
              3: {1: 3, 2: 3, 3: 1, 4: 3, 5: 3},
              4: {1: 3, 2: 3, 3: 3, 4: 1, 5: 3},
              5: {1: 3, 2: 3, 3: 3, 4: 3}},
             {1: {1: 1, 2: 3, 3: 3, 4: 3, 5: 3},
              2: {1: 3, 2: 1, 3: 3, 4: 3, 5: 3},
              3: {1: 3, 2: 3, 3: 1, 4: 3, 5: 3},
              4: {1: 3, 2: 3, 3: 3, 4: 1, 5: 3},
              5: {1: 3, 2: 3, 3: 3, 4: 3, 5: 1}}]

        gs_test = bfutils.call_undersamples(self.G)
        self.assertEqual(gs, gs_test)

    def test__call_undersample(self):
        u = 1
        g_u_1 = {1: {1: 1, 2: 3, 3: 3, 4: 3, 5: 2},
                 2: {1: 3, 2: 1, 3: 3, 4: 3, 5: 3},
                 3: {1: 3, 2: 3, 3: 1, 4: 3, 5: 3},
                 4: {1: 3, 2: 3, 3: 3, 4: 1},
                 5: {1: 3, 2: 3, 3: 3}}
        g2 = bfutils.undersample(self.G, u)
        self.assertEqual(g_u_1, g2)

        u = 2
        g_u_2 = {1: {1: 1, 2: 3, 3: 3, 4: 3, 5: 3},
                 2: {1: 3, 2: 1, 3: 3, 4: 3, 5: 3},
                 3: {1: 3, 2: 3, 3: 1, 4: 3, 5: 3},
                 4: {1: 3, 2: 3, 3: 3, 4: 1, 5: 3},
                 5: {1: 3, 2: 3, 3: 3, 4: 3}}
        g2 = bfutils.undersample(self.G, u)
        self.assertEqual(g_u_2, g2)

        u = 4
        g_u_4 = {1: {1: 1, 2: 3, 3: 3, 4: 3, 5: 3},
                 2: {1: 3, 2: 1, 3: 3, 4: 3, 5: 3},
                 3: {1: 3, 2: 3, 3: 1, 4: 3, 5: 3},
                 4: {1: 3, 2: 3, 3: 3, 4: 1, 5: 3},
                 5: {1: 3, 2: 3, 3: 3, 4: 3, 5: 1}}
        g2 = bfutils.undersample(self.G, u)
        self.assertEqual(g_u_4, g2)

    def test__is_sclique(self):
        sc_1 = {1: {1: 1, 2: 3, 3: 3, 4: 3},
                2: {1: 3, 2: 1, 3: 3, 4: 3},
                3: {1: 3, 2: 3, 3: 1, 4: 3},
                4: {1: 3, 2: 3, 3: 3, 4: 1}}

        self.assertTrue(bfutils.is_sclique(sc_1))

        no_sc_1 = {1: {1: 1, 2: 3, 4: 3},
                   2: {1: 3, 2: 1, 3: 3, 4: 3},
                   3: {1: 3, 2: 3, 4: 3},
                   4: {1: 3, 2: 3, 3: 3, 4: 1}}

        self.assertFalse(bfutils.is_sclique(no_sc_1))



class TestConversionFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._G = {'1': {'1': set([(0, 1)]),
                   '2': set([(0, 1)]),
                   '3': set([(0, 1)])},
             '3': {'1': set([(0, 1)]),
                   '3': set([(0, 1)]),
                   '2': set([(0, 1)]),
                   '4': set([(0, 1)])},
             '2': {'3': set([(0, 1)]),
                   '2': set([(0, 1)]),
                   '4': set([(0, 1)])},
             '5': {'1': set([(0, 1)])},
             '4': {'1': set([(0, 1)]),
                   '3': set([(0, 1)]),
                   '2': set([(0, 1)]),
                   '5': set([(0, 1)])}}

    def setUp(self):
        # Copy these before each test incase a test modifies them
        self.G = deepcopy(self._G)


    def test__dict_format_converter(self):
        expected = {1: {1: 1, 2: 1, 3: 1},
                    2: {2: 1, 3: 1, 4: 1},
                    3: {1: 1, 2: 1, 3: 1, 4: 1},
                    4: {1: 1, 2: 1, 3: 1, 5: 1},
                    5: {1: 1}}
        converted = conversions.dict_format_converter(self.G)
        self.assertEqual(expected, converted)

    def test__to_adj_matrix(self):
        g = {1: {1: 1, 2: 2, 3: 1},
             2: {1: 2, 2: 1, 3: 1, 4: 1},
             3: {1: 1, 2: 1, 3: 1, 4: 1},
             4: {1: 1, 2: 1, 3: 1, 5: 1},
             5: {1: 3}}

        expected_a = np.array([[1, 0, 1, 0, 0],
                               [0, 1, 1, 1, 0],
                               [1, 1, 1, 1, 0],
                               [1, 1, 1, 0, 1],
                               [1, 0, 0, 0, 0]], dtype=np.int8)

        expected_b = np.array([[0, 1, 0, 0, 0],
                               [1, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0],
                               [1, 0, 0, 0, 0]], dtype=np.int8)

        self.assertTrue((conversions.graph2adj(g) == expected_a).all())
        self.assertTrue((conversions.graph2badj(g) == expected_b).all())

        # test round trip
        A = conversions.graph2adj(g)
        B = conversions.graph2badj(g)
        self.assertEqual(conversions.adjs2graph(A, B), g)



class TestGraphkitFunctions(unittest.TestCase):

    def test__superclique(self):
        expected = {1: {1: 1, 2: 3, 3: 3, 4: 3},
                    2: {1: 3, 2: 1, 3: 3, 4: 3},
                    3: {1: 3, 2: 3, 3: 1, 4: 3},
                    4: {1: 3, 2: 3, 3: 3, 4: 1}}
        sc = graphkit.superclique(4)
        self.assertEqual(expected, sc)



class TestUnknownRateFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Read in pickle file with results
        cls._G = {1: {1: 1, 2: 1, 3: 1},
                  2: {2: 1, 3: 1, 4: 1},
                  3: {1: 1, 2: 1, 3: 1, 4: 1},
                  4: {1: 1, 2: 1, 3: 1, 5: 1},
                  5: {1: 1}}
        DIR_NAME = os.path.dirname(__file__)
        cls._ABS_PATH = os.path.abspath(os.path.join(DIR_NAME))

    def setUp(self):
        # Copy these before each test incase a test modifies them
        self.G = deepcopy(self._G)

    def test__liteqclass(self):
        u = 1
        set_of_u_1 = zkl.load("{}/liteqclass_output_n_5_u_1.zkl".format(self._ABS_PATH))
        g2 = bfutils.undersample(self.G, u)
        s = ur.liteqclass(g2, verbose=False, capsize=1000)
        self.assertEqual(s, set_of_u_1)

        u = 2
        set_of_u_2 = zkl.load("{}/liteqclass_output_n_5_u_2.zkl".format(self._ABS_PATH))
        g2 = bfutils.undersample(self.G, u)
        s = ur.liteqclass(g2, verbose=False, capsize=1000)
        self.assertEqual(s, set_of_u_2)

    def test__gen_loops(self):
        n = 3
        expected = [[1], [2], [3], [1, 2], [1, 3],
                    [2, 3], [1, 2, 3], [1, 3, 2]]
        self.assertEqual(expected, ur.gen_loops(n))

        n = 5
        expected = [[1], [2], [3], [4], [5], [1, 2], [1, 3], [1, 4], [1, 5], 
                    [2, 3], [2, 4], [2, 5], [3, 4], [3, 5], [4, 5], [1, 2, 3],
                    [1, 3, 2], [1, 2, 4], [1, 4, 2], [1, 2, 5], [1, 5, 2], [1, 3, 4], 
                    [1, 4, 3], [1, 3, 5], [1, 5, 3], [1, 4, 5], [1, 5, 4], [2, 3, 4], 
                    [2, 4, 3], [2, 3, 5], [2, 5, 3], [2, 4, 5], [2, 5, 4], [3, 4, 5], 
                    [3, 5, 4], [1, 2, 3, 4], [1, 2, 4, 3], [1, 3, 2, 4], [1, 3, 4, 2], 
                    [1, 4, 2, 3], [1, 4, 3, 2], [1, 2, 3, 5], [1, 2, 5, 3], 
                    [1, 3, 2, 5], [1, 3, 5, 2], [1, 5, 2, 3], [1, 5, 3, 2], 
                    [1, 2, 4, 5], [1, 2, 5, 4], [1, 4, 2, 5], [1, 4, 5, 2], 
                    [1, 5, 2, 4], [1, 5, 4, 2], [1, 3, 4, 5], [1, 3, 5, 4], 
                    [1, 4, 3, 5], [1, 4, 5, 3], [1, 5, 3, 4], [1, 5, 4, 3], 
                    [2, 3, 4, 5], [2, 3, 5, 4], [2, 4, 3, 5], [2, 4, 5, 3], 
                    [2, 5, 3, 4], [2, 5, 4, 3], [1, 2, 3, 4, 5], [1, 2, 3, 5, 4], 
                    [1, 2, 4, 3, 5], [1, 2, 4, 5, 3], [1, 2, 5, 3, 4], 
                    [1, 2, 5, 4, 3], [1, 3, 2, 4, 5], [1, 3, 2, 5, 4], 
                    [1, 3, 4, 2, 5], [1, 3, 4, 5, 2], [1, 3, 5, 2, 4], 
                    [1, 3, 5, 4, 2], [1, 4, 2, 3, 5], [1, 4, 2, 5, 3], 
                    [1, 4, 3, 2, 5], [1, 4, 3, 5, 2], [1, 4, 5, 2, 3], 
                    [1, 4, 5, 3, 2], [1, 5, 2, 3, 4], [1, 5, 2, 4, 3], 
                    [1, 5, 3, 2, 4], [1, 5, 3, 4, 2], [1, 5, 4, 2, 3], [1, 5, 4, 3, 2]]
        self.assertEqual(expected, ur.gen_loops(n))

    def test__loop2graph(self):
        n = 3
        e = [2, 3]
        expected = {1: {}, 2: {3: 1}, 3: {2: 1}}
        self.assertEqual(expected, ur.loop2graph(e, n))

        n = 3
        e = [1, 3, 2]
        expected = {1: {3: 1}, 2: {1: 1}, 3: {2: 1}}
        self.assertEqual(expected, ur.loop2graph(e, n))

        n = 4
        e = [1, 4, 3]
        expected = {1: {4: 1}, 2: {}, 3: {1: 1}, 4: {3: 1}}
        self.assertEqual(expected, ur.loop2graph(e, n))

        n = 5
        e = [1, 5, 4, 2]
        expected = {1: {5: 1}, 2: {1: 1}, 3: {}, 4: {2: 1}, 5: {4: 1}}
        self.assertEqual(expected, ur.loop2graph(e, n))


class TestTraversalFunctions(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Read in pickle file with results
        cls._G1 = {1: {1: 1, 2: 1, 3: 1},
                   2: {1: 1, 3: 1, 4: 1},
                   3: {1: 1, 2: 1, 3: 1, 4: 1},
                   4: {1: 1, 2: 1, 3: 1, 5: 1},
                   5: {2: 1}}
        cls._G2 = {1: {1: 1, 2: 3, 3: 3, 4: 3, 5: 2},
                   2: {1: 3, 2: 1, 3: 3, 4: 3, 5: 3},
                   3: {1: 3, 2: 3, 3: 1, 4: 3, 5: 3},
                   4: {1: 3, 2: 3, 3: 3, 4: 1},
                   5: {1: 3, 2: 2, 3: 3, 4: 1}}
        DIR_NAME = os.path.dirname(__file__)
        cls._ABS_PATH = os.path.abspath(os.path.join(DIR_NAME))

    def setUp(self):
        # Copy these before each test incase a test modifies them
        self.G1 = deepcopy(self._G1)
        self.G2 = deepcopy(self._G2)

    def test__v2g22g1(self):
        expected = zkl.load("{}/v2g22g1_output.zkl".format(self._ABS_PATH))
        self.assertEqual(expected, traversal.v2g22g1(self.G2))

    def test__supergraphs_in_eq(self):
        expected = set([])
        self.assertEqual(expected, traversal.supergraphs_in_eq(self.G1, self.G2))

        g1 = {1: {1: 1, 2: 1, 3: 1},
              2: {3: 1, 4: 1},
              3: {1: 1, 2: 1, 4: 1},
              4: {1: 1, 2: 1, 3: 1, 5: 1},
              5: {2: 1}}
        g2 = {1: {1: 1, 2: 3, 3: 3, 4: 3, 5: 2},
              2: {1: 3, 2: 1, 3: 3, 4: 3, 5: 3},
              3: {1: 3, 2: 3, 3: 1, 4: 3, 5: 3},
              4: {1: 3, 2: 3, 3: 3, 4: 1},
              5: {1: 2, 2: 2, 3: 3, 4: 1}}
        expected = {29588392}
        self.assertEqual(expected, traversal.supergraphs_in_eq(g1, g2))


    def test__checkvedge(self):
        expected = sorted([(2, 2), (4, 1), (1, 2), 
                    (3, 4), (4, 4), (2, 3), 
                    (2, 5), (3, 3), (5, 3), 
                    (4, 2), (1, 1), (1, 5), 
                    (1, 4), (5, 2), (4, 3), 
                    (2, 1), (1, 3), (3, 1), 
                    (3, 2), (2, 4), (3, 5), 
                    (5, 1)])
        self.assertEqual(expected, sorted(traversal.checkvedge((5, 1, 3), self.G2)))

        expected = sorted([(2, 2), (5, 5), (4, 1), 
                    (1, 2), (3, 4), (4, 4), 
                    (2, 3), (2, 5), (1, 5), 
                    (5, 3), (4, 2), (1, 1), 
                    (3, 3), (1, 4), (5, 2), 
                    (4, 3), (2, 1), (1, 3), 
                    (3, 1), (3, 2), (2, 4), 
                    (3, 5), (5, 1)])
        self.assertEqual(expected, sorted(traversal.checkvedge((4, 1, 3), self.G2)))

    def test__checkAedge(self):
        expected = sorted([(1, 3), (3, 1), (1, 2), 
                    (2, 1), (1, 5), (5, 1), 
                    (1, 4), (4, 1), (3, 2), 
                    (2, 3), (3, 5), (5, 3), 
                    (3, 4), (4, 3), (2, 5), 
                    (5, 2), (2, 4), (4, 2), 
                    (5, 4), (4, 5), (1, 1), 
                    (3, 3), (2, 2), (5, 5), 
                    (4, 4)])
        self.assertEqual(expected, sorted(traversal.checkAedge((1, 2, 3, 2), self.G2)))

    def test__checkedge(self):
        expected = [1, 2, 3, 4, 5]
        self.assertEqual(expected, sorted(traversal.checkedge((5, 4), self.G2)))

    def test__ok2addanedge(self):
        self.assertFalse(traversal.ok2addanedge(1, 5, self.G1, self.G2, rate=1))

    def test__ok2addanedge1(self):
        self.assertFalse(traversal.ok2addanedge1(1, 5, self.G1, self.G2, rate=1))

    def test__ok2addavedge(self):
        g = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
        self.assertTrue(traversal.ok2addavedge((1, 2, 4), (2, 2), g, self.G2))
        self.assertFalse(traversal.ok2addavedge((1, 2, 4), (5, 1), g, self.G2))

    def test__ok2add2edges(self):
        g = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
        self.assertTrue(traversal.ok2add2edges((5, 4), 1, g, self.G2))

    def test__ok2addacedge(self):
        g = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
        self.assertTrue(traversal.ok2addacedge((0, 3, 2, 2), (5, 3), g, self.G2))
        self.assertFalse(traversal.ok2addacedge((0, 3, 2, 2), (3, 5), g, self.G2))

    def test__edge_increment_ok(self):
        g = {1: {}, 2: {}, 3: {}, 4: {}, 5: {}}
        self.assertTrue(traversal.edge_increment_ok(5, 1, 4, g, self.G2))
        self.assertFalse(traversal.edge_increment_ok(5, 1, 4, self.G1, self.G2))

    def test__signature(self):
        edges = [(3, 5, 4), (0, 3, 2, 2), (2, 5, 4),
                 (4, 2, 4), (1, 2, 4), (2, 1, 3),
                 (1, 1, 3), (5, 4), (5, 1, 3), (3, 1, 3),
                 (4, 1, 3)]
        expected = (30112680, 1545134244133543132542131241130322L)
        self.assertEqual(expected, traversal.signature(self.G1, edges))

    def test__inorder_check2(self):
        in1 = (5, 4)
        in2 = (2, 5, 4)
        in3 = set([1, 3, 2, 5, 4])
        in4 = set([(2, 3), (2, 1), (2, 5), 
                   (1, 3), (3, 1), (1, 4), 
                   (1, 2), (1, 5), (3, 2), 
                   (5, 3), (3, 4), (5, 1), 
                   (4, 3), (3, 5), (4, 2)])
        expected_d = {1: set([(2, 3), (2, 1), (3, 1), (3, 2), 
                                (5, 3), (3, 4), (2, 5), (3, 5), 
                                (5, 1)]),
                      3: set([(2, 3), (2, 1), (4, 3), (1, 4), 
                                (1, 5), (5, 3), (1, 2), (4, 2), 
                                (2, 5), (1, 3), (5, 1)]), 
                      2: set([(4, 3), (3, 1), (3, 4), (1, 4), 
                                (1, 3)]), 
                      5: set([(2, 1), (3, 1), (2, 3), (3, 2), 
                                (3, 4)]), 
                      4: set([(2, 3), (2, 1), (3, 5), (3, 1), 
                                (1, 4), (3, 4), (1, 5), (3, 2), 
                                (1, 2), (2, 5), (1, 3)])}
        expected_s_i1 = set([1, 3, 2, 5, 4])
        expected_s_i2 = set([(2, 3), (2, 1), (4, 3), 
                             (1, 4), (1, 3), (3, 1), 
                             (3, 2), (1, 2), (1, 5), 
                             (5, 3), (3, 4), (4, 2), 
                             (2, 5), (3, 5), (5, 1)])

        d, s_i1, s_i2 = traversal.inorder_check2(in1, in2, in3, in4,
                                   self.G2, f=[], c=[])
        self.assertEqual(expected_d, d)
        self.assertEqual(expected_s_i1, s_i1)
        self.assertEqual(expected_s_i2, s_i2)

    def test__check3(self):
        in1 = (1, 1, 3)
        in2 = (3, 1, 3)
        in3 = (2, 1, 3)
        in4 =  set([(2, 3), (1, 1), (4, 4), (2, 1), (4, 3), 
                    (3, 3), (3, 5), (1, 3), (3, 1), (1, 4), 
                    (2, 4), (4, 1), (1, 5), (3, 2), (1, 2), 
                    (4, 2), (2, 5), (3, 4), (2, 2)])
        in5 =  set([(2, 3), (1, 1), (5, 2), (4, 4), (5, 1), 
                    (2, 1), (1, 3), (3, 1), (1, 4), (2, 4), 
                    (4, 1), (3, 3), (3, 2), (5, 3), (1, 2), 
                    (4, 2), (4, 3), (3, 4), (2, 2)])
        in6 =  set([(2, 2), (5, 5), (4, 1), (1, 2), (3, 4), 
                    (4, 4), (2, 3), (2, 5), (3, 3), (5, 3), 
                    (2, 4), (1, 1), (1, 5), (1, 4), (5, 2), 
                    (4, 3), (2, 1), (1, 3), (3, 1), (3, 2), 
                    (4, 2), (3, 5), (5, 1)])

        expected_s1 = set([(2, 3), (1, 1), (4, 4), (2, 1), (2, 5), 
                           (1, 5), (1, 3), (2, 2), (3, 1), (1, 4), 
                           (3, 4), (4, 1), (3, 3), (3, 2), (1, 2), 
                           (4, 2), (4, 3), (3, 5), (2, 4)])
        expected_s2 = set([(2, 3), (1, 1), (4, 4), (2, 1), (2, 2), 
                           (3, 1), (1, 4), (3, 4), (4, 1), (3, 3), 
                           (3, 2), (1, 2), (4, 2), (4, 3), (1, 3), 
                           (2, 4)])
        expected_s3 = set([(4, 4), (5, 5), (4, 1), (1, 2), (3, 4), 
                           (2, 2), (2, 3), (4, 3), (1, 5), (5, 3), 
                           (4, 2), (1, 1), (3, 3), (1, 4), (5, 2), 
                           (2, 5), (2, 1), (3, 5), (3, 1), (3, 2), 
                           (2, 4), (1, 3), (5, 1)])

        s1, s2, s3 = traversal.check3(in1, in2, in3, in4, 
                                in5, in6, self.G2, f=[], c=[])
        
        self.assertEqual(expected_s1, s1)
        self.assertEqual(expected_s2, s2)
        self.assertEqual(expected_s3, s3)

    def test__prune_sort_CDS(self):
        pool, CDS = zkl.load("{}/prune_sort_CDS_input.zkl".format(self._ABS_PATH))
        ex_cds, ex_order, ex_idx = zkl.load("{}/prune_sort_CDS_output.zkl".format(self._ABS_PATH))
        cds, order, idx = traversal.prune_sort_CDS(CDS, pool)

        ex_idx.sort()
        idx.sort()
        self.assertTrue((ex_idx == idx).all())
        self.assertEqual(sorted(ex_cds), sorted(cds))
        self.assertEqual(sorted(ex_order), sorted(order))

    def test__length_d_paths(self):
        expected = sorted([[1, 3, 2], [1, 3, 4], [1, 2, 3], [1, 2, 4]])
        output = sorted([ x for x in traversal.length_d_paths(self.G1, 1, 2)])
        self.assertEqual(expected, output)

        expected = sorted([[3, 1, 2],
                            [3, 2, 1],
                            [3, 2, 4],
                            [3, 4, 1],
                            [3, 4, 2],
                            [3, 4, 5]])
        output = sorted([ x for x in traversal.length_d_paths(self.G1, 3, 2)])
        self.assertEqual(expected, output)

    def test__length_d_loopy_paths(self):
        expected = [[4]]
        output = [ x for x in traversal.length_d_loopy_paths(self.G2, 4, 1, [2, 4])]
        self.assertEqual(expected, output)

        expected = sorted([[1, 1], [1, 3], [1, 2], [1, 4]])
        output = sorted([ x for x in traversal.length_d_loopy_paths(self.G2, 1, 2, [2, 3, 2])])
        self.assertEqual(expected, output)

    def test__checkApath(self):
        expected = sorted([(1, 1), (1, 3),
                            (1, 2), (1, 4), (3, 1),
                            (3, 3), (3, 2), (3, 5),
                            (3, 4), (2, 1), (2, 3),
                            (2, 2), (2, 4),
                            (4, 1), (4, 3), (4, 2),
                            (4, 4)])
        output = sorted(traversal.checkApath((1, 2, 3, 2), self.G2))
        self.assertEqual(expected, output)


if __name__ == '__main__':
    try:
        from teamcity import is_running_under_teamcity
        from teamcity.unittestpy import TeamcityTestRunner

        if is_running_under_teamcity():
            runner = TeamcityTestRunner()
        else:
            runner = unittest.TextTestRunner()
    except ImportError:
        runner = unittest.TextTestRunner()

    unittest.main(testRunner=runner, verbosity=2)
