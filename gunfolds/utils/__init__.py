import os
from gunfolds.utils import zickle as zkl

DIR_NAME = os.path.dirname(__file__)
ABS_PATH = os.path.abspath(os.path.join(DIR_NAME))

class LazyLoadData(object):
    """ Prevent import from waiting on pickle file loads """
    @property
    def alloops(self):
        if not hasattr(self, '_alloops'):
            self._alloops = zkl.load('{}/../data/allloops.zkl'.format(ABS_PATH))
        return self._alloops

    @property
    def circp(self):
        if not hasattr(self, '_circp'):
            self._circp = zkl.load('{}/../data/circular_p.zkl'.format(ABS_PATH))
        return self._circp

    @property
    def colors(self):
        if not hasattr(self, '_colors'):
            self._colors = zkl.load('{}/../data/colors.zkl'.format(ABS_PATH))
        return self._colors


load_data = LazyLoadData()
