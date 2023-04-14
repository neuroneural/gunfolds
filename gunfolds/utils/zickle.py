"""Generic object pickler and compressor

This module saves and reloads compressed representations of generic Python
objects to and from the disk.
"""

__author__ = "Bill McNeill <billmcn@speakeasy.net>"
__version__ = "1.0"

import pickle as cPickle
import gzip


def save(object, filename, protocol=-1):
    """
       Save an object to a compressed disk file.
       Works well with huge objects.
       
       :param object: object to be saved as a zickle file
       
       :param filename: name of the file
       :type filename: string
    """
    with gzip.GzipFile(filename, 'wb') as file:
        cPickle.dump(object, file, protocol)


def load(filename):
    """
       Loads a compressed object from disk

       :param filename: name of the file
       :type filename: string
    """
    with gzip.GzipFile(filename, 'rb') as file:
        object = cPickle.load(file)
    return object


if __name__ == "__main__":
    import sys
    import os.path

    class Object:
        x = 7
        y = "This is an object."

    filename = sys.argv[1]
    if os.path.isfile(filename):
        o = load(filename)
        print("Loaded %s" % o)
    else:
        o = Object()
        save(o, filename)
        print("Saved %s" % o)
