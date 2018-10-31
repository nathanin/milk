from __future__ import print_function
import numpy as np
import cv2
import os
import glob
import re
import time

class MILDataset(object):
    """ Eager-enabled dataset with py-func for multiple-instance bags

    bags / sets are stored as .npy or .npz files
    to pop a set we should return x=(n, h, w, c), y=(n_classes)

    internally we store a mapping from file name to label
    the generator selects a file name, and selectively loads a subset of the contents

    the contents are pre-processed with python functions

    the whole thing is python functions except for the iterator, and async loading ..
    async is done through tensorflow dataset.map(py_func, num_parallel_calls=n_threads)
    
    no batching. (or, optionally use batching)
    shuffle should work on the list of bags
    prefetch should work after async loading 

    see: https://github.com/nathanin/svs_reader/tests/test_multithreading.py


    ** Need to test how the disk impacts async loading from multiple files 
    ** And if it's OK to use np.load with mmap_mode='r'

    """

    def __init__(self):

