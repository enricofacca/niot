import argparse
import sys
import glob
import os
from copy import deepcopy as cp
import numpy as np
import cProfile
sys.path.append('../src/niot')
from niot import NiotSolver
from niot import Controls
import image2dat as i2d


from ufl import *
from firedrake import *
from firedrake import norm
from firedrake import Function
from firedrake import interpolate
# for writing to file
from firedrake import File
import firedrake as fire
from memory_profiler import profile

import utilities as utilities

from scipy.ndimage import gaussian_filter
from corrupt_and_reconstruct import corrupt_and_reconstruct as c_and_r

if __name__ == '__main__':
    c_and_r('frog_tongue/source.png',
            'frog_tongue/sink.png',
            'frog_tongue/network.png',
            'frog_tongue/mask01.png',
            0.5,
            'DG0DG0',
            0.5,
            [1,1,0],
            0,
            'MASK',
            'laplacian_smoothing',
            'frog_tongue/mask01/','frog_tongue/mask01/')
