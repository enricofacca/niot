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
#from memory_profiler import profile

import utilities as utilities

from scipy.ndimage import gaussian_filter
from corrupt_and_reconstruct import corrupt_and_reconstruct as c_and_r

if __name__ == '__main__':
    #dir = 'asymmetric/nref0'
    dir = 'frog_tongue/'
    c_and_r(img_source=f'{dir}/source.png',
            img_sink=f'{dir}/sink.png',
            img_network=f'{dir}/network.png',
            img_mask=f'{dir}/mask02.png',
            scaling_size=1,
            fem='DG0DG0',
            gamma=0.5,
            weights=[1,1,1e-4],
            corrupted_as_initial_guess=1,
            confidence='ONE',
            tdens2image='identity',
            directory=f'{dir}/mask02/',
            sigma_smoothing=1e0)
