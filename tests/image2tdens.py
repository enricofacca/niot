"""
This script is used to corrupt and reconstruct a network image.
The corrupted image is created adding a mask to the known network.
Two parameters are used to control the reconstruction:
    gamma: controls the branching angle
    weights: controls the importance of the discrepancy term in the objective function
Usage example:
    $ python corrupt_and_reconstruct.py lines/ lines/masks.png 0.5 1.0

The results are saved in directorry name according 
to mask and parameters used in lines/runs/
"""
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
import optimal_transport as ot 
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
import image2dat as i2d

from scipy.ndimage import gaussian_filter


def image2tdens(img_network, tdens2image='pm', sigma_smoothing=1, cell2face='arithmetic_mean'):
    print('Networks: ' + img_network)
    # Read image and convert to numpy array
    np_network = i2d.image2numpy(img_network,normalize=True,invert=True)
    
    # Build mesh from numpy array
    mesh = i2d.build_mesh_from_numpy(np_network)
    network = i2d.numpy2firedrake(mesh,np_network, name='network')
    
    ctrl = Controls(
        # globals controls
        tol=1e-1,
        max_iter=1000,
        spaces='DG0DG0',
        # niot controls
        gamma=0.5,
        weight_discrepancy=1,
        #weight_penalty=weights[1],
        weight_regularization=1,
        tdens2image=tdens2image,
        sigma_smoothing=sigma_smoothing,
        # optimization controls
        time_discretization_method='tdens_mirror_descent',
        deltat=1e-3,
        nonlinear_tol=1e-5,
        linear_tol=1e-6,
        nonlinear_max_iter=30,
        linear_max_iter=1000)

    ctrl.cell2face = cell2face
    #
    # solve the problem
    #
    R = FunctionSpace(mesh, 'R', 0)
    source = Function(R,val=0,name='source')
    sink = Function(R,val=0,name='sink')
    btp = ot.BranchedTransportProblem(source, sink, gamma=ctrl.gamma)

    niot_solver = NiotSolver(btp, network, confidence=1.0, ctrl=ctrl)
    ctrl.tdens2image = 'heat'
    ctrl.sigma_smoothing = 1e-3
    niot_solver_heat = NiotSolver(btp, network, confidence=1.0, ctrl=ctrl)
    smooth = Function(niot_solver.fems.tdens_space, name='smooth')
    smooth.interpolate(niot_solver_heat.tdens2image(network))

    utilities.save2pvd([network,smooth,niot_solver.image_h],'skeleton.pvd')

if (__name__ == '__main__'):    
    parser = argparse.ArgumentParser(description='Corrupt networks with masks and reconstruct via branched transport')
    parser.add_argument('--image', type=str, help="path for the network image")
    parser.add_argument('--tdens2image', type=str, default='pm', help="tdens2image method")
    parser.add_argument('--sigma', type=float, default=1.0, help="sigma for smoothing")
    parser.add_argument('--cell2face', type=str, default='arithmetic_mean', help="cell2face method")
    args = parser.parse_args()

    img_network = args.image
    image2tdens(args.image,args.tdens2image, sigma_smoothing=args.sigma,cell2face=args.cell2face)
    

    