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


def image2tdens(img_network, tdens2image='pm', sigma_smoothing=1):
    print('Networks: ' + img_network)
    # Read image and convert to numpy array
    np_network = i2d.image2numpy(img_network,normalize=True,invert=True)
    
    # Build mesh from numpy array
    mesh = i2d.build_mesh_from_numpy(np_network,mesh_type='cartesian')
    network = i2d.numpy2firedrake(mesh,np_network, name='network')
    

    #
    # solve the problem
    #
    R = FunctionSpace(mesh, 'R', 0)
    source = Function(R,val=0,name='source')
    sink = Function(R,val=0,name='sink')
    btp = ot.BranchedTransportProblem(source, sink, gamma=0.5)


    niot_solver = NiotSolver(btp, network, 
                             spaces = 'DG0DG0',
                             cell2face = 'harmonic_mean',
                             setup=False)
    niot_solver.set_solution(tdens=network+1e-6)
    
    
    # inpainting
    niot_solver.ctrl_set("discrepancy_weight", 0.1)
    niot_solver.ctrl_set("regularization_weight", 0)
    niot_solver.ctrl_set(['tdens2image','type'], tdens2image)
    niot_solver.ctrl_set(['tdens2image','scaling'], 1.0)
    if tdens2image == 'heat':
        niot_solver.ctrl_set(['tdens2image','heat','sigma'], sigma_smoothing)
    if tdens2image == 'pm':
        niot_solver.ctrl_set(['tdens2image','pm','sigma'], sigma_smoothing)
        niot_solver.ctrl_set(['tdens2image','pm','exp_m'], 2) # exponent to get r^4~M
    

    # optimization
    niot_solver.ctrl_set('optimization_tol', 1e-2)
    niot_solver.ctrl_set('constrain_tol', 1e-8)
    niot_solver.ctrl_set('max_iter', 1)
    niot_solver.ctrl_set('max_restart', 3)
    niot_solver.ctrl_set('verbose', 2)  # usign niot_solver method
    
    
    # time discretization
    method = 'tdens_mirror_descent_explicit'
    method = 'gfvar_gradient_descent_explicit'
    method = 'gfvar_gradient_descent_semi_implicit'
    niot_solver.ctrl_set(['dmk','type'], method)
    niot_solver.ctrl_set(['dmk','tdens_mirror_descent_explicit','gradient_scaling'], 'dmk')
    niot_solver.ctrl_set(['dmk','tdens_mirror_descent_semi_implicit','gradient_scaling'], 'dmk')

    # time step
    deltat_control = {
        'type': 'adaptive2',
        'lower_bound': 1e-7,
        'upper_bound': 1e-2,
        'expansion': 1.02,
    }
    niot_solver.ctrl_set(['dmk',method,'deltat'], deltat_control)
    

    ierr = niot_solver.solve()
    
    if tdens2image == 'identity':
        test = TestFunction(niot_solver.fems.tdens_space)
        pot, tdens = niot_solver.sol.subfunctions
        wd  = niot_solver.ctrl_get("discrepancy_weight")
        scaling = niot_solver.ctrl_get(["tdens2image","scaling"])
        exact_gradient = wd * niot_solver.confidence * (niot_solver.img_observed - tdens) * test * scaling * dx        
        g_exact = assemble(exact_gradient)

        with niot_solver.rhs_ode.dat.vec as g, g_exact.dat.vec as g_reference:
            niot_solver.print_info(
            utilities.msg_bounds(g,'gradient'),
            color='blue')
            niot_solver.print_info(
            utilities.msg_bounds(g_reference,'g_reference'),
            color='green')
            print(np.linalg.norm(g.array-g_reference.array))
        


    smooth = Function(niot_solver.fems.tdens_space, name='smooth')
    x,y = SpatialCoordinate(mesh)
    net2 = Function(niot_solver.fems.tdens_space)
    net2.rename('network2')
    #net2.interpolate(network+1e-6+3*network*conditional(x>0.5,1,0)*conditional(y>0.5,1,0))
    #img = niot_solver.tdens2image(net2)
    #smooth.interpolate(img)

    # value is small than 1 to have a stronger effect given by r**(1/p)
    value = 0.1
    net2.interpolate( conditional(x>0.48,value,0)
                     * conditional(x<0.52,value,0)
                     * conditional(y>0.5,1,2))
    img = niot_solver.tdens2image(net2)
    smooth.interpolate(img)



    utilities.save2pvd([network,net2,smooth],'skeleton.pvd')

if (__name__ == '__main__'):    
    parser = argparse.ArgumentParser(description='Corrupt networks with masks and reconstruct via branched transport')
    parser.add_argument('--image', type=str, help="path for the network image")
    parser.add_argument('--tdens2image', type=str, default='pm', help="tdens2image method")
    parser.add_argument('--sigma', type=float, default=1.0, help="sigma for smoothing")
    parser.add_argument('--cell2face', type=str, default='arithmetic_mean', help="cell2face method")
    args = parser.parse_args()

    img_network = args.image
    image2tdens(args.image,args.tdens2image, sigma_smoothing=args.sigma)
    

    