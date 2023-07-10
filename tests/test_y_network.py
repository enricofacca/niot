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

import sys
import glob
import os
from copy import deepcopy as cp
import numpy as np
from scipy.io import savemat

sys.path.append('../src/niot')
from niot import NiotSolver
from niot import Controls
from niot import spread
from niot import heat_spread
import image2dat as i2d

from ufl import *
from firedrake import *
from firedrake import norm
from firedrake import Function
from firedrake import interpolate
# for writing to file
from firedrake import File
import firedrake as fire

import utilities as utilities


def y_network(ref, weight_discrepancy, gamma, regularation, label=''):
    # pure optimization problem
    weights=[weight_discrepancy, gamma, regularation]
    
    label += f'ref{ref:d}_{weight_discrepancy:.1e}_{gamma:.1e}_{regularation:.1e}'

    # create problem label   
    ndiv_x = 40*2**ref
    ndiv_y = 80*2**ref
    # x ~ columns
    # y ~ rows
    shape = [ndiv_y,ndiv_x]
    np_sources = np.zeros(shape)
    np_sinks = np.zeros(shape)

    size = ndiv_x//10
    scale = 1e1

    np_sources[-size:,ndiv_x//2-size:ndiv_x//2+size] = 1.0 *scale
    size=ndiv_x//2
    np_sinks[0:size:,0:size] = 0.5 * scale
    np_sinks[0:size:,-size:] = 0.5 * scale  

    extra_width = ndiv_x//2
    extra = np.zeros([ndiv_y,extra_width])
    np_sources = np.concatenate([extra,np_sources,extra],axis=1)
    np_sinks = np.concatenate([extra,np_sinks,extra],axis=1)


    np_networks = np.zeros(np_sources.shape)
    np_masks = np.zeros(np_sources.shape)
    # the corrupted image is created adding a mask
    # to the known network
    np_corrupted = np_networks * (1-np_masks)
    np_confidence = np.ones(np_corrupted.shape)

    # Init. solver for a given reconstruction problem
    niot_solver = NiotSolver('CR1DG0', np_corrupted)
    label='CR1_'+label
    

    # save inputs     
    fire_sources = niot_solver.numpy2function(np_sources, name="sources")
    fire_sinks = niot_solver.numpy2function(np_sinks, name="sinks")
    fire_networks = niot_solver.numpy2function(np_networks, name="networks")
    fire_corrupted = niot_solver.numpy2function(np_corrupted, name="corrupted")
    fire_masks = niot_solver.numpy2function(np_masks, name="masks")
    fire_confidence = niot_solver.numpy2function(np_confidence, name="confidence")
    CG1 = fire.FunctionSpace(niot_solver.mesh ,'CG',1)
    fire_masks_for_contour = Function(CG1)
    fire_masks_for_contour.interpolate(fire_masks).rename("mask_countour","masks")
    fire_networks_for_contour = Function(CG1)
    fire_networks_for_contour.interpolate(fire_networks).rename("networks_countour","networks")
    fire_corrupted_for_contour = Function(CG1)
    fire_corrupted_for_contour.interpolate(fire_corrupted).rename("corrupted_countour","corrupted")


    directory = "./runs_y_network/"

    filename = os.path.join(directory,f"ref{ref}_inputs.pvd")
    utilities.save2pvd([fire_sources,
                     fire_sinks,
                     fire_networks,
                     fire_networks_for_contour,
                     fire_masks,
                     fire_masks_for_contour,
                     fire_corrupted,
                    fire_corrupted_for_contour,
                     fire_confidence],
                    filename)

    kappa = 1.0
    niot_solver.set_optimal_transport_parameters(fire_sources, fire_sinks,
                                                  gamma=gamma, 
                                                  kappa=kappa, 
                                                  force_balance=True)
    niot_solver.set_inpainting_parameters(weights=weights,
                                         confidence=fire_confidence)
    
    # set controls
    ctrl = Controls(
        tol=1e-2,
        #time_discretization_method='tdens_mirror_descent',
        time_discretization_method='mirror_descent',
        #time_discretization_method='gfvar_gradient_descent',
        deltat=1e-3,
        max_iter=2000,
        nonlinear_tol=1e-5,
        linear_tol=1e-6,
        nonlinear_max_iter=30,
        linear_max_iter=1000)

    ctrl.deltat_control = 'expansive'
    #ctrl.deltat_control = 'constant'
    #ctrl.deltat_control = 'adaptive'
    ctrl.deltat_expansion = 1.01
    ctrl.deltat_min = 1e-3
    ctrl.deltat_max = 0.05
    ctrl.verbose = 1
    ctrl.max_restarts = 5

    ctrl.save_solution = 'some'
    ctrl.save_solution_every = 10
    ctrl.save_directory = directory
    
    #
    # solve the problem
    #    
    sol = niot_solver.create_solution() # [pot, tdens] functions
    ierr = niot_solver.solve(ctrl, sol)
    print("Error code: ",ierr)

    # save solution
    filename = os.path.join(directory,label+'.pvd')
    niot_solver.save_solution(sol,filename)

    #filename = os.path.join(directory,'y_network_checkpoint.h5')
    #niot_solver.save_checkpoint(sol, filename)
    #sol = niot_solver.load_checkpoint(filename)

    with sol.dat.vec as d:
        tdens_vec = d.getSubVector(niot_solver.fems.tdens_is)
        np_tdens = tdens_vec.array

    filename = os.path.join(directory,label+'.mat')
    savemat(filename, mdict={'tdens': np_tdens})

    return ierr
    

if (__name__ == '__main__'):
    # parse command line arguments
    ref= int(sys.argv[1])
    weight_discrepancy = float(sys.argv[2])
    gamma = float(sys.argv[3])
    regularation = float(sys.argv[4])
    try:
        label = sys.argv[5]
    except:
        label = ''
    ierr = y_network(ref, weight_discrepancy, gamma, regularation, label)