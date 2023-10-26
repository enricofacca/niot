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


def compute_lyapunov(img_sources,img_sinks,img_networks,label):
    # pure optimization problem
    weights=[0.0, 1.0, 0e-4]
    
    # convert images to numpy matrices
    # 1 is the white background
    scaling_size = 1.0
    factor = 5e3
    np_sources = i2d.image2matrix(img_sources,scaling_size)*factor
    np_sinks = i2d.image2matrix(img_sinks,scaling_size)*factor

    np_networks = i2d.image2matrix(img_networks,scaling_size)
    
    # the corrupted image is created adding a mask
    np_corrupted = np_networks
    np_confidence = np.ones(np_corrupted.shape)

    # Init. solver for a given reconstruction problem
    niot_solver = NiotSolver('CR1DG0', np_corrupted)

    # save inputs     
    fire_sources = niot_solver.numpy2function(np_sources, name="sources")
    fire_sinks = niot_solver.numpy2function(np_sinks, name="sinks")
    fire_networks = niot_solver.numpy2function(np_networks, name="networks")
    fire_corrupted = niot_solver.numpy2function(np_corrupted, name="corrupted")
    fire_confidence = niot_solver.numpy2function(np_confidence, name="confidence")

    if (not os.path.exists(label)):
        os.mkdir(label)
    directory = label

    kappa = 1.0
    niot_solver.set_optimal_transport_parameters(fire_sources, fire_sinks,
                                                  gamma=0.1, 
                                                  kappa=kappa, 
                                                  force_balance=True)
    print(f'mass injection = {assemble(niot_solver.source*dx)}')
    niot_solver.set_inpainting_parameters(weights=weights,
                                         confidence=fire_confidence)
    niot_solver.save_inputs(os.path.join(directory,'inputs.pvd'))

    # set controls
    ctrl = Controls(
        tol=1e-5,
        #time_discretization_method='mirror_descent',
        time_discretization_method='gfvar_gradient_descent',
        deltat=1e-3,
        max_iter=500,
        nonlinear_tol=1e-5,
        linear_tol=1e-6,
        nonlinear_max_iter=30,
        linear_max_iter=1000)

    ctrl.deltat_control = 'expansive'
    #ctrl.deltat_control = 'constant'
    ctrl.deltat_expansion = 1.05
    ctrl.deltat_min = 0.001
    ctrl.deltat_max = 0.5
    ctrl.verbose = 1

    ctrl.save_solution = 'no'
    ctrl.save_solution_every = 10
    ctrl.save_directory = './runs_y_network/'
    
    #
    # solve the problem
    #    
    sol = niot_solver.create_solution() # [pot, tdens] functions
    pot, tdens = sol.split()
    tdens.interpolate(fire_networks+1e-9)
    
    


    # solve the problem
    ierr = niot_solver.solve_pot_PDE(ctrl, sol)
    print("Error code: ",ierr)

    # save solution
    filename = os.path.join(directory,'sol.pvd')
    niot_solver.save_solution(sol,filename)
    
    
    # joule energy 
    joule = assemble(niot_solver.joule(pot, tdens))
    
    data = []
    for scale_tdens in 2**(np.linspace(-16,16,100)):
        print(f'{scale_tdens=:.2e}')
        for gamma in np.linspace(0.01,0.99,100):
            pot, tdens = sol.split()
            niot_solver.gamma = gamma
            w_mass = assemble(niot_solver.weighted_mass(pot, tdens))
            j = joule / scale_tdens
            wm = w_mass * scale_tdens ** gamma
            lyapunov = j + wm
            print(f'{gamma=:.2e} {lyapunov=:.2e} {j=:2e} {wm=:2e}')
            data.append([scale_tdens,gamma,lyapunov,j,wm])
    
    # save data as csv
    data = np.array(data)
    lyapunov = data[:,2]
    # find minumum of lyapunov and print corresponding parameters
    idx = np.argmin(lyapunov)
    print(f'lyapunov min = {lyapunov[idx]} {idx=}')
    print(f'scale_tdens = {data[idx,0]} gamma = {data[idx,1]}')
    np.savetxt(os.path.join(directory,'lyapunov.csv'),
               data,delimiter=',',
               fmt='%.3e',
               header='scale_tdens,gamma,lyapunov,joule,wmass',
               comments='')

    return ierr
    

if (__name__ == '__main__'):
    path_sources = sys.argv[1]
    path_sinks = sys.argv[2]
    path_networks = sys.argv[3]
    label = sys.argv[4]
    ierr = compute_lyapunov(path_sources,path_sinks,path_networks,label)