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

sys.path.append('../src/niot')
from niot import NiotSolver
from niot import InpaitingProblem
from niot import SpaceDiscretization
from niot import Controls
from niot import msg_bounds
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

from scipy.ndimage.filters import gaussian_filter

def reconstruct(img_sources,img_sinks,img_networks,directory, gamma=0.12,weights=[1e0,1e1,0.0],corrupted_as_initial_guess=0, scaling_size=1.0):
    print("Reconstructing")
    print("Sources: "+img_sources)
    print("Sinks: "+img_sinks)
    print("Networks: "+img_networks)
    
    # create problem label
    label = [f"scale{scaling_size:.2f}",
             f"gamma{gamma:.1e}",
             f"weight{weights[1]:.1e}",
             f"initial{curropted_as_initial_guess:d}"]
    label = "_".join(label)
    print("Problem label: "+label)


    # convert images to numpy matrices
    # 1 is the white background
    np_sources =i2d.image2matrix(img_sources,scaling_size)
    np_sinks = i2d.image2matrix(img_sinks,scaling_size)
    np_networks = i2d.image2matrix(img_networks,scaling_size)
    
    # the corrupted image is created adding a mask
    # to the known network
    np_corrupted = np_networks

    # define the confidence function
    blurred_networks = gaussian_filter(np_networks, sigma=1)


    np_confidence = np.ones(np_networks.shape)

    # Init. solver, set in/out flow and parameters
    niot_solver = NiotSolver('CR1DG0', np_corrupted, np_sources, np_sinks, force_balance=True)
    niot_solver.set_parameters(gamma=gamma, weights=weights)

    # save inputs     
    fire_sources = niot_solver.numpy2function(np_sources, name="sources")
    fire_sinks = niot_solver.numpy2function(np_sinks, name="sinks")
    fire_corrupted = niot_solver.numpy2function(np_corrupted, name="corrupted")
    fire_confidence = niot_solver.numpy2function(np_confidence, name="confidence")
    fire_blurred_networks = niot_solver.numpy2function(blurred_networks, name="blurred_networks")

    out_file = File(os.path.join(directory,"inputs_recostruction.pvd"))
    out_file.write(fire_sources, fire_sinks, fire_corrupted, fire_confidence, fire_blurred_networks)
    #niot_solver.save_inputs(os.path.join(directory,"inputs_recostruction.pvd"))
    
    #sys.exit()
    
    ctrl = Controls(
        tol=1e-4,
        time_discretization_method='mirrow_descent',
        deltat=0.01,
        max_iter=300,
        nonlinear_tol=1e-6,
        linear_tol=1e-6,
        nonlinear_max_iter=30,
        linear_max_iter=1000)

    ctrl.deltat_control = 'expansive'
    ctrl.deltat_expansion = 1.05
    ctrl.deltat_min = 0.005
    ctrl.deltat_max = 0.05
    ctrl.verbose = 1
    
    #
    # solve the problem
    #
        
    sol = niot_solver.create_solution() # [pot, tdens] functions
    if corrupted_as_initial_guess == 1:
        sol.sub(1).assign(1e2*fire_corrupted+1e-6)
    
    # compute pot associated to new tdens
    ierr = niot_solver.syncronize( sol, ctrl)

    niot_solver.save_solution(sol,os.path.join(directory,'initial_'+label+'.pvd'))
    
    ierr = niot_solver.solve(sol, ctrl)
    
    

    niot_solver.save_solution(sol,os.path.join(directory,'sol_'+label+'.pvd'))
    
    pot, tdens = sol.split()
    tdens_plot = Function(niot_solver.fems.tdens_space).interpolate(tdens)
    support = Function(niot_solver.fems.tdens_space).interpolate(conditional(tdens>1e2*ctrl.tdens_min,1,0))    
    i2d.function2image(tdens_plot,os.path.join(directory,label+"tdens.png"),vmin=ctrl.tdens_min)
    i2d.function2image(tdens_plot,os.path.join(directory,label+"support_tdens.png"),vmin=ctrl.tdens_min)



    
    
def get_image_path(directory_example, name):
    string = os.path.join(directory_example,name)
    print("Looking for "+string)
    matches = glob.glob(string)
    if len(matches) == 0:
        print("Error: no image "+name+" found in "+directory_example)
        exit(1)
    elif len(matches) > 1:
        print("Error: more than one image "+name+" found in "+directory_example)
        exit(1)
    else:       
        return matches[0]


if (__name__ == '__main__'):
    # get paths of input images (sources, sinks, networks in one directory)
    # and a mask image.
    # The mask image is used to corrupt the network image
    # Results are saved in a directory having the same name of the mask image stored in the runs directory.
    # The latter is created where the sources, sinks and networks images are stored.
    # The gamma and weight discrepancy parameters are passed.
    # 
    # example: python3 corrupt_reconstruct.py example examples/mask.png 1e-2 1e-2
    # 
    # results are stored in runs/mask
    directory_example = os.path.dirname(sys.argv[1])
    
    img_sources = get_image_path(directory_example,'*sources*.png')
    img_sinks = get_image_path(directory_example,'*sinks*.png')
    img_networks = get_image_path(directory_example,'*networks*.png')
    
    gamma = float(sys.argv[2])
    weight_discrepancy = float(sys.argv[3])
    curropted_as_initial_guess = int(sys.argv[4])
    scaling_size = float(sys.argv[5])
    weights = np.array([1.0, weight_discrepancy, 1e-16])

    
    dir_mask = os.path.dirname(img_networks)
    out_dir = os.path.join(dir_mask,'runs')
    experiment_dir = out_dir
    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    
    reconstruct(img_sources,img_sinks,img_networks,experiment_dir,gamma,weights,curropted_as_initial_guess,scaling_size=scaling_size)
