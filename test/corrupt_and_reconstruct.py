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

import utilities as utilities

from scipy.ndimage.filters import gaussian_filter

def corrupt_and_reconstruct(img_sources,img_sinks,img_networks,img_masks,directory, gamma=0.12,weights=[1e0,1e1,0.0],corrupted_as_initial_guess=0, scaling_size=1.0):
    print("Corrupting and reconstructing")
    print("Sources: "+img_sources)
    print("Sinks: "+img_sinks)
    print("Networks: "+img_networks)
    print("Masks: "+img_masks)


    # create problem label
    label = [f"scale{scaling_size:.2f}",
             f"gamma{gamma:.1e}",
             f"weight{weights[1]:.1e}",
             f"initial{curropted_as_initial_guess:d}"]
    label = "_".join(label)
    print("Problem label: "+label)


    # convert images to numpy matrices
    # 1 is the white background
    np_sources = i2d.image2matrix(img_sources,scaling_size)
    np_sinks = i2d.image2matrix(img_sinks,scaling_size)
    np_networks = i2d.image2matrix(img_networks,scaling_size)
    np_masks = i2d.image2matrix(img_masks,scaling_size)
    
    # the corrupted image is created adding a mask
    # to the known network
    np_corrupted = np_networks * (1-np_masks)

    # define the confidence function
    blurred_networks = gaussian_filter(np_networks, sigma=1)


    np_confidence = (1.0 - np_masks) #/ (blurred_networks + 1e-1)

    # Init. solver for a given reconstruction problem
    niot_solver = NiotSolver('CR1DG0', np_corrupted)
    

    # save inputs     
    fire_sources = niot_solver.numpy2function(np_sources, name="sources")
    fire_sinks = niot_solver.numpy2function(np_sinks, name="sinks")
    fire_networks = niot_solver.numpy2function(np_networks, name="networks")
    fire_corrupted = niot_solver.numpy2function(np_corrupted, name="corrupted")
    fire_masks = niot_solver.numpy2function(np_masks, name="masks")
    CG1 = fire.FunctionSpace(niot_solver.mesh ,'CG',1)
    fire_masks_for_contour = Function(CG1)
    fire_masks_for_contour.interpolate(fire_masks).rename("mask_countour","masks")
    fire_confidence = niot_solver.numpy2function(np_confidence, name="confidence")
    fire_blurred_networks = niot_solver.numpy2function(blurred_networks, name="blurred_networks")

    out_file = File(os.path.join(directory,"inputs_recostruction.pvd"))
    out_file.write(fire_sources, fire_sinks, fire_networks,fire_masks,fire_masks_for_contour,fire_corrupted, fire_confidence, fire_blurred_networks)
    #niot_solver.save_inputs(os.path.join(directory,"inputs_recostruction.pvd"))
    
    #kappa = fire_confidence + 1.0
    kappa = 1.0

    niot_solver.set_optimal_transport_parameters(fire_sources, fire_sinks,
                                                  gamma=gamma, 
                                                  kappa=kappa, 
                                                  force_balance=True)
    niot_solver.set_inpainting_parameters(weights=weights,
                                         confidence=fire_confidence)
    
    niot_solver.save_inputs(os.path.join(directory,"parameters_recostruction.pvd"))
    
    ctrl = Controls(
        tol=1e-4,
        time_discretization_method='mirror_descent',
        deltat=0.1,
        max_iter=200,
        nonlinear_tol=1e-6,
        linear_tol=1e-6,
        nonlinear_max_iter=30,
        linear_max_iter=1000)

    ctrl.deltat_control = 'expansive'
    ctrl.deltat_expansion = 1.05
    ctrl.deltat_min = 0.01
    ctrl.deltat_max = 0.2
    ctrl.verbose = 1
    
    #
    # solve the problem
    #
        
    sol = niot_solver.create_solution() # [pot, tdens] functions
    if corrupted_as_initial_guess == 1:
        sol.sub(1).assign(1e2*fire_corrupted+1e-6)
    
    # solve the problem
    ierr = niot_solver.solve(ctrl, sol)
    print("Error code: ",ierr)

    # save solution
    filaname = os.path.join(directory,'sol_'+label+'.pvd')
    print("Saving solution to "+filaname)
    niot_solver.save_solution(sol,os.path.join(directory,'sol_'+label+'.pvd'))
    
    pot, tdens = sol.split()
    tdens_plot = Function(niot_solver.fems.tdens_space).interpolate(tdens)
    support = Function(niot_solver.fems.tdens_space).interpolate(conditional(tdens>1e2*ctrl.tdens_min,1,0))    
    i2d.function2image(tdens_plot,os.path.join(directory,label+"tdens.png"),vmin=ctrl.tdens_min)
    i2d.function2image(tdens_plot,os.path.join(directory,label+"support_tdens.png"),vmin=ctrl.tdens_min)

    tdens_smooth_0 = niot_solver.heat_smoothing(tdens_plot,tau=1e-4)
    tdens_smooth_0.rename("tdens_smooth_1","tdens_smooth_1")
    tdens_smooth_1 = niot_solver.heat_smoothing(tdens_plot,tau=1e-8,tdens=tdens_plot)
    tdens_smooth_1.rename("tdens_smooth_tdens","tdens_smooth_tdens")

    filename = os.path.join(directory,'smooth_tdens'+label+'.pvd')
    print("Saving smoothed tdens to "+filename)
    file = File(filename)
    file.write(tdens_smooth_0,tdens_smooth_1)








    
    
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
    
    print('arg:',sys.argv[1:])

    img_masks = sys.argv[2]
    gamma = float(sys.argv[3])
    weight_discrepancy = float(sys.argv[4])
    curropted_as_initial_guess = int(sys.argv[5])
    scaling_size = float(sys.argv[6])
    weights = np.array([1.0, weight_discrepancy, 1e-16])

    
    dir_mask = os.path.dirname(img_masks)
    out_dir = os.path.join(dir_mask,'runs')
    base_mask = os.path.splitext(os.path.basename(img_masks))[0]
    experiment_dir = os.path.join(out_dir,base_mask)
    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    if (not os.path.exists(experiment_dir)):
        os.mkdir(experiment_dir)
        
    corrupt_and_reconstruct(img_sources,img_sinks,img_networks,img_masks,experiment_dir,gamma,weights,curropted_as_initial_guess,scaling_size=scaling_size)

    