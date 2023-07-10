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

from scipy.ndimage.filters import gaussian_filter

def corrupt_and_reconstruct(img_sources,img_sinks,img_networks,img_masks,directory, gamma=0.12,weights=[1e0,1e1,0.0],corrupted_as_initial_guess=0, scaling_size=1.0):
    print("Corrupting and reconstructing")
    print("Sources: "+img_sources)
    print("Sinks: "+img_sinks)
    print("Networks: "+img_networks)
    print("Masks: "+img_masks)


    # create problem label
    label = [f"scale{scaling_size:.2f}",
             f"discr{weights[0]:.1e}",
             f"gamma{gamma:.1e}",
             f"reg{weights[2]:.1e}",
             f"initial{curropted_as_initial_guess:d}"]
    label = "_".join(label)
    print("Problem label: "+label)


    # convert images to numpy matrices
    # 1 is the white background
    scaling_forcing = 1e1
    np_sources = i2d.image2matrix(img_sources,scaling_size) * scaling_forcing
    np_sinks = i2d.image2matrix(img_sinks,scaling_size) * scaling_forcing
    np_networks = i2d.image2matrix(img_networks,scaling_size)
    np_masks = i2d.image2matrix(img_masks,scaling_size)
    
    # the corrupted image is created adding a mask
    # to the known network
    np_corrupted = np_networks * (1-np_masks)

    # define the confidence function
    blurred_networks = gaussian_filter(np_networks, sigma=1)


    np_confidence = (1.0 - np_masks) #/ (blurred_networks + 1e-1)

    # Init. solver for a given reconstruction problem
    niot_solver = NiotSolver('DG0DG0', np_corrupted)
    
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


    filename = os.path.join(directory,"inputs_recostruction.pvd")
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
    #niot_solver.save_inputs(os.path.join(directory,"inputs_recostruction.pvd"))
    
    #kappa = 1.0/(1.0 + fire_confidence + 1e-4)
    kappa = 1.0

    niot_solver.set_optimal_transport_parameters(fire_sources, fire_sinks,
                                                  gamma=gamma, 
                                                  kappa=kappa, 
                                                  force_balance=True)
    niot_solver.set_inpainting_parameters(weights=weights,
                                         confidence=fire_confidence)
    
    niot_solver.save_inputs(os.path.join(directory,"parameters_recostruction.pvd"))
    
    ctrl = Controls(
        tol=1e-2,
        time_discretization_method='mirror_descent',
        deltat=1e-4,
        max_iter=4000,
        nonlinear_tol=1e-6,
        linear_tol=1e-6,
        nonlinear_max_iter=30,
        linear_max_iter=1000)

    ctrl.deltat_control = 'expansive'#'adaptive'
    ctrl.deltat_expansion = 1.01
    ctrl.deltat_min = 1e-3
    ctrl.deltat_max = 0.05
    ctrl.verbose = 1
    ctrl.max_restarts = 5
    
    ctrl.save_solution = 'no'
    ctrl.save_solution_every = 10
    ctrl.save_directory = os.path.join(directory,'evolution')
    
    if (not os.path.exists(ctrl.save_directory)):
        os.makedirs(ctrl.save_directory)
    
    #
    # solve the problem
    #
        
    sol = niot_solver.create_solution() # [pot, tdens] functions
    if corrupted_as_initial_guess == 1:
        sol.sub(1).assign(fire_corrupted+1e-6)
    
    # solve the problem
    ierr = niot_solver.solve(ctrl, sol)
    print("Error code: ",ierr)

    # save solution
    filename = os.path.join(directory,'sol_'+label+'.pvd')
    print("Saving solution to "+filename)
    niot_solver.save_solution(sol,os.path.join(directory,'sol_'+label+'.pvd'))

    filename = os.path.join(directory, f'sol_{label}.h5')
    niot_solver.save_checkpoint(sol, filename)
    #sol = niot_solver.load_checkpoint(filename)

    _, tdens = sol.subfunctions
    DG0 = FunctionSpace(sol.function_space().mesh(),'DG',0)
    tdens_plot = Function(DG0)
    tdens_plot.interpolate(tdens)

   
    support = Function(DG0).interpolate(conditional(tdens>1e2*ctrl.tdens_min,1,0))    
    i2d.function2image(tdens_plot, os.path.join(directory,label+"tdens.png"),vmin=ctrl.tdens_min)
    i2d.function2image(support, os.path.join(directory, label+"support_tdens.png"),vmin=ctrl.tdens_min)

    return 
    tdens_smooth_1 = heat_spread(tdens_plot,tau=1e-4)
    tdens_smooth_1.rename("tdens_smooth_1","tdens_smooth_1")

    #tdens_smooth_15 = spread(tdens_plot,tau=1e-5,m_exponent=1.5)
    #tdens_smooth_15.rename("tdens_smooth_15","tdens_smooth_15")
    

    tdens_smooth_2 = spread(tdens_plot,tau=1e-4,m_exponent=2, nsteps=100)
    tdens_smooth_2.rename("tdens_smooth_2","tdens_smooth_2")
    

    filename = os.path.join(directory,'smooth_tdens'+label+'.pvd')
    utilities.save2pvd([tdens_smooth_1,
                        #tdens_smooth_15,
                        tdens_smooth_2],
                        filename)








    
    
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
    
    #img_sources = get_image_path(directory_example,'*sources*.png')
    #img_sinks = get_image_path(directory_example,'*sinks*.png')
    #img_networks = get_image_path(directory_example,'*networks*.png')

    img_sources, img_sinks, img_networks, img_masks = sys.argv[1:5]

    nargs = 5
    directory = os.path.abspath(sys.argv[nargs])
    nargs += 1
    gamma = float(sys.argv[nargs])
    nargs += 1
    weight_discrepancy = float(sys.argv[nargs])
    nargs += 1
    weight_regularization = float(sys.argv[nargs])
    nargs += 1
    curropted_as_initial_guess = int(sys.argv[nargs])
    nargs += 1
    scaling_size = float(sys.argv[nargs])

    print ("directory = ",directory)
    print ("gamma = ",gamma)
    print ("weight_discrepancy = ",weight_discrepancy)
    print ("weight_regularization = ",weight_regularization)
    print ("curropted_as_initial_guess = ",curropted_as_initial_guess)
    print ("scaling_size = ",scaling_size)
    
    weights = np.array([weight_discrepancy, 1.0, weight_regularization])

    
    dir_mask = os.path.dirname(img_masks)
    out_dir = os.path.join(dir_mask,'runs')
    base_mask = os.path.splitext(os.path.basename(img_masks))[0]
    experiment_dir = os.path.join(out_dir,base_mask)
    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    if (not os.path.exists(experiment_dir)):
        os.mkdir(experiment_dir)
        
    corrupt_and_reconstruct(img_sources,img_sinks,img_networks,img_masks,experiment_dir,gamma,weights,curropted_as_initial_guess,scaling_size=scaling_size)

    