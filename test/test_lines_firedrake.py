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
from ufl.classes import Expr
from firedrake import UnitSquareMesh

from firedrake import norm
from firedrake import Function
from firedrake import interpolate
# for writing to file
from firedrake import File
import firedrake as fire


def corrupt_and_reconstruct(img_sources,img_sinks,img_networks,img_masks,directory,gamma=0.12,weights=[1e0,1e1,0.0]):
    print("Corrupting and reconstructing")
    print("Sources: "+img_sources)
    print("Sinks: "+img_sinks)
    print("Networks: "+img_networks)
    print("Masks: "+img_masks)

    # create a triangulation mesh from images
    factor = 0.5
    
    # convert images to numpy matrices
    # 1 is the white background
    np_sources = 2e1 * i2d.image2matrix(img_sources,factor)
    np_sinks = 2e1 * i2d.image2matrix(img_sinks,factor)
    np_networks = i2d.image2matrix(img_networks,factor)
    np_masks = i2d.image2matrix(img_masks,factor)
    
    # the corrupted image is created adding a mask
    # to the known network
    np_corrupted = np_networks * (1-np_masks)
    np_confidence = 1.0 - np_masks

    # Init. solver, set in/out flow and parameters
    niot_solver = NiotSolver('CR1DG0', np_corrupted, np_sources, np_sinks, force_balance=True)
    niot_solver.set_parameters(gamma=gamma, weights=weights)

    # save inputs 
    fire_sources = niot_solver.numpy2function(np_sources, name="sources")
    fire_sinks = niot_solver.numpy2function(np_sinks, name="sinks")
    fire_networks = niot_solver.numpy2function(np_networks, name="networks")
    fire_corrupted = niot_solver.numpy2function(np_corrupted, name="corrupted")
    fire_masks = niot_solver.numpy2function(np_masks, name="masks")
    fire_confidence = niot_solver.numpy2function(np_confidence, name="confidence")
    
    out_file = File(os.path.join(directory,"inputs_recostruction.pvd"))
    out_file.write(fire_sources, fire_sinks, fire_networks,fire_masks,fire_corrupted, fire_confidence)
    #niot_solver.save_inputs(os.path.join(directory,"inputs_recostruction.pvd"))
    
    # initialize a mixed function [pot,tdens] and solver controls
    sol = niot_solver.create_solution()

    
    ctrl = Controls(
        tol=1e-4,
        time_discretization_method='mirrow_descent',
        deltat=0.01,
        max_iter=1000,
        nonlinear_tol=1e-6,
        linear_tol=1e-6,
        nonlinear_max_iter=30,
        linear_max_iter=1000)

    ctrl.control_deltat = 'adaptive'
    ctrl.increment_deltat = 1.001
    ctrl.deltat_min = 0.005
    ctrl.deltat_max = 0.4
    ctrl.verbose = 1
    
    # solve the problem
    ierr = niot_solver.solve(sol, ctrl)
    
    # save data and print reconstructed images
    str_gamma = f"{niot_solver.gamma:.2e}"
    str_weight_discrepancy = f"{niot_solver.weights[1]:.2e}"
    label = "gamma_"+str_gamma+"_weight_discrepancy_"+str_weight_discrepancy

    niot_solver.save_solution(sol,os.path.join(directory,'sol'+label+'.pvd'))
    
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
    
    img_masks = sys.argv[2]
    gamma = float(sys.argv[3])
    weight_discrepancy = float(sys.argv[4])
    weights = np.array([1.0, weight_discrepancy, 1e-16])

    
    dir_mask = os.path.dirname(img_masks)
    out_dir = os.path.join(dir_mask,'runs')
    base_mask = os.path.splitext(os.path.basename(img_masks))[0]
    experiment_dir = os.path.join(out_dir,base_mask)
    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    if (not os.path.exists(experiment_dir)):
        os.mkdir(experiment_dir)
        
    corrupt_and_reconstruct(img_sources,img_sinks,img_networks,img_masks,experiment_dir,gamma,weights)
