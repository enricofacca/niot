import sys
import glob
import os
from copy import deepcopy as cp

sys.path.append('../src/niot')
from niot import NiotSolver
from niot import InpaitingProblem
from niot import TdensPotential
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
    mesh = i2d.image2grid(img_sources,factor)

    # convert images to numpy matrices
    # 1 is the white background
    np_sources = 2e1 * i2d.image2matrix(img_sources,factor)
    np_sinks = 2e1 * i2d.image2matrix(img_sinks,factor)
    np_networks = i2d.image2matrix(img_networks,factor)
    np_masks = i2d.image2matrix(img_masks,factor)
    
    # the corrupted image is created adding a mask
    # to the known network
    np_corrupted = np_networks * (1-np_masks)
    np_confidence = 1-np_masks
    #np_confidence = np.ones(np_masks.shape)


    # convert to firedrake function
    fire_sources = i2d.matrix2function(np_sources,mesh)
    fire_sources.rename("sources")
    fire_sinks = i2d.matrix2function(np_sinks,mesh)
    fire_sinks.rename("sinks")
    fire_networks = i2d.matrix2function(np_networks,mesh)
    fire_networks.rename("networks")
    fire_corrupted = i2d.matrix2function(np_corrupted,mesh)
    fire_corrupted.rename("corrupted")
    fire_masks = i2d.matrix2function(np_masks,mesh)
    fire_masks.rename("masks")
    fire_confidence = i2d.matrix2function(np_confidence,mesh)
    fire_confidence.rename("confidence")

    out_file = File(os.path.join(directory,"inputs_recostruction.pvd"))
    out_file.write(fire_sources, fire_sinks, fire_networks,fire_masks,fire_corrupted, fire_confidence)

    

    # set solver inputs 
    inputs = InpaitingProblem(fire_corrupted,fire_sources,fire_sinks,force_balance=True)

    # Init. niot solver and set parameters 
    fem = SpaceDiscretization(mesh)
    niot_solver = NiotSolver(fem)
    niot_solver.set_parameters(
        gamma=gamma, # branching exponent  
        #confidence = fire_confidence, # confidence function
        weights=[1.0,weight_discrepancy,0.0])

    # create a mixed function [pot,tdens] and controls
    sol = fem.create_solution()
    ctrl = Controls(
        tol=1e-4,
        time_discretization_method='mirrow_descent',
        deltat=0.01,
        max_iter=200,
        nonlinear_tol=1e-6,
        linear_tol=1e-6,
        nonlinear_max_iter=30,
        linear_max_iter=1000)

    ctrl.control_deltat = 'adaptive'
    ctrl.increment_deltat = 1.001
    ctrl.deltat_min = 0.01
    ctrl.deltat_max = 0.4
    ctrl.verbose = 1
    
    # solve the problem
    ierr = niot_solver.solve(inputs, sol, ctrl)
    
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
    # Results are saved in a subdirectory of the mask image directory
    directory_example = os.path.dirname(sys.argv[1])
    
    img_sources = get_image_path(directory_example,'*sources*.png')
    img_sinks = get_image_path(directory_example,'*sinks*.png')
    img_networks = get_image_path(directory_example,'*networks*.png')
    
    img_masks = sys.argv[2]
    gamma = float(sys.argv[3])
    weight_discrepancy = float(sys.argv[4])


    
    dir_mask = os.path.dirname(img_masks)
    out_dir = os.path.join(dir_mask,'runs')
    base_mask = os.path.splitext(os.path.basename(img_masks))[0]
    experiment_dir = os.path.join(out_dir,base_mask)
    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    if (not os.path.exists(experiment_dir)):
        os.mkdir(experiment_dir)
        
    corrupt_and_reconstruct(img_sources,img_sinks,img_networks,img_masks,experiment_dir,gamma,weight_discrepancy)
