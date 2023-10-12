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

#@profile
def corrupt_and_reconstruct(img_source,img_sink,img_network,img_mask, 
                            scaling_size,
                            fem,
                            gamma, weights, corrupted_as_initial_guess,
                            confidence,
                            tdens2image,
                            directory,
                            runs_directory,
                            sigma_smoothing=1e-6):
    print('Corrupting and reconstructing')
    print('Sources: '+img_source)
    print('Sinks: '+img_sink)
    print('Networks: '+img_network)
    print('Masks: '+img_mask)


    # create problem label
    if abs(scaling_size-1)>1e-16:
        scaling_size = 1.0
        label = [f'scale{scaling_size:.2f}']
    else:
        label = []
    label+= [f'fem{fem}',
             f'gamma{gamma:.1e}',
             f'wd{weights[0]:.1e}',
             f'wr{weights[2]:.1e}',
             f'ini{corrupted_as_initial_guess:d}',
             f'conf{confidence}']
    if tdens2image == 'laplacian_smoothing':
        label.append(f'mu2i{tdens2image}sigma{sigma_smoothing:.1e}')
    elif tdens2image == 'identity':
        label.append(f'mu2i{tdens2image}')
    label = '_'.join(label)
    print('Problem label: '+label)


    # convert images to numpy matrices
    # 1 is the white background
    scaling_forcing = 1e1
    np_source = i2d.image2numpy(img_source,normalize=True,invert=True,factor=scaling_size) * scaling_forcing
    np_sink = i2d.image2numpy(img_sink,normalize=True,invert=True,factor=scaling_size) * scaling_forcing
    np_network = i2d.image2numpy(img_network,normalize=True,invert=True,factor=scaling_size)
    np_mask = i2d.image2numpy(img_mask,normalize=True,invert=True,factor=scaling_size)
    nx = np_source.shape[0]
    ny = np_source.shape[1]


    # the corrupted image is created adding a mask
    # to the known network
    np_corrupted = np_network * (1-np_mask)
    if confidence == 'MASK':
        np_confidence = (1.0 - np_mask)
    elif confidence == 'CORRUPTED':
        np_confidence = gaussian_filter(np_corrupted, sigma=2)
        #normalize to [0,1]
        np_confidence = np_confidence/np.max(np_confidence)
    elif confidence == 'ONE':
        np_confidence = np.ones(np_masks.shape)
    else:
        raise ValueError(f'Unknown confidence {confidence}')


    # Init. solver for a given reconstruction problem
    #niot_solver = NiotSolver(fem, np_corrupted, DG0_cell2face = 'harmonic_mean')
    mesh = i2d.build_mesh_from_numpy(np_corrupted)


    # save inputs    
    source = i2d.numpy2firedrake(mesh, np_source, name="source")
    sink = i2d.numpy2firedrake(mesh, np_sink, name="sink")
    network = i2d.numpy2firedrake(mesh, np_network, name="network")
    corrupted = i2d.numpy2firedrake(mesh, np_corrupted, name="corrupted")
    masks = i2d.numpy2firedrake(mesh, np_mask, name="mask")
    confidence = i2d.numpy2firedrake(mesh, np_confidence, name="confidence")
    CG1 = fire.FunctionSpace(mesh ,'CG',1)
    masks_for_contour = Function(CG1)
    masks_for_contour.interpolate(masks).rename("mask_countour","masks")
    networks_for_contour = Function(CG1)
    networks_for_contour.interpolate(network).rename("networks_countour","networks")
    corrupted_for_contour = Function(CG1)
    corrupted_for_contour.interpolate(corrupted).rename("corrupted_countour","corrupted")
    confidence_for_contour = Function(CG1)
    confidence_for_contour.interpolate(confidence).rename("confidence_countour","confidence")

    filename = os.path.join(directory,"inputs_reconstruction.pvd")
    # utilities.save2pvd([fire_sources,
    #                  fire_sinks,
    #                  fire_networks,
    #                  fire_networks_for_contour,
    #                  fire_masks,
    #                  fire_masks_for_contour,
    #                  fire_corrupted,
    #                  fire_corrupted_for_contour,
    #                  fire_confidence,
    #                  fire_confidence_for_contour
    #                  ],
    #                 filename)

    #kappa = 1.0/(1.0 + fire_confidence + 1e-4)
    
    # kappa = 1.0
    # niot_solver.set_optimal_transport_parameters(fire_sources, fire_sinks,
    #                                               gamma=gamma, 
    #                                               kappa=kappa, 
    #                                               force_balance=True,
    #                                               min_tdens=1e-8)
    # niot_solver.set_inpainting_parameters(weights=weights,
    #                                      confidence=fire_confidence,
    #                                      tdens2image=tdens2image,
    #                                      sigma_smoothing=sigma_smoothing)
    
    #niot_solver.save_inputs(os.path.join(directory,"parameters_recostruction.pvd"))
    ctrl = Controls(
        # globals controls
        tol=1e-2,
        max_iter=10,
        spaces=fem,
        # niot controls
        gamma=gamma,
        weight_discrepancy=weights[0],
        #weight_penalty=weights[1],
        weight_regularization=weights[2],
        tdens2image=tdens2image,
        # optimization controls
        time_discretization_method='tdens_mirror_descent',
        deltat=1e-3,
        nonlinear_tol=1e-5,
        linear_tol=1e-6,
        nonlinear_max_iter=30,
        linear_max_iter=1000)

    # set "manually" the controls
    ctrl.deltat_control = 'expansive'#'adaptive'
    ctrl.deltat_expansion = 1.05
    ctrl.deltat_min = 1e-7
    ctrl.deltat_max = 1e-1
    ctrl.verbose = 1
    ctrl.max_restarts = 7
    ctrl.save_solution = 'no'
    ctrl.save_solution_every = 25
    ctrl.save_directory = os.path.join(directory,'evolution')
    
    if (not os.path.exists(ctrl.save_directory)):
        os.makedirs(ctrl.save_directory)
    
    #
    # solve the problem
    #
    ot.balance(source, sink)
    opt_inputs = ot.OTPInputs(source, sink)
    btp = ot.BranchedTransportProblem(source, sink, gamma=ctrl.gamma)

    niot_solver = NiotSolver(btp, corrupted,  confidence, ctrl)      
    sol = niot_solver.create_solution()
    if corrupted_as_initial_guess == 1:
        sol.sub(1).assign(corrupted+1e-6)

    pot, tdens = sol.subfunctions
    reconstruction = niot_solver.tdens2image(tdens)

    #print('solving pot_0')
    #ierr = niot_solver.solve_pot_PDE(ctrl, sol)
    #avg_outer = niot_solver.outer_iterations
    #print(utilities.color('green',f'It: {0} '+f' avgouter: {avg_outer:.1f}'))
    #initial_sol = cp(sol)

    # solve the problem
    ierr = niot_solver.solve(ctrl, sol)
    print("Error code: ",ierr)

    # save solution
    
    
    filename = os.path.join(runs_directory, f'{label}.h5')
    pot, tdens = sol.subfunctions
    pot.rename('pot','Potential')
    tdens.rename('tdens','Optimal Transport Tdens')

    DG0_vec = VectorFunctionSpace(niot_solver.mesh,'DG',0)
    vel = Function(DG0_vec)
    vel.interpolate(-tdens * grad(pot))
    vel.rename('vel','Velocity')

    #initial_tdens = Function(niot_solver.fems.tdens_space)
    #initial_tdens.interpolate(initial_sol.subfunctions[1])
    #initial_tdens.rename('initial_tdens','Initial guess')

    #initial_pot = Function(niot_solver.fems.pot_space)
    #initial_pot.interpolate(initial_sol.subfunctions[0])
    #initial_pot.rename('initial_pot','Initial guess')

    reconstruction = niot_solver.tdens2image(tdens) 
    reconstruction.rename('reconstruction','Reconstruction')

    filename = os.path.join(runs_directory, f'{label}.pvd')
    print("Saving solution to "+filename)
    niot_solver.save_solution(sol,filename)
    utilities.save2pvd([pot, tdens, vel, 
                        niot_solver.source,
                        niot_solver.sink,
    #                    initial_pot,initial_tdens,
                        reconstruction,
                        network,
                        networks_for_contour,
                        masks,
                        masks_for_contour,
                        corrupted,
                        corrupted_for_contour,
                        confidence,
                        confidence_for_contour],filename)

    # save numpy arrays
    np_reconstruction = i2d.firedrake2numpy(reconstruction,[nx,ny])
    filename = os.path.join(runs_directory, f'{label}.npy')
    np.save(filename, np_reconstruction)
    
    # save images
    img_name = os.path.join(runs_directory, f'{label}_tdens.png')
    i2d.numpy2image(np_reconstruction, img_name, normalized=True, inverted=True)

    img_name = os.path.join(runs_directory, f'{label}_support.png')
    np_reconstruction = np_reconstruction/np.max(np_reconstruction)
    i2d.numpy2image(np_reconstruction, img_name, normalized=True, inverted=True)

    




    
    
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
    # example: python3 corrupt_reconstruct.py --network=network.png --mask=mask.png
    #     --source=source.png --sink=sink.png 
    #     --scale=0.5 --wd=10 --gamma=0.8
    #     --dir=example_mask
    # 
    # results are stored in example_mask/runs
    
    parser = argparse.ArgumentParser(description='Corrupt networks with masks and reconstruct via branched transport')
    parser.add_argument('--network', type=str, help="path for the network image")
    parser.add_argument('--mask', type=str, help="path to mask image")
    parser.add_argument('--source', type=str, help="path for source")
    parser.add_argument('--sink', type=str, help="path for sink")
    parser.add_argument('--scale', type=float, default=1.0, help="scaling (between 0 and 1) of the mask image")
    parser.add_argument('--fem', type=str, default='CR1DG0', help='CR1DG0,DG0DG0')
    parser.add_argument('--wd', type=float, default=1.0, help="weight discrepancy")
    parser.add_argument('--wp', type=float, default=1.0, help="weight penalty")
    parser.add_argument('--wr', type=float, default=0.0, help="weight regularation")
    parser.add_argument('--gamma', type=float, default=0.5, help="branch exponent")
    parser.add_argument('--ini', type=int, default=0, help="0/1 use corrupted image as initial guess")
    parser.add_argument('--conf', type=str, default='MASK', help="MASK(=opposite to mask),CORRUPTED(=gaussian filter of corrupted)")
    parser.add_argument('--t2i', type=str, default='identity', help="identity,laplacian_smoothing")
    parser.add_argument('--dir', type=str, default='niot_recostruction', help="directory storing results")

    args = parser.parse_args()


    print ("network = ",args.network)
    print ("mask = ",args.mask)
    print ("source = ",args.source)
    print ("sink = ",args.sink)
    print ("gamma = ",args.gamma)
    print ("weight_discrepancy = ",args.wd)
    print ("weight_regularization = ",args.wr)
    print ("directory = ",args.dir)
    print ("curropted_as_initial_guess = ",args.ini)
    print ("scaling_size = ",args.scale)
    
    # set weights (default 1.0, 1.0, 0.0)
    weights = np.array([args.wd, args.wp, args.wr])

    if (not os.path.exists(args.dir)):
        os.mkdir(args.dir)

    out_dir = os.path.join(args.dir,'runs')
    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)

    
        
    corrupt_and_reconstruct(args.source,
                            args.sink,
                            args.network,
                            args.mask,
                            args.scale,
                            args.fem,
                            args.gamma,
                            weights,
                            args.ini,
                            args.conf,
                            args.t2i,
                            args.dir,
                            out_dir)
    print ("network = ",args.network)
    print ("mask = ",args.mask)
    print ("source = ",args.source)
    print ("sink = ",args.sink)
    print ("fem= ",args.fem)
    print ("gamma = ",args.gamma)
    print ("weight_discrepancy = ",args.wd)
    print ("weight_regularization = ",args.wr)
    print ("directory = ",args.dir)
    print ("curropted_as_initial_guess = ",args.ini)
    print ("scaling_size = ",args.scale)

    