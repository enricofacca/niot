"""
e script is used to corrupt and reconstruct a network image.
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

def labels(nref,fem,
           gamma,wd,wr,
           corrupted_as_initial_guess,
           confidence,
           tdens2image,
           tdens2image_scaling, 
           method):
    label= [
        f'nref{nref}',
        f'fem{fem}',
        f'gamma{gamma:.1e}',
        f'wd{wd:.1e}',
        f'wr{wr:.1e}',
        f'ini{corrupted_as_initial_guess:d}',
        f'conf{confidence}']
    if tdens2image['type'] == 'identity':
        label.append(f'mu2iidentity')
    elif tdens2image['type'] == 'heat':
        label.append(f"mu2iheat{tdens2image['sigma']:.1e}")
    elif tdens2image['type'] == 'pm':
        label.append(f"mu2ipm{tdens2image['sigma']:.1e}")
    else:
        raise ValueError(f'Unknown tdens2image {tdens2image}')
    label.append(f'scaling{tdens2image_scaling:.1e}')  
    if method is not None:
        if method == 'tdens_mirror_descent_explicit':
            short_method = 'te'
        elif method == 'tdens_mirror_descent_semi_implicit':
            short_method = 'tsi'
        elif method == 'gfvar_gradient_descent_explicit':
            short_method = 'ge'
        elif method == 'gfvar_gradient_descent_semi_implicit':
            short_method = 'gsi'
        else:
            raise ValueError(f'Unknown method {method}')
    label.append(f'method{short_method}')
    return label


#@profile
def corrupt_and_reconstruct(img_source,
                            img_sink,
                            img_network,
                            img_mask, 
                            nref=0,
                            fem="DG0DG0",
                            gamma=0.5, 
                            wd=1e-2,
                            wr=1e-4, 
                            corrupted_as_initial_guess=0,
                            confidence='ONE',
                            tdens2image='identity',
                            tdens2image_scaling=1e0,
                            method='tdens_mirror_descent_explicit'  ,
                            directory='out/'
                            ):
    
    # label= [
    #     f'nref{nref}',
    #     f'fem{fem}',
    #          f'gamma{gamma:.1e}',
    #          f'wd{weights[0]:.1e}',
    #          f'wr{weights[2]:.1e}',
    #          f'ini{corrupted_as_initial_guess:d}',
    #          f'conf{confidence}']
    # if tdens2image == 'identity':
    #     label.append(f'mu2i{tdens2image}')
    # elif tdens2image == 'heat':
    #     label.append(f'mu2i{tdens2image}{sigma_smoothing:.1e}')
    # elif tdens2image == 'pm':
    #     label.append(f'mu2i{tdens2image}{sigma_smoothing:.1e}')
    # else:
    #     raise ValueError(f'Unknown tdens2image {tdens2image}')
    # label.append(f'scaling{tdens2image_scaling:.1e}')  
    # if method is not None:
    #     if method == 'tdens_mirror_descent_explicit':
    #         short_method = 'te'
    #     elif method == 'tdens_mirror_descent_semi_implicit':
    #         short_method = 'tsi'
    #     elif method == 'gfvar_gradient_descent_explicit':
    #         short_method = 'ge'
    #     elif method == 'gfvar_gradient_descent_semi_implicit':
    #         short_method = 'gsi'
    #     else:
    #         raise ValueError(f'Unknown method {method}')
    #     label.append(f'method{short_method}')
    labels_problem = labels(nref,fem,gamma,wd,wr,
              corrupted_as_initial_guess,
                confidence,
                tdens2image,
                tdens2image_scaling,
                method)
    
    label = '_'.join(labels_problem)
    print('Problem label: '+label)
    


    # convert images to numpy matrices
    # 1 is the white background
    np_source = i2d.image2numpy(img_source,normalize=True,invert=True,factor=2**nref)
    np_sink = i2d.image2numpy(img_sink,normalize=True,invert=True,factor=2**nref)
    np_network = i2d.image2numpy(img_network,normalize=True,invert=True,factor=2**nref)
    np_mask = i2d.image2numpy(img_mask,normalize=True,invert=True,factor=2**nref)

    # save image to file
    # get the filename with no directory
    image_name = img_source.split('/')[-1]
    i2d.numpy2image(np_source,f'{directory}/nref{nref}_{image_name}')
    image_name = img_sink.split('/')[-1]
    i2d.numpy2image(np_sink,f'{directory}/nref{nref}_{image_name}')
    image_name = img_mask.split('/')[-1]
    i2d.numpy2image(np_mask,f'{directory}/nref{nref}_{image_name}')
    image_name = img_network.split('/')[-1]
    i2d.numpy2image(np_network,f'{directory}/nref{nref}_{image_name}')
   


    # taking just the support of the sources and sinks
    np_source[np.where(np_source>0.0)] = 1.0
    np_sink[np.where(np_sink>0.0)] = 1.0


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
        np_confidence = np.ones(np_mask.shape)
    else:
        raise ValueError(f'Unknown confidence {confidence}')


    # Init. solver for a given reconstruction problem
    #niot_solver = NiotSolver(fem, np_corrupted, DG0_cell2face = 'harmonic_mean')
    if fem == 'CR1DG0':
        mesh_type = 'simplicial'
    elif fem == 'DG0DG0':
        mesh_type = 'cartesian'

    mesh = i2d.build_mesh_from_numpy(np_corrupted, mesh_type=mesh_type)


    # save inputs    
    source = i2d.numpy2firedrake(mesh, np_source, name="source")
    sink = i2d.numpy2firedrake(mesh, np_sink, name="sink")
    network = i2d.numpy2firedrake(mesh, np_network, name="network")
    corrupted = i2d.numpy2firedrake(mesh, np_corrupted, name="corrupted")
    mask = i2d.numpy2firedrake(mesh, np_mask, name="mask")
    confidence = i2d.numpy2firedrake(mesh, np_confidence, name="confidence")
    CG1 = fire.FunctionSpace(mesh ,'CG',1)
    mask_for_contour = Function(CG1)
    mask_for_contour.interpolate(mask).rename("mask_countour","mask")
    network_for_contour = Function(CG1)
    network_for_contour.interpolate(network).rename("network_countour","network")
    corrupted_for_contour = Function(CG1)
    corrupted_for_contour.interpolate(corrupted).rename("corrupted_countour","corrupted")
    confidence_for_contour = Function(CG1)
    confidence_for_contour.interpolate(confidence).rename("confidence_countour","confidence")

    #
    # solve the problem
    #
    ot.balance(source, sink)
    btp = ot.BranchedTransportProblem(source, sink, gamma=gamma)

    
    niot_solver = NiotSolver(btp, corrupted,  confidence, 
                             spaces = fem,
                             cell2face = 'harmonic_mean',
                             setup=False)
    
    if corrupted_as_initial_guess == 1:
        niot_solver.sol.sub(1).assign(corrupted+1e-5)


    # inpainting
    niot_solver.ctrl_set('discrepancy_weight', wd)
    niot_solver.ctrl_set('regularization_weight', wr)
    map_ctrl={**tdens2image, **{'scaling': tdens2image_scaling}}
    niot_solver.ctrl_set(['tdens2image'], map_ctrl)

    # optimization
    niot_solver.ctrl_set('optimization_tol', 1e-3)
    niot_solver.ctrl_set('constrain_tol', 1e-8)
    niot_solver.ctrl_set('max_iter', 5000)
    niot_solver.ctrl_set('max_restart', 3)
    niot_solver.ctrl_set('verbose', 0)  
    
    niot_solver.ctrl_set('log_verbose', 2) 
    log_file = os.path.join(directory, f'{label}.log')
    niot_solver.ctrl_set('log_file', log_file)
    # remove if exists
    if niot_solver.mesh.comm.rank == 0:
        if os.path.exists(log_file):
            os.remove(log_file)
    

    # time discretization
    if method is None:
        method = 'tdens_mirror_descent_explicit'
    print(f"Using {method}")
    niot_solver.ctrl_set(['dmk','type'], method)
    if 'tdens' in method:
        niot_solver.ctrl_set(['dmk',method,'gradient_scaling'], 'no')
        

    # time step
    deltat_control = {
        'type': 'adaptive2',
        'lower_bound': 1e-13,
        'upper_bound': 1e-2,
        'expansion': 1.02,
        'contraction': 0.5,
    }
    niot_solver.ctrl_set(['dmk',method,'deltat'], deltat_control)
    
    
    
    
    # solve the potential PDE
    max_iter = niot_solver.ctrl_get('max_iter')
    niot_solver.ctrl_set('max_iter', 1)
    ierr = niot_solver.solve()
    sol0 = cp(niot_solver.sol)

    niot_solver.ctrl_set('max_iter', max_iter)
    


    # solve the problem
    ierr = niot_solver.solve()
    
    # save solution
    pot, tdens, vel = niot_solver.get_otp_solution(niot_solver.sol)
    pot0, tdens0, vel0 = niot_solver.get_otp_solution(sol0)
    pot0.rename('pot_0','pot_0')
    tdens0.rename('tdens_0','tdens_0')
    vel0.rename('vel_0','vel_0')

    reconstruction = Function(niot_solver.fems.tdens_space)
    reconstruction.interpolate(niot_solver.tdens2image(tdens) )
    reconstruction.rename('reconstruction','Reconstruction')

    reconstruction_for_contour = Function(CG1)
    reconstruction_for_contour.interpolate(reconstruction).rename("reconstruction_countour","reconstruction_countour")

    tdens_for_contour = Function(CG1)
    tdens_for_contour.interpolate(tdens).rename("tdens_countour","tdens_countour")


    filename = os.path.join(directory, f'{label}.pvd')
    print(f"{ierr=} Saving solution to \n"+filename)
    utilities.save2pvd([
        pot, tdens, 
        #pot0, tdens0,
        reconstruction,
        reconstruction_for_contour,
        tdens_for_contour,
                        ],filename)
    
    filename = os.path.join(directory, f'{labels_problem[0]}_btp.pvd')
    if (not os.path.exists(filename)):
        print(filename)
        source_for_contour = Function(CG1)
        source_for_contour.interpolate(source).rename("source_countour","source")
        sink_for_contour = Function(CG1)
        sink_for_contour.interpolate(sink).rename("sink_countour","sink")
        utilities.save2pvd([
            niot_solver.btp.source,
            niot_solver.btp.sink,
            source_for_contour,
            sink_for_contour,
            ],filename)
        
    filename = os.path.join(directory, f'{labels_problem[0]}_network_mask.pvd')
    print(filename)
    if (not os.path.exists(filename)):
        print(filename)
        utilities.save2pvd([
            corrupted,
            corrupted_for_contour,
            network,
            network_for_contour,
            mask,
            mask_for_contour,
            ],filename)
        
    filename = os.path.join(directory, f'{labels_problem[0]}_confidence.pvd')
    if (not os.path.exists(filename)):
        print(filename)
        utilities.save2pvd([
            confidence,
            confidence_for_contour,
            ],filename)
    
    #mask_for_contour.rename('mask_for_contour')
    #filename = os.path.join(directory, f'simple_{label}.pvd')
    #print("Saving solution to "+filename)
    #utilities.save2pvd([
    #    tdens, network, mask_for_contour,
    #                    ],filename)

    return ierr
    
    
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
    parser.add_argument('--t2i', type=str, default='identity', help="identity,heat")
    parser.add_argument('--dir', type=str, default='niot_recostruction', help="directory storing results")

    args = parser.parse_args()


    print ("network = ",args.network)
    print ("mask = ",args.mask)
    print ("source = ",args.source)
    print ("sink = ",args.sink)
    print ("gamma = ",args.gamma)
    print ("dicrepancy_weight = ",args.wd)
    print ("regularization_weight = ",args.wr)
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
                            nref=args.scale,
                            fem=args.fem,
                            gamma=args.gamma,
                            wd=args.wd,
                            wr=args.wr,
                            corrupted_as_initial_guess=args,
                            confidence=args.conf,
                            tdens2image=args.t2i,
                            tdens2image_scaling=10,
                            method='tdens_mirror_descent_explicit'  ,
                            out_dir=out_dir)
    
    
    print ("network = ",args.network)
    print ("mask = ",args.mask)
    print ("source = ",args.source)
    print ("sink = ",args.sink)
    print ("fem= ",args.fem)
    print ("gamma = ",args.gamma)
    print ("dicrepancy_weight = ",args.wd)
    print ("regularization_weight = ",args.wr)
    print ("directory = ",args.dir)
    print ("curropted_as_initial_guess = ",args.ini)
    print ("scaling_size = ",args.scale)

    
