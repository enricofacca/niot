"""
This script contains the tools to
corrupt and reconstruct a network image

Use 
```
python corrupt_and_reconstruct.py --help 
```
for instructions on how to use it.
"""
import argparse
import os
from copy import deepcopy as cp
import numpy as np
import time 
from niot import NiotSolver # reconstruction solver
from niot import optimal_transport as ot # optimal transport
from niot import image2dat as i2d # processing images
from niot.conductivity2image import HeatMap 

from ufl import *
from firedrake import *
from firedrake import norm
from firedrake import Function
from firedrake import interpolate
# for writing to file
from firedrake import VTKFile
import firedrake as fire
#from memory_profiler import profile

from firedrake import COMM_WORLD,COMM_SELF
from firedrake.petsc import PETSc

from scipy.ndimage import zoom,gaussian_filter

def my_rm(filename):
    if os.path.exists(filename):
        try:
            os.remove(filename)
        except:
            print(f'Error removing {filename}. Skipping')
            pass

def my_mkdir(directory):
    if (not os.path.exists(directory)):
        try:
            os.mkdir(directory)
        except: 
            print('error creating directory')
            pass

def my_mv(origin, destination):
    if os.path.exists(origin):
        try:
            os.rename(origin, destination)
        except:
            print(f'Error moving {origin} to {destination}. Skipping')
            pass

def labels(nref,fem,
           gamma,wd,wr,
           network_file,
           corrupted_as_initial_guess,
           confidence,
           tdens2image, 
           method):
    # take filename and remove extension
    label_network = os.path.basename(network_file)
    # remove extension
    label_network = os.path.splitext(label_network)[0]
    label= [
        f'nref{nref}',
        f'fem{fem}',
        f'gamma{gamma:.1e}',
        f'wd{wd:.1e}',
        f'wr{wr:.1e}',
        f'net{label_network}',
        f'ini{corrupted_as_initial_guess:.1e}',
        f'conf{confidence}']
    if tdens2image['type'] == 'identity':
        label.append(f'mu2iidentity')
    elif tdens2image['type'] == 'heat':
        label.append(f"mu2iheat{tdens2image['sigma']:.1e}")
    elif tdens2image['type'] == 'pm':
        label.append(f"mu2ipm{tdens2image['sigma']:.1e}")
    else:
        raise ValueError(f'Unknown tdens2image {tdens2image}')
    label.append(f"scaling{tdens2image['scaling']:.1e}")  
    if method is not None:
        if method == 'tdens_mirror_descent_explicit':
            short_method = 'te'
        elif method == 'tdens_mirror_descent_semi_implicit':
            short_method = 'tsi'
        elif method == 'gfvar_gradient_descent_explicit':
            short_method = 'ge'
        elif method == 'gfvar_gradient_descent_semi_implicit':
            short_method = 'gsi'
        elif method == 'tdens_logarithmic_barrier':
            short_method = 'tlb'
        else:
            raise ValueError(f'Unknown method {method}')
    label.append(f'method{short_method}')
    return label



def corrupt_and_reconstruct(np_source,
                            np_sink,
                            np_network,
                            np_mask,
                            fem="DG0DG0",
                            gamma=0.5, 
                            wd=1e-2,
                            wr=1e-4, 
                            corrupted_as_initial_guess=0,
                            confidence='ONE',
                            tdens2image={
                                'type':'identity',
                                'scaling':1e0
                                },
                            method='tdens_mirror_descent_explicit'  ,
                            directory='out/',
                            labels_problem=['unnamed'],
                            comm=COMM_SELF,
                            n_ensemble=0,
                            verbose=0,
                            ):
    """
    Given a network image, a mask image, a source image and a sink image
    the function corrupts the network image and reconstructs it using the
    optimal transport solver.

    """
    label = '_'.join(labels_problem)
    
    # Corrupt the network image
    np_corrupted = np_network * (1-np_mask)
    
    
    # Set the confidence
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

    # Set mesh used according to discretization space
    if fem == 'CR1DG0':
        mesh_type = 'simplicial'
    elif fem == 'DG0DG0':
        mesh_type = 'cartesian'

    # Create mesh
    h = 1.0 / np_corrupted.shape[1]
    mesh = i2d.build_mesh_from_numpy(np_corrupted, mesh_type=mesh_type, lengths=[1.0,np_corrupted.shape[0]*h], comm=comm)
    
    # Convert numpy arrays to firedrake functions    
    source = i2d.numpy2firedrake(mesh, np_source, name="source")
    sink = i2d.numpy2firedrake(mesh, np_sink, name="sink")
    network = i2d.numpy2firedrake(mesh, np_network, name="network")
    corrupted = i2d.numpy2firedrake(mesh, np_corrupted, name="corrupted")
    mask = i2d.numpy2firedrake(mesh, np_mask, name="mask")
    confidence_w = i2d.numpy2firedrake(mesh, np_confidence, name="confidence")
    CG1 = fire.FunctionSpace(mesh ,'CG',1)
    mask_for_contour = Function(CG1)
    mask_for_contour.interpolate(mask).rename("mask_countour","mask")
    network_for_contour = Function(CG1)
    network_for_contour.interpolate(network).rename("network_countour","network")
    corrupted_for_contour = Function(CG1)
    corrupted_for_contour.interpolate(corrupted).rename("corrupted_countour","corrupted")
    confidence_for_contour = Function(CG1)
    confidence_for_contour.interpolate(confidence_w).rename("confidence_countour","confidence")
    
    # Define the branched transport problem
    ot.balance(source, sink)
    btp = ot.BranchedTransportProblem(source, sink, gamma=gamma)

    
    # save problem inputs 
    overwrite = True
    
    filename = os.path.join(directory, f'{labels_problem[0]}_{labels_problem[5]}_network_mask.pvd')
    if (not os.path.exists(filename) or overwrite):
        try:
            #PETSc.Sys.Print(f"{n_ensemble=} open {filename=}", comm=comm)   
            out_file = VTKFile(filename,comm=comm,mode='w')
            out_file.write(corrupted,
            corrupted_for_contour,
            network,
            network_for_contour,
            mask,
            mask_for_contour,
            )
        except:
            PETSc.Sys.Print(f'Error writing {filename}. Skipping',comm=comm)
            pass   
    

    # Save confidence inputs
    filename = os.path.join(directory, f'{labels_problem[0]}_{labels_problem[7]}.pvd')
    if (not os.path.exists(filename) or overwrite):
        try:
            #PETSc.Sys.Print(f"{n_ensemble=} open {filename=}", comm=comm)
            out_file = VTKFile(filename,comm=comm,mode='w')
            out_file.write(
                confidence_w,
                confidence_for_contour,
                )#,filename)
        except:
            PETSc.Sys.Print(f'Error writing {filename}. Skipping',comm=comm)
            pass
        
    # Save source and sink inputs
    filename = os.path.join(directory, f'{labels_problem[0]}_btp.pvd')
    if (not os.path.exists(filename) or overwrite):
        source_for_contour = Function(CG1)
        source_for_contour.interpolate(source).rename("source_countour","source")
        sink_for_contour = Function(CG1)
        sink_for_contour.interpolate(sink).rename("sink_countour","sink")
        #PETSc.Sys.Print(f"{n_ensemble=} open {filename=}", comm=comm)
        try:
            out_file = VTKFile(filename,comm=comm,mode='w')
            out_file.write(
            source,
            sink,
            source_for_contour,
            sink_for_contour,
            )
        except:
            pass

    # Setup reconstruction solver
    niot_solver = NiotSolver(btp, corrupted,  confidence_w, 
                             spaces = fem,
                             cell2face = 'harmonic_mean',
                             setup=False)
    
    
    # Set initial guess
    if abs(corrupted_as_initial_guess) < 1e-16:
        niot_solver.sol.sub(1).assign(1.0)
    else:
        heat_map = HeatMap(space=niot_solver.fems.tdens_space, scaling=1.0, sigma = 1.0/corrupted_as_initial_guess)
        with corrupted.dat.vec_ro as v:
            max_corrupted = max(v.max()[1],niot_solver.ctrl_get("min_tdens"))
        img0 = heat_map(corrupted+1e-5*max_corrupted)
        niot_solver.sol.sub(1).assign(img0 / tdens2image['scaling'] )
    
        
    # Setup the solver's parameters

    # inpainting
    niot_solver.ctrl_set('discrepancy_weight', wd)
    niot_solver.ctrl_set('regularization_weight', wr)
    niot_solver.ctrl_set(['tdens2image'], tdens2image)

    # optimization
    niot_solver.ctrl_set('optimization_tol', 1e-5)
    niot_solver.ctrl_set('constraint_tol', 1e-5)
    niot_solver.ctrl_set('max_iter', 5000)
    niot_solver.ctrl_set('max_restart', 4)
    niot_solver.ctrl_set('verbose', 0)  
    
    
    niot_solver.ctrl_set('log_verbose', 2) 
    log_file = os.path.join(directory, f'{label}.log')
    niot_solver.ctrl_set('log_file', log_file)
    # remove if exists
    if niot_solver.mesh.comm.rank == 0:
        my_rm(log_file)
        

    # time discretization
    if method is None:
        method = 'tdens_mirror_descent_explicit'
    niot_solver.ctrl_set(['dmk','type'], method)
    if 'tdens' in method:
        if method == 'tdens_logarithmic_barrier':
            pass
        else:
            niot_solver.ctrl_set(['dmk',method,'gradient_scaling'], 'dmk')
        

    # time step
    deltat_control = {
        'type': 'adaptive2',
        'lower_bound': 1e-13,
        'upper_bound': 5e-2,
        'expansion': 1.1,
        'contraction': 0.5,
    }
    niot_solver.ctrl_set(['dmk',method,'deltat'], deltat_control)
    
    
    
    # solve one PDE with one iteration to get the initial guess
    max_iter = niot_solver.ctrl_get('max_iter')
    niot_solver.ctrl_set('max_iter', 0)
    ierr = niot_solver.solve()

    filename = os.path.join(directory, f'{label}_discrepancy.pvd')
    if (not os.path.exists(filename) or overwrite):
        try:
            #PETSc.Sys.Print(f"{n_ensemble=} open {filename=}", comm=comm)
            out_file = VTKFile(filename,comm=comm,mode='w')
            out_file.write(niot_solver.confidence,niot_solver.img_observed)
        except:
            PETSc.Sys.Print(f'Error writing {filename}. Skipping',comm=comm)
            pass
    
    sol0 = cp(niot_solver.sol)
    pot0, tdens0 = sol0.subfunctions
    pot0.rename('pot_0','pot_0')
    tdens0.rename('tdens_0','tdens_0')
    niot_solver.ctrl_set('max_iter', max_iter)

    
    start = time.process_time()
    ierr = niot_solver.solve()
    cpu_rec = time.process_time() - start 
    filename = os.path.join(directory, f'{label}.cpu')
    with open(filename, 'w') as fd:
        fd.writelines(str(cpu_rec))

    
    # save solution
    pot, tdens, vel = niot_solver.get_otp_solution(niot_solver.sol)
    
    reconstruction = Function(niot_solver.fems.tdens_space)
    reconstruction.interpolate(niot_solver.tdens2image(tdens) )
    reconstruction.rename('reconstruction','Reconstruction')

        
    reconstruction_for_contour = Function(CG1)
    reconstruction_for_contour.interpolate(reconstruction).rename("reconstruction_countour","reconstruction_countour")

    tdens_for_contour = Function(CG1)
    tdens_for_contour.interpolate(tdens).rename("tdens_countour","tdens_countour")
    

    filename = os.path.join(directory, f'{label}.pvd')
    out_file = VTKFile(filename,comm=comm,mode='w')
    out_file.write(pot, tdens, 
       tdens0, pot0, 
       reconstruction,
        reconstruction_for_contour,
       tdens_for_contour)
    PETSc.Sys.Print(f"{ierr=}. {n_ensemble=} Saved solution to "+filename, comm=comm)
    
    return ierr
    
    
    

def load_input(example, nref, mask, network_file, comm=COMM_WORLD):
    # btp inputs
    img_sources = f'{example}/source.png'
    img_sinks = f'{example}/sink.png'
    
    np_source = i2d.image2numpy(img_sources,normalize=True,invert=True)
    np_sink = i2d.image2numpy(img_sinks,normalize=True,invert=True)
    
    # taking just the support of the sources and sinks
    np_source[np.where(np_source>0.0)] = 1.0
    np_sink[np.where(np_sink>0.0)] = 1.0

    # balancing the mass
    nx, ny = np_source.shape
    hx, hy = 1.0/nx, 1.0/ny
    mass_source = np.sum(np_source)*hx*hy
    mass_sink = np.sum(np_sink)*hx*hy
    if abs(mass_source-mass_sink)>1e-16:
        np_sink *= mass_source/mass_sink 

    
    # load image or numpy array
    try:
        img_networks = f'{example}/{network_file}'
        np_network = i2d.image2numpy(img_networks,normalize=True,invert=True)
        #PETSc.Sys.Print(f'using {img_networks}',comm=comm)
    except: 
        np_network = np.load(f'{example}/{network_file}')
        #if np_network.ndim == 2 and i2d.convention_2d_flipud:
        #np_network = np.flipud(np_network)
    np_mask = i2d.image2numpy(f'{example}/{mask}',normalize=True,invert=True)

    if nref != 0:
        np_source = zoom(np_source, 2**nref, order=0, mode='nearest')
        np_sink = zoom(np_sink, 2**nref, order=0, mode='nearest')
        np_network = zoom(np_network, 2**nref, order=0, mode='nearest')
        np_mask = zoom(np_mask, 2**nref, order=0, mode='nearest')


    return np_source, np_sink, np_network, np_mask


    
def run_experiment(example, mask, nref,fem,gamma,wd,wr,network_file,ini,conf,tdens2image, method, comm=COMM_WORLD, n_ensemble=1,out_directory='results'):
    """
    Function running the corrupt and reconstruct experiment
    given a list of parameters describing the problem
    and the method to solve it.
    """

    # set a list of labels 
    labels_problem = labels(nref,fem,gamma,wd,wr,
                network_file,   
                ini,
                conf,
                tdens2image,
                method)
    

    # load input    
    np_source, np_sink, np_network, np_mask = load_input(example, nref, mask, network_file, comm=comm)
    

    # create directory
    mask_name = os.path.splitext(mask)[0]
    if not os.path.exists(f'{example}/{mask_name}/'):
        try:
            os.makedirs(f'{example}/{mask_name}/')
        except:
            pass

    # check if the directory exists
    if not os.path.exists(f'{out_directory}/{example}/{mask_name}/'):
        try:
            os.mkdir(f'{out_directory}/')
        except:
            pass # directory already exists
    
    # run the experiment
    ierr = corrupt_and_reconstruct(np_source,
                                   np_sink,
                                   np_network,
                                   np_mask, 
                            fem=fem,
                            gamma=gamma,
                            wd=wd,wr=wr,
                            corrupted_as_initial_guess=ini,
                            confidence=conf,
                            tdens2image=tdens2image,
                            method=method,
                            directory=f'results/{example}/{mask_name}/',
                            labels_problem=labels_problem,
                            comm=comm,
                            n_ensemble=n_ensemble)
    return ierr 



if (__name__ == '__main__'):
    # 
    # Example of usage
    #

    parser = argparse.ArgumentParser(description='Corrupt networks with masks and reconstruct via branched transport')
    parser.add_argument('--example', type=str, help="path to the example directory")
    parser.add_argument('--mask', type=str, help="mask name in the example directory")
    parser.add_argument('--nref', type=int, default=0, help="scaling (between 0 and 1) of the mask image")
    parser.add_argument('--fem', type=str, default='DG0DG0', help='CR1DG0,DG0DG0')
    parser.add_argument('--gamma', type=float, default=0.5, help="branch exponent")
    parser.add_argument('--wd', type=float, default=1.0, help="weight discrepancy")
    parser.add_argument('--wp', type=float, default=1.0, help="weight penalty")
    parser.add_argument('--wr', type=float, default=0.0, help="weight regularation")
    parser.add_argument('--network', type=str, help="network name in the example directory")
    parser.add_argument('--ini', type=int, default=0, help="0/1 use corrupted image as initial guess")
    parser.add_argument('--conf', type=str, default='MASK', help="MASK(=opposite to mask),CORRUPTED(=gaussian filter of corrupted)")
    parser.add_argument('--t2i', type=str, default='identity', help="identity,heat")
    parser.add_argument('--t2i_sigma', type=float, default=1e-3, help="sigma for heat and pm")
    parser.add_argument('--t2i_m', type=float, default=2.0, help="pm exponent m")
    parser.add_argument('--scale', type=float, default=1.0, help="scaling of the corrupted image")
    parser.add_argument('--dir', type=str, default='niot_recostruction', help="directory storing results")

    args = parser.parse_args()

    if args.tdi == 'identity':
        t2i={'type': args.tdi, 'scaling':args.scale}
    elif args.tdi == 'heat':
        t2i={'type': args.tdi, 'sigma':args.t2i_sigma, 'scaling':args.scale}
    else:
        t2i={'type': args.tdi, 'sigma':args.t2i_sigma, 'exponent_m':args.t2i_m, 'scaling':args.scale}


    ierr = run_experiment(example=args.example,
        mask=args.mask,
        nref=args.nref,
        fem=args.fem,
        gamma=args.gamma,
        wd=args.wd,
        wr=args.wr,
        network_file=args.network,
        ini=args.ini,
        conf=args.conf,
        tdens2image=t2i,
        method='gfvar_gradient_descent_semi_explicit')
    
    label = '_'.join(labels(
        args.nref,args.fem,args.gamma,args.wd,args.wr,
                args.network,   
                args.ini,
                args.conf,
                t2i,
                args.scale,
                'gfvar_gradient_descent_semi_explicit'))
    
    print(f'{ierr=}. {label=}')


    
# Example of callback
def callback(solver):
    if solver.iteration % 100 != 0:
        return
        
    PETSc.Sys.Print(f'{n_ensemble=} {solver.iteration=}',comm=solver.comm)
    filename = os.path.join(directory,'runs',f'{label}_{solver.iteration:03d}.pvd')
    #PETSc.Sys.Print(f"{ierr=}. {n_ensemble=} {comm.size} Open "+filename, comm=comm)
    out_file = VTKFile(filename,comm=comm,mode='w')
    pot, tdens, vel = solver.get_otp_solution(solver.sol)
    out_file.write(pot, tdens, 
                    solver.gradient_discrepancy, 
                    solver.gradient_penalization
    )
    PETSc.Sys.Print(f"{ierr=}. {n_ensemble=} Saved solution to "+filename, comm=comm)
    
def range_on_sink(solver):
    _, tdens = solver.sol.subfunctions
    solver.tdens_h.interpolate(tdens*conditional(solver.btp.sink>0.0,1.0,1e30),annotate=False)
    min_on_sink = solver.tdens_h.dat.data_ro.min()

    solver.tdens_h.interpolate(tdens*conditional(solver.btp.sink>0.0,1.0,0),annotate=False)
    max_on_sink = solver.tdens_h.dat.data_ro.max()

    solver.print_info(f'{solver.iteration=}. {min_on_sink:.1e}<=TDENS ON SINK<={max_on_sink:.1e}',
                    priority=2,
                    where=['stdout','log'],
                    color='yellow')
