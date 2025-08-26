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
import numpy as np
from niot import conductivity2image
from numpy.random import rand

import copy as cp

import gc

from ufl import *
from firedrake import *
from firedrake import norm
from firedrake import Function
from firedrake import interpolate
# for writing to file
from firedrake import File
import firedrake as fire
#from memory_profiler import profile
import firedrake.adjoint as fire_adj
from firedrake.petsc import PETSc


import localthickness as lt
from skimage.morphology import skeletonize
from connected_components_tof import connected_components, main_network_equal_one
from niot import utilities as utilities
from niot import image2dat as i2d

import nibabel


def save_as_nifti(function, affine, filename):
    function_np = i2d.firedrake2numpy(function)
    nibabel.save(nibabel.Nifti1Image(function_np, affine), filename)
    function_np = None
    gc.collect()
    return

def save_np_as_nifti(function_np, affine, filename):
    nibabel.save(nibabel.Nifti1Image(function_np, affine), filename)
    function_np = None
    gc.collect()
    return

def indices_surronding_box(array):
    """
        Given a nd-array, find the indices that contains all 
        nonzeros values.
        """
    indices = np.where(array > 0)
    min_indices = np.min(indices, axis=1)
    max_indices = np.max(indices, axis=1)
    return min_indices, max_indices


tape = fire_adj.get_working_tape()

solver_parameters={
                'snes_type': 'newtonls',
                'snes_rtol': 1e-16,
                'snes_atol': 1e-16,
                'snes_stol': 1e-16,
                'snes_max_it': 100,
                'snes_linesearch_type':'bt',
                'ksp_type': 'gmres',
                'ksp_rtol': 1e-10,
                'ksp_atol': 1e-10,
                'ksp_max_it': 500,
                'pc_type': 'hypre',
                'snes_monitor': None,
                #'snes_linesearch_monitor': None,
                'ksp_monitor': None,
                }


def skeletonthickness_to_tubular(cond, cond_zero, exponent_p, dt0, nsteps=3, verbose=False):
    "Convert the skeleton thickness to a tubular image"
    
    space = cond.function_space()
    mesh = cond.function_space().mesh()

    PETSc.Sys.Print(f"dim {mesh.geometric_dimension()=}")
    d = mesh.geometric_dimension()-1
    if exponent_p < d:
        raise ValueError('p<d')
    exponent_m = (2 + exponent_p - d ) / (exponent_p - d)
    
    
    scaling = 1.0 
    dim = mesh.geometric_dimension()-1
    Bar = conductivity2image.Barenblatt(exponent_m,dim)
    B = Bar.B
    alpha = Bar.alpha
    beta = Bar.beta
    K_md = Bar.K_md

    

    # find the time to get 
    # M = M_0 * (r(\sigma))**p 
    # sigma = (cond_zero**(-1/exponent_p) * K_md ** (-1/2) * B **(1/2))**(1/beta)
    sigma = Bar.sigma(cond_zero,exponent_p)

    #cond.vector()[:] = 100*(np.ones(cond.vector().local_size())+rand(cond.vector().local_size()))
    #test_adjoint(cond, exponent_m, sigma=sigma, scaling=1.0, nsteps=1000)

    #height = conductivity2image.Barenblatt().height(exponent_m,dim,sigma,M_max)
    if verbose:
        PETSc.Sys.Print(
        f'p={exponent_p:.1e} d={d} m={exponent_m}'
        + f'B={B:.1e} alpha={alpha:.1e} beta={beta:.1e} K_md={K_md:.1e} sigma={sigma:.1e} f={sigma**alpha:.1e}"')
    

    with cond.dat.vec as cond_vec:
        PETSc.Sys.Print(utilities.msg_bounds(cond_vec, 'mu'))

    pm_map = conductivity2image.PorousMediaMap(
        space,
        sigma=sigma, 
        exponent_m=exponent_m, 
        scaling=scaling, 
        nsteps=nsteps,
        dt0=dt0,
        solver_parameters=solver_parameters)
    image = pm_map(cond)
    name = f'img_pm'
    image.rename(name)
    with image.dat.vec as image_vec:
        PETSc.Sys.Print(utilities.msg_bounds(image_vec, "IMG"))
    
    return image

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser()
    parser.add_argument('--tof', type=str, default='TOF.nii.gz', help='path to the TOF image')
    parser.add_argument('--t', type=float, default=200, help='threshold for TOF')
    parser.add_argument('--blur', type=float, default=1.5, help='blur for TOF')
    parser.add_argument('--mu0', type=float, default=8/np.pi*3e-3, help='conductivity at which the thickness is zero')
    parser.add_argument('--dt0', type=float, default=1e-2, help='initial time step for porous media map')
    parser.add_argument('--nsteps', type=int, default=3, help='number of time steps for porous media map')
    parser.add_argument('--out', type=str, default="./check_pm", help='output directory')
    args = parser.parse_args()

    exponent_p = 4.0

    # create output directory
    import os
    os.makedirs(args.out, exist_ok=True)




    # load tof data and get basic info
    tof_nii = nibabel.load(args.tof)
    tof_np = tof_nii.get_fdata()
    dimensions = tof_np.shape
    hx, hy, hz = tof_nii.header['pixdim'][1:4]
    lengths = np.array([float(dimensions[0]*hx), 
                        float(dimensions[1]*hy), 
                        float(dimensions[2]*hz)])
    
    PETSc.Sys.Print(f"{dimensions=} {lengths=}")
    
    # set common label
    threshold = args.t
    blur = args.blur

    label = f"t{threshold:.2e}"
    if blur > 0:
        label += f"_blur{blur:.2e}"
    print(f"Label: {label}")

    # blur 
    if blur > 0:
        from scipy.ndimage import gaussian_filter
        # get pixel size
        tof_np = gaussian_filter(tof_np, sigma=blur*hx)


    # separe connected components
    labels_np, nlabels = connected_components(tof_np, threshold)
    PETSc.Sys.Print(f"Found {nlabels=} with tof>={threshold:.2e}")
    labels_np = main_network_equal_one(labels_np, nlabels, tof_np)

    
    # save as nifti 
    main_network = np.zeros_like(labels_np, dtype=np.uint8)
    main_network[labels_np == 1] = 1
    
    # restrict
    min_indices, max_indices = indices_surronding_box(main_network)
    main_network_np = main_network[
        min_indices[0]:max_indices[0]+1,
        min_indices[1]:max_indices[1]+1,
        min_indices[2]:max_indices[2]+1
    ]
    dimensions = main_network_np.shape
    lengths = [
        hx * dimensions[0],
        hy * dimensions[1],
        hz * dimensions[2]
    ]

    
    offsets = [hx*min_indices[0],
               hy*min_indices[1],
               hz*min_indices[2]]
    PETSc.Sys.Print(f"{dimensions=} {lengths=}")
    
    new_affine = cp.copy(tof_nii.affine)
    new_affine[0,0] = hx
    new_affine[1,1] = hy
    new_affine[2,2] = hz

    new_affine[0,3] += offsets[0]
    new_affine[1,3] += offsets[1]
    new_affine[2,3] += offsets[2]
    
    new_header = cp.copy(tof_nii.header)
    new_header['pixdim'][1:4] = main_network.shape

    filename = f"{args.out}/main_network_{label}.nii.gz"
    save_np_as_nifti(main_network, new_affine, filename) 

    
    

    # get the skeleton of main network
    skeleton_np = skeletonize(main_network)
    skeleton_np = skeleton_np.astype(np.uint8)
    filename = f"{args.out}/skeleton_{label}.nii.gz"
    save_np_as_nifti(skeleton_np, new_affine, filename) 
    

    # compute local thickness of the main network
    thickness_np = lt.local_thickness(main_network)
    # scale by thickness 
    thickness_np *= hx
    filename = f"{args.out}/thichness_{label}.nii.gz"
    save_np_as_nifti(thickness_np, new_affine, filename) 
    

    
    # define the mesh
    PETSc.Sys.Print(f"Mesh", end="")
    cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths)

    dim = cartesian_mesh.geometric_dimension()
    exponent_p = 4.0
    skeleton_radius_np = skeleton_np * (thickness_np/2)
    # The formula is 
    #
    # mu = 8 nu / pi r^4
    # 
    # The scaling by hx**(dim-1) is to get a Dirac-like distribution
    #
    tdens_np = args.mu0 * skeleton_radius_np**exponent_p / hx**(dim-1)
    filename = f"{args.out}/tdens_{label}_c{args.mu0:.2e}.nii.gz"
    save_np_as_nifti(tdens_np, new_affine, filename) 
    


    # move to fire
    tdens = i2d.numpy2firedrake(cartesian_mesh, 
                                             tdens_np, 
                                             name="tdens")

    
    # map tdens into a tubular image
    tubular = skeletonthickness_to_tubular(
            tdens,         
            exponent_p=exponent_p,
            cond_zero=args.mu0,
            dt0=args.dt0,
            nsteps=args.nsteps,
            verbose=True)
    with tubular.dat.vec as tubular_vec:
        PETSc.Sys.Print(utilities.msg_bounds(tubular_vec, 'IMG'))
    
    tubular_np = i2d.firedrake2numpy(tubular)
    filename = f"{args.out}/tubular_{label}_c{args.mu0:.2e}.nii.gz"
    save_np_as_nifti(tubular_np, new_affine, filename)

    tof_np = tof_np[
        min_indices[0]:max_indices[0]+1,
        min_indices[1]:max_indices[1]+1,
        min_indices[2]:max_indices[2]+1
    ]
    tof_np = np.ascontiguousarray(tof_np)

    
    data = [tof_np, skeleton_np, thickness_np, tdens_np, tubular_np]
    names = ["tof", "skeleton", "thickness", "tdens", "tubular"]

    offset = tof_nii.affine[:3, 3]
    lx, ly, lz = dimensions[0]*hx, dimensions[1]*hy, dimensions[2]*hz
    i2d.numpy2vtr(data, [lx, ly, lz], f"{args.out}/results.vtr", names=names, offset=offset)
    

