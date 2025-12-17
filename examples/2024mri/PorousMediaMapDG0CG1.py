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



from niot import utilities as utilities
from niot import image2dat as i2d

import nibabel
from time import time

def save_as_nifti(function, affine, filename):
    function_np = i2d.firedrake2numpy(function)
    nibabel.save(nibabel.Nifti1Image(function_np, affine), filename)
    function_np = None
    gc.collect()
    return

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


def skeletonthickness_to_tubular(skeleton_thickness, h, cond_zero, exponent_p, verbose=False, save=False):
    "Convert the skeleton thickness to a tubular image"
    
    space = skeleton_thickness.function_space()
    mesh = skeleton_thickness.function_space().mesh()

    PETSc.Sys.Print(f"dim {mesh.geometric_dimension()=}")
    d = mesh.geometric_dimension()-1
    if exponent_p < d:
        raise ValueError('p<d')
    exponent_m = (2 + exponent_p - d ) / (exponent_p - d)
    
    
    images = []
    scaling = 1.0 
    dim = mesh.geometric_dimension()-1
    Bar = conductivity2image.Barenblatt(exponent_m,dim)
    B = Bar.B# = conductivity2image.Barenblatt().B(exponent_m,dim)
    alpha = Bar.alpha#conductivity2image.Barenblatt().alpha(exponent_m,dim)
    beta = Bar.beta# conductivity2image.Barenblatt().beta(exponent_m,dim)
    K_md = Bar.K_md#conductivity2image.Barenblatt().K_md(exponent_m,dim)

    

    # find the time to get 
    # M = M_0 * (r(\sigma))**p 
    # sigma = (cond_zero**(-1/exponent_p) * K_md ** (-1/2) * B **(1/2))**(1/beta)
    sigma = Bar.sigma(cond_zero,exponent_p)

    cond = Function(space)
    cond.rename('cond')
    radius = skeleton_thickness/2 
    # cond = mu0 * r **p / h**(dim-1) 
    # where the /h**(dim-1) is to get a dirac like measure
    dim_domain = mesh.geometric_dimension()
    cond_no_dirac = Function(space, name='cond_no_dirac')
    cond_no_dirac.rename('cond_no_dirac')
    cond_no_dirac.interpolate(cond_zero*(radius**exponent_p))
    cond.interpolate(cond_no_dirac / (h**(dim_domain-1)))
    cond.rename('cond')

    
    #cond.vector()[:] = 100*(np.ones(cond.vector().local_size())+rand(cond.vector().local_size()))
    #test_adjoint(cond, exponent_m, sigma=sigma, scaling=1.0, nsteps=1000)

    # get the max of cond
    M_max = cond_no_dirac.dat.data.max()
    cond_lift = interpolate(conditional(cond_no_dirac<1e-15,1e15,0)+cond_no_dirac, space)
    M_min = cond_lift.dat.data.min()
    #height = conductivity2image.Barenblatt().height(exponent_m,dim,sigma,M_max)
    height_max = Bar.height(sigma,M_max)
    height_min = Bar.height(sigma,M_min)
    if verbose:
        PETSc.Sys.Print(
        f'p={exponent_p:.1e} d={d} m={exponent_m}'
        + f'B={B:.1e} alpha={alpha:.1e} beta={beta:.1e} K_md={K_md:.1e} sigma={sigma:.1e} f={sigma**alpha:.1e}"')
        PETSc.Sys.Print(f'M_min{M_min:.1e} M_max={M_max:.1e}')
        PETSc.Sys.Print(f'img_height={height_min:.1e} height_max={height_max:.1e} ')
    

    PETSc.Sys.Print(f'{cond.dat.data_ro.min():.2e}<=cond<={cond.dat.data_ro.max():.2e}')

    pm_map = conductivity2image.PorousMediaMap(
        space,
        sigma=sigma, 
        exponent_m=exponent_m, 
        scaling=scaling, 
        nsteps=3,
        solver_parameters=solver_parameters)
    image = pm_map(cond)
    name = f'img_pm'
    image.rename(name)
    PETSc.Sys.Print(f'{image.dat.data_ro.min():.2e}<=img<={image.dat.data_ro.max():.2e}')
    
    if save:
        filename = f'images_img0.pvd'
        cond.rename('img')
        print(f'Saving {filename}')      
        utilities.save2pvd(cond,filename)
        for i, img in enumerate(pm_map.images):
            filename = f'images_img{i+1}.pvd'
            pm_map.images[i].rename('img')
            print(f'Saving {filename}')      
            utilities.save2pvd(pm_map.images[i],filename)

    return cond, image

if (__name__ == '__main__'):

    parser = argparse.ArgumentParser()
    parser.add_argument('--h5', type=str, help='path of h5 file for skeleton')
    #parser.add_argument('--thickness', type=str, default='thickness.nii.gz', help='path of nifti file for thickness')
    parser.add_argument('--cond', type=float, default=1.0)
    args = parser.parse_args()

    exponent_p = 4.0

    # get data
    comm = COMM_WORLD
    n_proc = COMM_WORLD.size
    h5_file = args.h5
    PETSc.Sys.Print(f"Loading data from {h5_file}")
    with CheckpointFile(h5_file, 'r',comm=comm) as afile:
        start = time()
        PETSc.Sys.Print(f"Start mesh ")
        mesh = afile.load_mesh("relabeled_mesh")
        PETSc.Sys.Print(f" Loaded mesh in {time()-start:.2f} seconds")
        start = time()
        skeleton = afile.load_function(mesh, "skeleton")
        PETSc.Sys.Print(f" Loaded skeleton in {time()-start:.2f} seconds")
        start = time()
        thickness = afile.load_function(mesh, "thickness")
        PETSc.Sys.Print(f" Loaded thickness in {time()-start:.2f} seconds")

        # get data
        affine = afile.get_attr("/info/", "affine")
        PETSc.Sys.Print(f" affine ")
        offset = afile.get_attr("/info/", "offset")
        PETSc.Sys.Print(f" offeet")
        voxel_size = afile.get_attr("/info/", "voxel_size")
        PETSc.Sys.Print(f" voxel_size ")
        dimensions = afile.get_attr("/info/", "dimensions")
        PETSc.Sys.Print(f" dimensions ")

        hx, hy, hz = voxel_size
        lengths = np.array([float(dimensions[0]*voxel_size[0]), 
                            float(dimensions[1]*voxel_size[1]), 
                            float(dimensions[2]*voxel_size[2])])
    
    
    
    skeleton_thickness = Function(mesh, name="skeleton_thickness")
    skeleton_thickness.interpolate(skeleton*thickness)

    
    cond, tubular = skeletonthickness_to_tubular(
            skeleton_thickness,         
            exponent_p=exponent_p,
            h=hx,
            cond_zero=args.cond,
            verbose=True,
            save=False)
    PETSc.Sys.Print("done")
    
    VTKfile = File(f"tubular_c{args.cond:.2e}.pvd").write(tubular, cond)    
