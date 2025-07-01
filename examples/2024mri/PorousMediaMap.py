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
    

    pm_map = conductivity2image.PorousMediaMap(
        space,
        sigma=sigma, 
        exponent_m=exponent_m, 
        scaling=scaling, 
        nsteps=10,
        solver_parameters=solver_parameters)
    image = pm_map(cond)
    name = f'img_pm'
    image.rename(name)
    PETSc.Sys.Print(f"Mesh", end="")
    
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
    parser.add_argument('--st', type=str, default='skeleton.nii.gz', help='path of nifti file for skeleton')
    #parser.add_argument('--thickness', type=str, default='thickness.nii.gz', help='path of nifti file for thickness')
    parser.add_argument('--cond', type=float, default=1.0)
    args = parser.parse_args()

    exponent_p = 4.0

    # get data
    skeleton_nii = nibabel.load(args.st)
    skeleton_thickness_np = skeleton_nii.get_fdata()
    dimensions = skeleton_thickness_np.shape
    hx, hy, hz = skeleton_nii.header['pixdim'][1:4]
    lengths = np.array([float(dimensions[0]*hx), 
                        float(dimensions[1]*hy), 
                        float(dimensions[2]*hz)])
    
    #thickness_nii = nibabel.load(args.thickness)
    #thickness_np = thickness_nii.get_fdata()
    
    
    
    
    
    # scale by h, because the skeleton thickness is in pixels
    #skeleton_thickness_np = skeleton_np * thickness_np

    # define the mesh
    PETSc.Sys.Print(f"Mesh", end="")
    cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths)

    # move to fire
    skeleton_thickness = i2d.numpy2firedrake(cartesian_mesh, 
                                             skeleton_thickness_np, 
                                             name="st")
    
    
    cond, tubular = skeletonthickness_to_tubular(
            skeleton_thickness,         
            exponent_p=exponent_p,
            h=hx,
            cond_zero=args.cond,
            verbose=True,
            save=False)
    
    
    filename = f"tubular_c{args.cond:.2e}.nii.gz"
    save_as_nifti(tubular, skeleton_nii.affine, filename) 
    
    filename = f"cond_c{args.cond:.2e}.nii.gz"
    save_as_nifti(cond, skeleton_nii.affine, filename)

    
    # for threshold in [ 1e-3]:
    #     image_support.interpolate(conditional(tubular>threshold*max_image,1,0))

    #     filename = f'pm.pvd'
    #     PETSc.Sys.Print(f'Saving {filename}')      
    #     utilities.save2pvd([
    #         skeleton_thickness,
    #         tubular,
    #         image_support,
    #         ],filename)

    
