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
import os
import argparse
import numpy as np
from niot import conductivity2image
from niot import image2dat as i2d


from ufl import *
from firedrake import *
from firedrake import norm
from firedrake import Function
from firedrake.__future__ import interpolate
# for writing to file
from firedrake import File
import firedrake as fire
#from memory_profiler import profile
import firedrake.adjoint as fire_adj
from firedrake import COMM_WORLD, Ensemble


from niot import utilities as utilities

tape = fire_adj.get_working_tape()

solver_parameters={
                #'snes_type': 'anderson',
                'snes_type' : 'newtonls',
                #'snes_type': 'qn',
                'snes_qn_m': 20,
                'snes_anderson_m': 20,
                # 
                'snes_rtol': 1e-16,
                'snes_atol': 1e-16,
                'snes_stol': 1e-16,
                'snes_max_it': 100,
                'snes_linesearch_type':'l2',
                'ksp_type': 'gmres',
                'ksp_rtol': 1e-10,
                'ksp_atol': 1e-10,
                'ksp_max_it': 500,
                'pc_type': 'hypre',
                'snes_monitor': None,
                #'snes_linesearch_monitor': None,
                #'ksp_monitor': None,
                }



def image2tdens(txt_file_thickness, img_skeleton,img_network, exponent_p=3.0, cond_zero=1e-1):
    np_thickness = np.loadtxt(txt_file_thickness, dtype=np.float32, delimiter='\t', skiprows=0)
    # replace Nan with 0 in data
    np_thickness = np.nan_to_num(np_thickness)
    print(np_thickness.shape)
    pixel_h = 1.0 / np_thickness.shape[1]
    np_thickness *= pixel_h
    np_thickness = np.flip(np_thickness, axis=0)
    
    
    np_skeleton = i2d.image2numpy(img_skeleton,normalize=True,invert=True)
    np_network = i2d.image2numpy(img_network,normalize=True,invert=True)
    

    h = 1.0 / np_skeleton.shape[1]
    print(f'{np_thickness.shape=} {np_skeleton.shape=} {h=}')

    mesh = i2d.build_mesh_from_numpy(np_thickness,mesh_type='cartesian',lengths=[1.0,np_thickness.shape[0]*h])

    thickness = i2d.numpy2firedrake(mesh,np_thickness, name='thickness')
    skeleton = i2d.numpy2firedrake(mesh,np_skeleton, name='skeleton')
    network = i2d.numpy2firedrake(mesh, np_network, name='network')

    space = FunctionSpace(mesh, 'DG', 0)
    thickness_skeleton = assemble(interpolate(skeleton * thickness,space))
    thickness_skeleton.rename('thickness_skeleton')

    pouiseuille = Function(space)
    pouiseuille = assemble(interpolate(cond_zero/h *(thickness_skeleton/2)**exponent_p,space))
    pouiseuille.rename('poiseuille')
    

    pouiseuille_constant = Function(space)
    pouiseuille_constant = assemble(interpolate(h*cond_zero * (thickness/2)**(exponent_p-1)/2,space))
    pouiseuille_constant.rename('poiseuille_constant')
    

    d = mesh.geometric_dimension()-1
    if exponent_p < d:
        raise ValueError('p<d')
    exponent_m = (2 + exponent_p - d ) / (exponent_p - d)
    
        
    dim = mesh.geometric_dimension()-1
    B = conductivity2image.Barenblatt().B(exponent_m,dim)
    alpha = conductivity2image.Barenblatt().alpha(exponent_m,dim)
    beta = conductivity2image.Barenblatt().beta(exponent_m,dim)
    K_md = conductivity2image.Barenblatt().K_md(exponent_m,dim)

    

    # find the time to get 
    # M = M_0 * (r(\sigma))**p 
    sigma = (cond_zero**(-1/exponent_p) * K_md ** (-1/2) * B **(1/2))**(1/beta)
    
    
    M_max = pouiseuille.dat.data.max() * 2 * h 
    height = conductivity2image.Barenblatt().height(exponent_m,dim,sigma,M_max)
    print(
    f'p={exponent_p:.1e} d={d} m={exponent_m}'
    + f'B={B:.1e} alpha={alpha:.1e} beta={beta:.1e} K_md={K_md:.1e} sigma={sigma:.1e} f={sigma**alpha:.1e} img_height={height:.1e} M_max={M_max:.1e}')
    
    
    pm_map = conductivity2image.PorousMediaMap(
        space,
        sigma=sigma, 
        exponent_m=exponent_m, 
        scaling=1.0, 
        nsteps=20,
        solver_parameters=solver_parameters)
    image = pm_map(pouiseuille)
    name = f'pm_map'
    image.rename(name)

    directory = os.path.dirname(txt_file_thickness)

    filename = f'{directory}/pm_thickness.pvd'
    print(f'Saving {filename}') 
    out_file=File(filename)     
    out_file.write(network, thickness, skeleton, thickness_skeleton, pouiseuille,pouiseuille_constant, image)




    mu_pm = i2d.firedrake2numpy(image)
    np.save(f'{directory}/mup{exponent_p:.1e}zero{cond_zero:.1e}.npy',mu_pm)

    mu_constant_pm = i2d.firedrake2numpy(pouiseuille_constant)
    np.save(f'{directory}/mucnstp{exponent_p:.1e}zero{cond_zero:.1e}.npy', mu_constant_pm)

    mu_pouiseuille = i2d.firedrake2numpy(pouiseuille)
    np.save(f'{directory}/mupou{exponent_p:.1e}zero{cond_zero:.1e}.npy',mu_pouiseuille)

    mu_constant_pm = i2d.firedrake2numpy(thickness)
    np.save(f'{directory}/muthickness.npy', mu_constant_pm)


    filename = f'images_img0.pvd'
    pouiseuille.rename('img')
    print(f'Saving {filename}')      
    utilities.save2pvd(pouiseuille,filename)
    for i, img in enumerate(pm_map.images):
        filename = f'images_img{i+1}.pvd'
        pm_map.images[i].rename('img')
        print(f'Saving {filename}')      
        utilities.save2pvd(pm_map.images[i],filename)

    

if (__name__ == '__main__'):    
    parser = argparse.ArgumentParser(description='Corrupt networks with masks and reconstruct via branched transport')
    parser.add_argument('--thk', type=str, help="numpy thickness file")
    parser.add_argument('--s', type=str, help="skeleton image")
    parser.add_argument('--n', type=str, help="network image")
    parser.add_argument('--p', type=float, default=3, help="power")
    parser.add_argument('--c', type=float, default=1.0, help="condzero")
    args = parser.parse_args()

    image2tdens(args.thk, args.s, args.n, args.p, args.c)

    
    

    