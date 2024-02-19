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
import pytest
import argparse
import numpy as np
from niot import conductivity2image
from numpy.random import rand


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
                #'snes_monitor': None,
                #'snes_linesearch_monitor': None,
                #'ksp_monitor': None,
                }

@pytest.mark.parametrize('nref', [1])
def test_adjoint(nref, exponent_m=2.0, sigma=0.1, scaling=1.0, nsteps=10, verbose=False):
    n0 = 32
    nx = n0 * 2**nref
    ny = n0*2 * 2**nref
    h = 1.0 / nx
    
    mesh = RectangleMesh(nx,ny,1,ny*h,quadrilateral=True)
    space = FunctionSpace(mesh, 'DG', 0)
    conductivity = Function(space)
    conductivity.rename('cond')
    conductivity.vector()[:] = 1*(np.ones(conductivity.vector().local_size())+rand(conductivity.vector().local_size()))
    
    pm_map = conductivity2image.PorousMediaMap(space,
                                            sigma=sigma, 
                                            exponent_m=exponent_m, 
                                            scaling=scaling, 
                                            nsteps=nsteps,
                                            solver_parameters=solver_parameters)
    if verbose:
        print(f'Now Conductivity to image map')
    image = pm_map(conductivity)
    image.rename('img')

    integral = assemble(image**2 * dx)
    integral_reduced = fire_adj.ReducedFunctional(integral, fire_adj.Control(conductivity))
       
    if verbose:
        print(f'Now we should see nsteps={pm_map.steps_done} linear system solves in adjoint mode')
    gradient = integral_reduced.derivative()
    if verbose:
        print(f'Now test convergence')
    h = conductivity.copy(deepcopy=True)  
    h *= 0.5     
    conv = fire_adj.taylor_test(integral_reduced, conductivity, h)
    if verbose:
        print(f'conv={conv}')

    
    assert conv > 1.9, 'taylor test failed'

# This dec

@pytest.mark.parametrize('nref', [1])
@pytest.mark.parametrize('lower_factor',[4,8])
@pytest.mark.parametrize('cond_zero', [1e0])
@pytest.mark.parametrize('exponent_p', [3.0])
def test_tdens2image(nref,lower_factor,cond_zero,exponent_p, verbose=False, save=False):
    n0 = 32
    nx = n0 * 2**nref
    ny = n0*2 * 2**nref
    h = 1.0 / nx
    
    mesh = RectangleMesh(nx,ny,1,ny*h,quadrilateral=True)
    space = FunctionSpace(mesh, 'DG', 0)
    x,y = SpatialCoordinate(mesh)
    
    
    # value is small than 1 to have a stronger effect given by r**(1/p)
    value = 1
    line = Function(space)
    line.rename('line')
    width = 2*1/n0 # 2 pixels of the coarsest mesh
    line.interpolate( conditional(x>1/2 -  width/2,1,0)
                    * conditional(x<1/2 +  width/2,1,0)
                    * conditional(y>=1,1,0)
                    +
                    conditional(x>1/2 - lower_factor  * width/2,1,0)
                    * conditional(x<1/2 + lower_factor  * width/2,1,0)
                    * conditional(y<=1,1,0) 
                    )

    
    value = 1
    thickness = Function(space)
    thickness.rename('thickness')
    #thickness.interpolate( conditional(x>1/2-h*2**nref,width,0)
    #                * conditional(x<1/2+h*2**nref,width,0)
    #                * conditional(y>1,1,2))
    thickness.interpolate( 
            conditional(x>1/2-h,1,0)
            * conditional(x<1/2+h,1,0)
            * conditional(y>=1,1,0)*width
            +
            conditional(x>1/2-h,1,0)
            * conditional(x<1/2+h,1,0)
            * conditional(y<=1,1,0)*lower_factor*width
            )
    
    #thickness.interpolate( conditional(x>1/2-h*2**nref,width,0)
    #                * conditional(x<1/2+h*2**nref,width,0)
    #                * conditional(y>1,1,2))
    thickness_dirac = Function(space)
    thickness_dirac.rename('thickness_dirac')
    thickness_dirac.interpolate( conditional(x>1/2-h,1,0)
                    * conditional(x<1/2+h,1,0)
                    * conditional(y<=1,lower_factor,1)
                    * width/h)


  
    

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
    #sigma = (cond_zero**(-1/exponent_p) * K_md ** (-1/2) * B **(1/2))**(1/beta)
    sigma = Bar.sigma(cond_zero,exponent_p)

    pouiseuille = Function(space)
    pouiseuille.rename('poiseuille')
    cond = Function(space)
    cond.rename('cond')
    
    pouiseuille.interpolate(1e-15+cond_zero*(thickness/2)**exponent_p/h)
    cond.interpolate(pouiseuille)
    
    #cond.vector()[:] = 100*(np.ones(cond.vector().local_size())+rand(cond.vector().local_size()))
    #test_adjoint(cond, exponent_m, sigma=sigma, scaling=1.0, nsteps=1000)

    # get the max of cond
    M_max = cond.dat.data.max() * 2 * h 
    #height = conductivity2image.Barenblatt().height(exponent_m,dim,sigma,M_max)
    height = Bar.height(sigma,M_max)
    if verbose:
        print(
        f'p={exponent_p:.1e} d={d} m={exponent_m}'
        + f'B={B:.1e} alpha={alpha:.1e} beta={beta:.1e} K_md={K_md:.1e} sigma={sigma:.1e} f={sigma**alpha:.1e} img_height={height:.1e} M_max={M_max:.1e}')
    
    pm_map = conductivity2image.PorousMediaMap(
        space,
        sigma=sigma, 
        exponent_m=exponent_m, 
        scaling=scaling, 
        nsteps=8,
        solver_parameters=solver_parameters)
    image = pm_map(cond)
    name = f'img_pm'
    image.rename(name)

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


    max_image = image.dat.data.max()
    rel_err_height = abs(max_image-height)/height
    #print(f'max_image={max_image:.1e} height={height:.1e} rel_error={rel_err_height:.1e}')

    #except:
    #    print(f'failed for cond0={cond_zero:.1e}')
    image_support = Function(space)
    image_support.rename('image_support')
    for threshold in [ 1e-3]:
        image_support.interpolate(conditional(image>threshold*max_image,1,0))

        upper_width = assemble(conditional(y>=3/2,1,0) * image_support * dx)/(0.5)
        lower_width = assemble(conditional(y<=1/2,1,0) * image_support * dx)/(0.5)

        rel_err_upper = abs(upper_width-width)/width
        rel_err_lower = abs(lower_width-width*lower_factor)/(width*lower_factor)


        #print(
        #    f'threshold={threshold:.1e}'
        #+f' | 
        #+f' | 
        if verbose:
            print(f'{nref=} {lower_factor=} {cond_zero=} {exponent_p=:.1e} {exponent_m=:.1e}')
            print(f'{rel_err_height=:.1e} {rel_err_upper=:.1e} {rel_err_lower=:.1e}')
        
        assert rel_err_height < 0.2, f'height={max_image:.1e} real={height:.1e} err={rel_err_height:.1e}'
        assert rel_err_upper < 0.6, f'upper approx={upper_width:.1e} real={width:.1e} err={rel_err_upper:.1e}'
        assert rel_err_lower < 0.3, f'lower approx={lower_width:.1e} real={width*lower_factor:.1e} err={rel_err_lower:.1e}'

    if save:
        filename = f'pm_line_nref{nref}.pvd'
        print(f'Saving {filename}')      
        utilities.save2pvd([line,thickness, thickness_dirac,pouiseuille,image_support, cond, *images],filename)


if (__name__ == '__main__'):    
    test_adjoint(nref=1, exponent_m=2.0, sigma=0.1, scaling=1.0, nsteps=10, verbose=True)

    for nref in [1,2,3]:
        for lower_factor in [4,8]:
            for cond_zero in [1e0, 1e-1]:
                for exponent_p in [3.0]:
                    test_tdens2image(nref=nref,
                                lower_factor=lower_factor,
                                cond_zero=cond_zero,
                                exponent_p=exponent_p,
                                verbose=True,
                                save=True)


    