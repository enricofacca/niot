import sys
import glob
import os
from copy import deepcopy as cp


from niot import image2dat as i2d
import numpy as np
from niot import utilities
from firedrake import *
from scipy.ndimage import zoom
import time
from firedrake import VTKFile as File

from firedrake.petsc import PETSc

import sys

field = "TOF"

coarseness = int(sys.argv[1])

load_and_convert = True
if load_and_convert:

    data = np.load(f'../../mri/{field}.npy')



    PETSc.Sys.Print(f"Data shape: {data.shape}") 


    if coarseness > 1:
        PETSc.Sys.Print('coarsening image')
        data = zoom(data, (1/coarseness,1/coarseness,1/coarseness), order=0)
        PETSc.Sys.Print(data.shape)

    # create mesh
    PETSc.Sys.Print('building mesh')
    start = time.time()
    lengths = [1,data.shape[1]/data.shape[0],data.shape[2]/data.shape[0]]
    mesh = i2d.build_mesh_from_numpy(data, mesh_type='cartesian',lengths=lengths)
    mesh.name = 'mesh'
    end = time.time()
    PETSc.Sys.Print(f'mesh built {end-start}s')

    # convert to firedrake
    PETSc.Sys.Print('converting image to firedrake')
    start = time.time()
    tof_fire = i2d.numpy2firedrake(mesh, data, name="TOF")
    end = time.time()
    PETSc.Sys.Print(f'image converted {end-start}s')

    # convert to firedrake
    DG0 = FunctionSpace(mesh,"DG",0)
    source = Function(DG0,name="source")
    sink = Function(DG0,name="sink")
    x,y,z = SpatialCoordinate(mesh)
    source.interpolate(conditional(z<0.01,1,0)*conditional(tof_fire>250,1,0))
    sink.interpolate(conditional(tof_fire>150,1,0))


    mass_source = assemble(source*dx)
    mass_sink = assemble(sink*dx)

    source /= mass_source
    sink /= mass_sink

    
    corrupted = Function(DG0,name="corrupted")
    corrupted.interpolate(conditional(tof_fire>250,1,0)*tof_fire)

    out_file = File(f'c{coarseness}.pvd')
    out_file.write(tof_fire,source,sink,corrupted)



def reconstruct( source,sink,corrupted,
                 fem="DG0DG0",
                 gamma=0.5, 
                 wd=1e-2,
                 wr=0.0, 
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
                 verbose=0)


   
    

    confidence = Function(source.function_space())
    confidence.assgin(1.0)


    # Define the branched transport problem
    ot.balance(source, sink)
    gamma=0.5
    btp = ot.BranchedTransportProblem(source, sink, gamma=gamma)


    niot_solver = NiotSolver(btp, corrupted=,  confidence_w, 
                             spaces = fem,
                             cell2face = 'harmonic_mean',
                             setup=False)


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
    niot_solver.ctrl_set('max_iter', 0)


    start = time.process_time()
    ierr = niot_solver.solve()
    cpu_rec = time.process_time() - start 



    # save solution
    pot, tdens, vel = niot_solver.get_otp_solution(niot_solver.sol)
    
    reconstruction = Function(niot_solver.fems.tdens_space)
    reconstruction.interpolate(niot_solver.tdens2image(tdens) )
    reconstruction.rename('reconstruction','Reconstruction')

    filename = os.path.join(directory, f'{label}.pvd')
    out_file = VTKFile(filename,comm=comm,mode='w')
    out_file.write(pot, tdens)
    PETSc.Sys.Print(f"{ierr=}. {n_ensemble=} Saved solution to "+filename, comm=comm)


    

    # PETSc.Sys.Print('saving to file')
    # start = time.time()
    # fname = f'{field}_c{coarseness}.h5'
    # with CheckpointFile(fname, 'w') as afile:
    #     afile.save_mesh(mesh)  # optional
    #     afile.save_function(data_fire)
    # end = time.time()
    # PETSc.Sys.Print(f'saved to file {end-start}s')

PETSc.Sys.Print('loading from file')
fname = f'{field}_c{coarseness}.h5'
start = time.time()
with CheckpointFile(fname, 'r') as afile:
    mesh = afile.load_mesh("mesh")
    field_read = afile.load_function(mesh, f"{field}")
end = time.time()
PETSc.Sys.Print(f'loaded from file {end-start}s')
