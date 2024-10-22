import sys
import glob
import os
from copy import deepcopy as cp

import numpy as np
from niot import image2dat as i2d
from niot import utilities
from niot import optimal_transport as ot
from niot import NiotSolver


from firedrake import *
from scipy.ndimage import zoom
import time
from firedrake import VTKFile as File

from firedrake.petsc import PETSc

import sys

import itertools
import argparse





def load_data(field, coarseness, data_folder="../../../mri/",mesh_type="simplicial"):

    data = np.load(f'{data_folder}/{field}.npy')

   

    PETSc.Sys.Print(f"Data shape: {data.shape}") 


    if coarseness > 1:
        PETSc.Sys.Print('coarsening image')
        data = zoom(data, (1/coarseness,1/coarseness,1/coarseness), order=0)
        PETSc.Sys.Print(data.shape)

    # create mesh
    PETSc.Sys.Print('building mesh')
    start = time.time()
    lengths = [1.0,data.shape[1]/data.shape[0],data.shape[2]/data.shape[0]]
    mesh = i2d.build_mesh_from_numpy(data, mesh_type=mesh_type,lengths=lengths,label_boundary=True)
    #if mesh_type == "simplicial":
    #    cartesian_mesh = i2d.cartesian_grid_3d(data.shape,lengths)
    #else:
    #    cartesian_mesh = mesh
    
    #mesh = Mesh("Cube-03.msh")
    #mesh.name = 'mesh'
    #mesh.init()
    end = time.time()
    PETSc.Sys.Print(f'mesh built {end-start}s')

    # convert to firedrake
    PETSc.Sys.Print('converting image to firedrake')
    start = time.time()
    tof_fire = i2d.numpy2firedrake(mesh, data, name="TOF")
    end = time.time()
    PETSc.Sys.Print(f'image converted {end-start}s')

    return tof_fire#, cartesian_mesh
    

def btp_inputs(tof_fire):
    mesh = tof_fire.function_space().mesh()
    
    
    # convert to firedrake
    DG0 = FunctionSpace(mesh,"DG",0)
    source = Function(DG0,name="source")
    sink = Function(DG0,name="sink")
    x,y,z = SpatialCoordinate(mesh)
    source.interpolate(conditional(z<0.1,1,0)*conditional(tof_fire>250,1,0))
    sink.interpolate(conditional(tof_fire>150,1,0))


    mass_source = assemble(source*dx)
    mass_sink = assemble(sink*dx)

    PETSc.Sys.Print(f"{mass_source=} {mass_sink=}")


    source /= mass_source
    sink /= mass_sink

    
    corrupted = Function(DG0,name="corrupted")
    corrupted.interpolate(conditional(tof_fire>250,1,0)*tof_fire)

    return source, sink, corrupted

    


def labels(fem,
           gamma,wd,wr,
           corrupted_as_initial_guess,
           confidence,
           tdens2image, 
           method):
    label= [
        f'fem{fem}',
        f'gamma{gamma:.1e}',
        f'wd{wd:.1e}',
        f'wr{wr:.1e}',
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



def setup_solver(source, sink, corrupted,
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
                 verbose=0):


   
    

    confidence = Function(source.function_space())
    confidence.assign(1.0)


    # Define the branched transport problem
    ot.balance(source, sink)
    gamma=0.5
    btp = ot.BranchedTransportProblem(source, sink, gamma=gamma)


    niot_solver = NiotSolver(btp, 
                             corrupted,  
                             confidence=confidence, 
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
    niot_solver.ctrl_set('max_iter', 10000)
    niot_solver.ctrl_set('max_restart', 4)
    niot_solver.ctrl_set('verbose', 2)

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
    
    return niot_solver



   
#
# common setup
#
fems = ["CG1DG0"]
wr = [0.0]
method = [
    "tdens_mirror_descent_explicit",
]

def figure1():
    #
    # Combinations producting the data for Figure 2
    #
    gamma = [0.5] # 
    wd = [1e0]  # set the discrepancy to zero
    ini = [0]
    # the following are not influent since wd=weight discrepancy is zero
    conf = ["ONE"]
    maps = [
        #{"type": "identity", "scaling": 1/20},
        {"type": "identity", "scaling": 1/10},
    ]
    parameters = [
        fems,
        gamma,
        wd,
        wr,
        ini,
        conf,
        maps,
        method,
    ]
    combinations = list(itertools.product(*parameters))

    return combinations


    
     

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Reconstruct network')
    parser.add_argument("--field", type=str, default='TOF', help="TOF")
    parser.add_argument("--c", type=int, default=8, help="coarseing factor")
    args, unknown = parser.parse_known_args()

    field = args.field
    coarseness = args.c

    results = "./results_dg0/"
    if not  os.path.exists(results):
        os.mkdir(results)

    test_case = f"{field}_{coarseness:02}"
    out_directory = results+test_case
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)




    #setup inputs
    mesh_type = "cartesian" if fems[0]=="DG0DG0" else "simplicial"
    tof =  load_data(field, coarseness, mesh_type=mesh_type)

    source, sink, corrupted = btp_inputs(tof)

    out_file = File(f'{out_directory}/inputs.pvd')
    out_file.write(tof,source,sink,corrupted)

    
      
    #setup controls
    combinations = figure1()
    
    for combination in combinations:
    
        label = "_".join(labels(*combination))+"4000"

        PETSc.Sys.Print(label)

        label_dir = sys.join(out_directory,label)
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        
        # setup solvers
        niot_solver = setup_solver( source, sink, corrupted, *combinations[0])

        ierr = niot_solver.solve()

        # save solution
        pot, tdens, vel = niot_solver.get_otp_solution(niot_solver.sol)

        DQ0 = FunctionSpace(cartesian_mesh,"DQ",0)
        tdens_grid = Fuction(DQ0, name="tdens_grid")
        pot_grid = Fuction(DQ0, name="pot_grid")

        tdens_grid.interpolate(tdens)
        pot_grid.interpolate(pot)


        reconstruction = Function(niot_solver.fems.tdens_space)
        reconstruction.interpolate(niot_solver.tdens2image(tdens) )
        reconstruction.rename('reconstruction','Reconstruction')

        filename = f'{label_dir}/reconstruction.pvd'
        out_file = VTKFile(filename,mode='w')
        out_file.write(pot, tdens)
        PETSc.Sys.Print(f"{ierr=}. Saved solution to "+filename)

        numpy_name = f"tdens_{comm.rank:.02d}.npy"
        with tdens_grid.dat.vec_ro as v:
            v_np = v.array
            v_np.tofile(numpy_name)

        numpy_name = f"pot_{comm.rank:.02d}.npy"
        with pot_grid.dat.vec_ro as v:
            v_np = v.array
            v_np.tofile(numpy_name)




