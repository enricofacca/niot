import sys
import glob
import os
from copy import deepcopy as cp
 
import gc
import cc3d
import numpy as np
from niot import image2dat as i2d
from niot import utilities
from niot import optimal_transport as ot
from niot import NiotSolver
from niot import SpaceDiscretization
#from memory_profiler import profile

from niot.conductivity2image import HeatMap

from firedrake import *
from scipy.ndimage import zoom
import time
from firedrake import VTKFile as File

from firedrake.petsc import PETSc

import sys

import itertools
import argparse
import nibabel

from scipy.ndimage import gaussian_filter

import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(formatter={'float': '{:0.2e}'.format})

def mpi_mkdir(directory, comm=COMM_WORLD):
    if os.path.exists(directory):
        return
    
    
    comm.Barrier()
    if comm.rank == 0:
        os.mkdir(directory)
    comm.Barrier()
    return

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    m = len(lst)// n
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def build_meshes_from_numpy(data, mesh_type="simplicial",lengths=None, comm=COMM_WORLD, label_boundary=False):
    if lengths is None:
        lengths = [1.0,data.shape[1]/data.shape[0],data.shape[2]/data.shape[0]]
    
    mesh = i2d.build_mesh_from_numpy(data, mesh_type=mesh_type,lengths=lengths,comm=comm,label_boundary=label_boundary)
    if mesh_type == "simplicial":
        cartesian_mesh = i2d.cartesian_grid_3d(data.shape,lengths,comm=comm)
    else:
        cartesian_mesh = mesh
    return mesh, cartesian_mesh

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
    if mesh_type == "simplicial":
        cartesian_mesh = i2d.cartesian_grid_3d(data.shape,lengths)
    else:
        cartesian_mesh = mesh
    
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

    return tof_fire, cartesian_mesh
    

def setup_btp(brain_mask, inlets, corrupted, constant_absorption = 1):
    mesh = brain_mask.function_space().mesh()
    
    # convert to firedrake
    DG0 = FunctionSpace(mesh,"DG",0)
    sink = Function(DG0,name="sink")
   
    #source.interpolate(conditional(z<15,1,0)*conditional(tof_fire>250,1,0))
    R = FunctionSpace(mesh,"R",0)
    source = Function(R,val=0.0, name="source")
    # above 150 define the approximate support of absortion
    # above 250 is remove beacuse where we know we have blood vessels
    #sink.interpolate(conditional(tof_fire>threshold_domain,1,0) 
    #                 * conditional(tof_fire<|etwork,1,0))
    
    
    sink.interpolate(-conditional(brain_mask>1e-10,constant_absorption,0)
                     * conditional(corrupted>0,0,1)) # remove blood vessels outside the mask 
    

    mass_source = assemble(source*dx)
    mass_sink = assemble(sink*dx)

    #source /= mass_source
    #sink /= mass_sink
    PETSc.Sys.Print(f"{mass_source=:.2e} {mass_sink=:.2e}")


    threshold_bones = 50
    kappa = Function(R,val=1.0,name="kappa")
    #kappa = Function(DG0,name="kappa")
    #kappa.interpolate(conditional(tof_fire < threshold_bones,0.1,1.0))


    # Define the branched transport problem
    gamma=0.5
    
    inlet_pressure = Function(inlets.function_space())
    inlet_pressure.assign(0.0)
    weak_Dirichlet = [(inlet_pressure, ds_b, inlets)]
    #strong_Dirichlet = [(source, ds_b, inlets)]
    btp = ot.BranchedTransportProblem(source, sink, 
                                      gamma=gamma, 
                                      Dirichlet = None,
                                      weak_Dirichlet = weak_Dirichlet,
                                      kappa=kappa)

    return btp
    


def labels(fem,
           gamma,wd,wr,
           ini,
           confidence,
           tdens2image, 
           method,
           absortion):
    name = ini.name()
    label= [
        f'fem{fem}',
        f'gamma{gamma:.1e}',
        f'wd{wd:.1e}',
        f'wr{wr:.1e}',
        f'ini'+name,
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
    label.append(f"sink{absortion:.1e}")
    return label


def setup_solver(btp, 
                 corrupted, 
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
                 ensemble_comm=None,
                 verbose=0):


   
    mesh = btp.source.function_space().mesh()

    confidence = Function(btp.source.function_space())
    confidence.assign(1.0)


    

    niot_solver = NiotSolver(btp, 
                             corrupted,  
                             confidence=confidence, 
                             spaces = fem,
                             cell2face = 'harmonic_mean',
                             setup=False,
                            ensemble_comm=ensemble_comm,
                             )


    # Setup the solver's parameters

    # inpainting
    niot_solver.ctrl_set('discrepancy_weight', wd)
    niot_solver.ctrl_set('regularization_weight', wr)
    niot_solver.ctrl_set(['tdens2image'], tdens2image)

    # optimization
    niot_solver.ctrl_set('optimization_tol', 1e-5)
    niot_solver.ctrl_set('constraint_tol', 1e-6)
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
    
    return niot_solver



   


def downsample(data,coarseness,mode="zoom"):
    """
    Downsample data by a factor of coarseness
    Args:
    data: numpy 2d or 3d array
    coarseness: int
    mode: str
        "zoom": uses scipy.ndimage.zoom
        "max" : uses max of neighbours cells
    Returns:
        data: numpy 2d or 3d array
    """
    if coarseness <= 1:
        return data
      
    if mode == "zoom":
        factors = tuple([1 if n==1 else 1/coarseness for n in data.shape])
        data = zoom(data, factors, order=0)
        PETSc.Sys.Print(data.shape)
        return data

    elif mode == "max":
        # the extra boundary required
        pad_shape = tuple([(0, int(coarseness*np.ceil(n/coarseness)-n)) for n in data.shape])

        # pad with -inf
        data = np.pad(data, pad_shape, mode='constant', constant_values=-np.inf)
        

        # coarseness = 2
        # (nx//2,2,ny//2,2,nz//2,2)
        reshaped_shape = sum([[n//coarseness, coarseness] for n in data.shape],[])
        
        
        # reshape gathering neighbours cells and get the max
        extra_dim = tuple([2*i+1 for i in range(data.ndim)])
        
        data = data.reshape(reshaped_shape).max(axis=extra_dim)
        
        return data

def indices_restrict(data, lengths, xyz_bounds):
    """ 
    Assuming that LX=1
    """
    indices_bounds = []
    for axis_index in range(len(data.shape)):
        n_axis = data.shape[axis_index]
        len_axis = lengths[axis_index]
        if xyz_bounds[axis_index] is None:
            indices_bounds.append([0,n_axis])
        else:
            lower, upper = xyz_bounds[axis_index]
            indices_bound = [max(0,int(lower/len_axis*n_axis)),min(int(upper/len_axis*n_axis),n_axis)]
            indices_bounds.append( indices_bound)
                

    return np.array(indices_bounds)

def restrict(data, indices_bounds):
    """ 
    Assuming that LX=1
    """
    data = data[indices_bounds[0][0]:indices_bounds[0][1],
                indices_bounds[1][0]:indices_bounds[1][1],
                indices_bounds[2][0]:indices_bounds[2][1]]
    data = np.ascontiguousarray(data)
    return data




def select_slice(tof_np, out_directory):
    # get bottom slice of tof
    tof_bottom_np = np.flipud(tof_np[:,:,0])
    tof_bottom_np /= tof_bottom_np.max()
    tof_bottom_np[tof_bottom_np<0.25] = 0
    # save as vtr and png
    #i2d.numpy2vtr(tof_bottom_np, lengths[0:2], f"{out_directory}/tof", name='tof')
    i2d.numpy2image(tof_bottom_np, f"{out_directory}/tof_bottom.png") 


#@profile
def poisson(cartesian_mesh, btp):
    """
    Test solver for possion equation
    """

    SD = SpaceDiscretization(cartesian_mesh, "DG", 0)

    # the forcing term
    pot_space = FunctionSpace(cartesian_mesh,"DG",0)
    pot_h = Function(pot_space, name="pot_h")

    energy_form = SD.Laplacian_Lagrangian(pot_h)
    PDE = derivative(energy_form, pot_h)
    a = derivative(PDE, pot_h)

    test = TestFunction(SD.pot_space)
    L = (btp.source - btp.sink) * test * dx

    # fix boundary conditions
    PETSc.Sys.Print(f"Imposing Dirichlet weakly :")
    # impose weak Dirichlet BCsdegree
    a = SD.apply_weak_Dirichlet_lhs(btp.weak_Dirichlet, a)# penalty=1e6)
    L = SD.apply_weak_Dirichlet_rhs(btp.weak_Dirichlet, L)#, penalty=1e6)
    bcs = None
        

    # set nullspace (if no Dirichlet BCs)
    bcs = None
    nullspace = None
    

    # setup solver
    solver_parameters={ "ksp_type": "cg",
                        "ksp_max_it": 1000,
                        "ksp_rtol": 1e-13, 
                        "ksp_atol": 1e-13,
                        "pc_type": "hypre",
                        #"ksp_monitor_true_residual": None,
                        }
    
    # solver of poisson equation
    petsc_controls ={
        "snes_monitor": None,
        # krylov solver controls
        'ksp_type': 'cg',
        'pc_type': 'hypre',
        'ksp_atol': 1e-16,
        'ksp_rtol': 1e-5,
        'ksp_dtol': 1e5,
        'ksp_max_it' : 1000,
        'ksp_initial_guess_nonzero': True, 
        'ksp_norm_type': 'unpreconditioned',
        #'ksp_monitor_true_residual' : None, 
    }

    dim = cartesian_mesh.geometric_dimension()
    PETSc.Sys.Print(f"3d hypre {dim}") 
    if cartesian_mesh.geometric_dimension() == 3:
        hypre_ctrl_3d = {
                        # tuning parameters for the multigrid
                        # https://mooseframework.inl.gov/releases/moose/2021-09-15/application_development/hypre.html
                        "pc_hypre_type": "boomeramg",
                        "pc_hypre_boomeramg_strong_threshold": 0.8,
                        "pc_hypre_boomeramg_max_iter": 1,
                        "pc_hypre_boomeramg_agg_nl": 4,
                        "pc_hypre_boomeramg_interp_type": "ext+i",  # "classic" or "ext+i"
                    }
        petsc_controls.update(hypre_ctrl_3d)
        PETSc.Sys.Print(f"3d hypre") 

    problem = LinearVariationalProblem(a, L, pot_h, bcs=bcs)
    solver = LinearVariationalSolver(problem, 
                                     solver_parameters=solver_parameters, 
                                     nullspace=nullspace)
    
    # solve problem
    solver.solve()
    #solver.snes.ksp.view()
    ksp_iterations = solver.snes.ksp.getIterationNumber()

    PETSc.Sys.Print(f"Number of iterations: {ksp_iterations}")

    # save solution to pvd
    out_file = File(f'pot.pvd')
    out_file.write(pot_h)



#@profile
def experiment(args):

    field = "TOF"
    coarseness = args.c
    results = args.out
    threshold_network = args.threshold_network


    # make directories
    mpi_mkdir(results)

    test_case = f"{field}_{coarseness:02}_threshold_{threshold_network:.1e}"
    out_directory = results+test_case
    mpi_mkdir(out_directory)

    my_ensemble = Ensemble(COMM_WORLD, args.n_ensemble)
    
    # load tof data
    tof_data = nibabel.load(args.mri+'TOF.nii.gz')
    original_dimensions = tof_data.header.get_data_shape()[:3]
    PETSc.Sys.Print(f"Data shape: {original_dimensions=}")
    hx, hy, hz = tof_data.header['pixdim'][1:4]
    lengths = np.array([float(original_dimensions[0]*hx), 
                        float(original_dimensions[1]*hy), 
                        float(original_dimensions[2]*hz)])

    # create mesh
    time0 = time.time()
    mesh_type = "cartesian"
    

    if my_ensemble.ensemble_comm.rank == 0:
        tof_np = tof_data.get_fdata()    
        
        
        
        # load t1 data
        t1_data = nibabel.load(args.mri+'/T1.nii.gz')
        t1_np = t1_data.get_fdata() 
        PETSc.Sys.Print(f"Data shape: {tof_np.shape} Lengths: {lengths}")

        # load brain mask
        brain_mask_data = nibabel.load(args.mri+'/brain_resampled.nii.gz')
        brain_mask_np = brain_mask_data.get_fdata() 
        PETSc.Sys.Print(f"Data shape: {brain_mask_np.shape} Lengths: {lengths}")

        # load inlet data
        #tof_inlet_np = i2d.image2numpy(f"{args.mri}/labels.png",normalize=False)
        #tof_inlet_np = np.flipud(tof_inlet_np)
        # read from file inlets.npy
        tof_inlet_np = np.load(f"{args.mri}/inlets.npy")
        tof_inlet_np = tof_inlet_np.reshape((tof_inlet_np.shape[0],tof_inlet_np.shape[1],1),order='F', copy=True)
        
        # load labels
        labels_data = nibabel.load(args.mri+'/labels.nii.gz')
        labels_np = labels_data.get_fdata()

        
        # setup problem
        main_network = np.zeros_like(labels_np)
        main_network[labels_np == 60000] = 1
        external_network = np.zeros_like(labels_np)
        for value in [50000,40000,30000,20000]:
            external_network[labels_np == value] = 1
        
        corrupted_np = np.copy(tof_np)
        corrupted_np[tof_np < threshold_network] = 0.0
        # remove blood vessels outside the mask
        corrupted_np[brain_mask_np < 1e-10] = 0.0
        # re-integrate main network outise the mask
        corrupted_np[main_network == 1] = tof_np[main_network == 1]
        # remove external network inside the mask
        corrupted_np[external_network == 1] = 0


        # inlets points
        inlets_np = np.zeros_like(tof_inlet_np)
        for value in [5,7,14,17]:
            inlets_np[tof_inlet_np == value] = 1
        
        
        # restrict data
        restrict_domain = True
        if restrict_domain:
            indices_bounds = indices_restrict(tof_np,
                                            lengths=lengths,
                                            xyz_bounds=[
                                                [args.xmin,args.xmax],
                                                [args.ymin,args.ymax],
                                                [args.zmin,args.zmax]
                                            ])    
            tof_np = restrict(tof_np,indices_bounds)
            t1_np = restrict(t1_np,indices_bounds)
            tof_inlet_np = restrict(tof_inlet_np,indices_bounds)
            inlets_np = restrict(inlets_np,indices_bounds)
            brain_mask_np = restrict(brain_mask_np,indices_bounds)  
            main_network = restrict(main_network,indices_bounds)
            external_network = restrict(external_network,indices_bounds)
            corrupted_np = restrict(corrupted_np,indices_bounds)            
            
        dimensions = tof_np.shape
        lengths = np.array(tof_np.shape)*np.array([hx,hy,hz])
        PETSc.Sys.Print(f"Data shape: {tof_np.shape} Lengths: {lengths} After restriction ")

        # coarsen data
        mode = "max"
        if coarseness > 1:
            # check if coarseness is a power of 2
            if not np.log2(coarseness).is_integer():
                raise ValueError("Coarseness must be a power of 2")

            tof_np = downsample(tof_np,coarseness,mode)
            t1_np = downsample(t1_np,coarseness,mode)
            tof_inlet_np = downsample(tof_inlet_np,coarseness,mode)
            inlets_np = downsample(inlets_np,coarseness,mode)
            corrupted_np = downsample(corrupted_np,coarseness,mode)
            brain_mask_np = downsample(brain_mask_np,coarseness,mode)
            main_network = downsample(main_network,coarseness,mode)
            external_network = downsample(external_network,coarseness,mode)
            PETSc.Sys.Print(f"Coarse Data shape: {t1_np.shape}")

            hx *= coarseness
            hy *= coarseness
            hz *= coarseness

        dimensions = tof_np.shape
        lengths = np.array(tof_np.shape)*np.array([hx,hy,hz])
        PETSc.Sys.Print(f"Data shape: {tof_np.shape} Lengths: {lengths} After Downsample ")


        cc = False
        if cc:
            support_corrupted = np.zeros_like(corrupted_np)
            support_corrupted[corrupted_np > 0] = 1
            cc_corrupted, n_cc_corrupted = cc3d.connected_components(
                support_corrupted, 
                connectivity=26, 
                binary_image=True, 
                return_N=True)
            PETSc.Sys.Print(f"{n_cc_corrupted=}")   


        # saving inputs in vtr
        PETSc.Sys.Print("start saving inputs as vtr")
        start = time.time()
        i2d.numpy2vtr([t1_np,tof_np, corrupted_np,main_network,external_network,brain_mask_np],#,cc_corrupted], 
                    lengths, f"{out_directory}/mri", 
                    names=['t1','tof','corrupted','main_network','external_network','brain_mask'],#,'cc_corrupted'])
                    comm=my_ensemble.comm)
        i2d.numpy2vtr([tof_inlet_np,inlets_np], 
                    [lengths[0],lengths[1],hz],
                    f"{out_directory}/inlets", names=['tof_inlets','inlets'],
                    comm=my_ensemble.comm)
        PETSc.Sys.Print("saved inputs in "+f'{out_directory}/inputs'+f" in {time.time()-start:.2f}s")

        

        # free memor
        tof_np = None
        t1_np = None
        tof_inlet_np = None
        tof_inlet_np = None
        main_network = None
        external_network = None
        #cc_corrupted = None
        support_corrupted = None
        dimensions = np.array(dimensions,dtype=int)
        lengths = np.array(lengths,dtype=float)
    else:
        dimensions = np.empty(3,dtype=int)
        lengths = np.empty(3,dtype=float)

    my_ensemble.ensemble_comm.barrier()
    my_ensemble._ensemble_comm.Bcast(dimensions, root=0)
    my_ensemble._ensemble_comm.Bcast(lengths, root=0)
    

    cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths,comm=my_ensemble.comm)
    PETSc.Sys.Print(f"Mesh built")

    space = FunctionSpace(cartesian_mesh,"DG",0)
    if my_ensemble.ensemble_comm.rank == 0:
        PETSc.Sys.Print("converting into firedrake")
        # inlets
        inlets_3d_np = np.zeros_like(brain_mask_np)
        inlets_3d_np[:,:,0] = inlets_np[:,:,0]
        inlets = i2d.numpy2firedrake(cartesian_mesh, inlets_3d_np, name="Inlets")
        brain_mask = i2d.numpy2firedrake(cartesian_mesh, brain_mask_np, name="Brain_mask")
        
        # corrupted data
        corrupted = i2d.numpy2firedrake(cartesian_mesh, corrupted_np, name="corrupted")

        PETSc.Sys.Print(f"converted into firedrake")
    else:
        
        inlets = Function(space,name="Inlets")
        brain_mask = Function(space,name="Brain_mask")
        corrupted = Function(space,name="corrupted")

    brain_mask_np = None
    inlets_np = None
    inlets_3d_np = None
    corrupted_np = None
    
    my_ensemble.ensemble_comm.barrier()
    PETSc.Sys.Print(f"start broadcasting")
    my_ensemble.bcast(inlets,0)
    my_ensemble.bcast(brain_mask,0)
    my_ensemble.bcast(corrupted,0)
    PETSc.Sys.Print(f"end broadcasting")
    
    # initial guess
    heat_flow = True
    sigma0 = 1.0
    base = 4
    min_tdens = 1e-4

    initial_guess = False
    if initial_guess:
        PETSc.Sys.Print(f"Initial guess")
        if heat_flow:
            heat = HeatMap(corrupted.function_space(), scaling=1.0, sigma=1e1)
            low = Function(space,name="LOW")
            medium = Function(space,name="MEDIUM")
            high = Function(space,name="HIGH")
            
            low.assign(heat(corrupted+min_tdens) + min_tdens)
            medium.assign(heat(low+min_tdens) + min_tdens)
            high.assign(heat(medium+min_tdens) + min_tdens)
            
            low_np = i2d.firedrake2numpy(low)
            medium_np = i2d.firedrake2numpy(medium)
            high_np = i2d.firedrake2numpy(high)
        else:
            low_np = gaussian_filter(corrupted_np, sigma=base**0*sigma0, truncate=1e0)
            medium_np = gaussian_filter(low_np, sigma=base, truncate=1e0)
            high_np = gaussian_filter(medium_np, sigma=base, truncate=1e0)
            
            low = i2d.numpy2firedrake(cartesian_mesh, low_np, name="LOW")
            medium = i2d.numpy2firedrake(cartesian_mesh, medium_np, name="MEDIUM")
            high = i2d.numpy2firedrake(cartesian_mesh, high_np, name="HIGH")
            for ini in [low,medium,high]:
                ini += min_tdens

        if my_ensemble.ensemble_comm.rank == 0:
            # save tof_low
            i2d.numpy2vtr([low_np, medium_np, high_np],
                            lengths, 
                            f"{out_directory}/initial_data",
                            names=['tof_low','tof_medium','tof_high'],
                            comm=my_ensemble.comm)

        # free memory    
        low_np = None
        medium_np = None
        high_np = None
        
    one = Function(space, name="ONE")
    one.assign(1.0)
    PETSc.Sys.Print(f"Initial guess built")
    

    #initials = [low,medium,high,one]

    save_inputs_as_pvd = False
    if save_inputs_as_pvd:
        PETSc.Sys.Print("start saving inputs as pvd")
        start = time.time()
        out_file = VTKFile(f'{out_directory}/inputs{my_ensemble.ensemble_comm.rank}.pvd',comm=my_ensemble.comm)
        fun = initials[my_ensemble.ensemble_comm.rank]
        out_file.write(fun)
        PETSc.Sys.Print("saved inputs in "+f'{out_directory}/inputs.pvd'+f" in {time.time()-start:.2f}s")
        
        PETSc.Sys.Print("start saving inputs as pvd")
        start = time.time()
        out_file = VTKFile(f'{out_directory}/corrupted.pvd', comm=my_ensemble.comm)
        out_file.write(corrupted)
        PETSc.Sys.Print(f"saved inputs in {out_file}"+f" in {time.time()-start:.2f}s")
        
        

    

    def figure1():
            #
        # common setup
        #
        fems = ["DG0DG0"]
        wr = [0.0]
        method = [
            "tdens_mirror_descent_explicit",
        ]


        #
        # Combinations producting the data for Figure 2
        #
        gamma = [0.5] # 
        wd = [1e-4,1e-3]  # set the discrepancy to zero
        ini = [one]
        # the following are not influent since wd=weight discrepancy is zero
        conf = ["ONE"]
        maps = [
            #{"type": "identity", "scaling": 1/20},
            #{"type": "identity", "scaling": 1},
            {"type": "identity", "scaling": 10},
            #{"type": "identity", "scaling": 100},
        ]
        absorption = [1e-3]#,1e-4]
        parameters = [
            fems,
            gamma,
            wd,
            wr,
            ini,
            conf,
            maps,
            method,
            absorption,
        ]
        combinations = list(itertools.product(*parameters))

        return combinations

    

    #setup controls
    combinations = figure1()

    
    


    test_poisson = False
    if test_poisson:
        btp = setup_btp(brain_mask, inlets, corrupted, constant_absorption = 1.0)
        poisson(cartesian_mesh, btp)
        exit()


    
    
    # divide the combinations in the ensemble
    def lol(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    sub_combinations = list(lol(combinations, my_ensemble.ensemble_comm.size))
    todo = sub_combinations[my_ensemble.ensemble_comm.rank]

    for i in range(my_ensemble.ensemble_comm.size):
        if i == my_ensemble.ensemble_comm.rank:
            print(f"ENSEMBLE {i}")
            for j, comb in enumerate(todo):
                label = "_".join(labels(*comb))
                if my_ensemble.comm.rank == 0:
                    print(f"{j} {label}")   
        my_ensemble.ensemble_comm.barrier()

    my_ensemble.ensemble_comm.barrier()


    PETSc.Sys.Print(f"{my_ensemble.ensemble_comm.rank=} {len(todo)=}")
    for i, combination in enumerate(todo):
        label = "_".join(labels(*combination))
        PETSc.Sys.Print(f"{i} {my_ensemble.ensemble_comm.rank=} {label}")

        # btp inputs
        absortion = combination[-1]
        btp = setup_btp(brain_mask, inlets, corrupted, constant_absorption = absortion)

        label_dir = os.path.join(out_directory,label)
        mpi_mkdir(label_dir, my_ensemble.comm)

        save_inputs_as_pvd = False
        if save_inputs_as_pvd :
            PETSc.Sys.Print("start saving sink as pvd")
            start = time.time()
            filename = f"{label_dir}/sink.pvd"
            PETSc.Sys.Print(f"Saving {filename}")
            outfile = VTKFile(filename, comm=my_ensemble.comm)
            outfile.write(btp.sink)
            PETSc.Sys.Print(f"saved inputs in {filename}"+f" in {time.time()-start:.2f}s")
        
        save_inputs_as_vtr = False
        if save_inputs_as_vtr:
            filename=f"{label_dir}/sink"
            PETSc.Sys.Print(f"Saving {filename}")
            sink_np = i2d.firedrake2numpy(btp.sink)
            i2d.numpy2vtr([sink_np], lengths, filename, names=['sink'], comm=my_ensemble.comm)
        

        my_ensemble.ensemble_comm.barrier()
        
        
        
        # setup solvers
        if my_ensemble.comm.rank == 0:
            print(f"BEGIN {i+1}/{len(todo)} ensemble {my_ensemble.ensemble_comm.rank}: {label}")
        niot_solver = setup_solver( btp, corrupted, *combination)
        
        # setup log file
        log_filename = os.path.join(label_dir,f"niot.log")
        niot_solver.ctrl_set("log_file",log_filename)
        
        # set solvers according to controls
        niot_solver.setup()
        
        #
        initial = combination[4]
        niot_solver.set_solution(tdens=initial)
        
        
        # run solver
        total_iterations = niot_solver.ctrl_get('max_iter')
        buffer_saving = min(10000,total_iterations)


        def solve_and_save(niot_solver, label_dir):
            ierr = niot_solver.solve()

            # save solution
            pot, tdens, vel = niot_solver.get_otp_solution(niot_solver.sol)

            tdens_np = i2d.firedrake2numpy(tdens)
            pot_np = i2d.firedrake2numpy(pot)

            tdens = None
            pot = None


            # save tdens and pot as npy files
            np.save(f"{label_dir}/tdens.npy",tdens_np)
            np.save(f"{label_dir}/pot.npy",pot_np)
        
            filename=f"{label_dir}/tdens_pot"
            PETSc.Sys.Print(f"Saving {filename}")
            i2d.numpy2vtr([tdens_np,pot_np],
            lengths, 
            filename, 
            names=['tdens','pot'],
            comm=my_ensemble.comm)
            if my_ensemble.comm.rank == 0:
                print(f"DONE  {i+1}/{len(todo)} ensemble {my_ensemble.ensemble_comm.rank}: {label}")

            tdens_np = None
            pot_np = None

        # run and save
        niot_solver.ctrl_set('max_iter', total_iterations%buffer_saving)
        solve_and_save(niot_solver, label_dir)
        PETSc.Sys.Print(f"First {total_iterations%buffer_saving} iterations done")

        # run and save, skipping initial steps
        niot_solver.ctrl_set('max_iter',buffer_saving)
        niot_solver.ctrl_set('restart',True)
        for i in range(total_iterations//buffer_saving):
            PETSc.Sys.Print(f"Restarting {i+1}/{total_iterations//buffer_saving} {label}")
            solve_and_save(niot_solver, label_dir)

        gc.collect()
        
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Reconstruct network')
    #parser.add_argument("--field", type=str, default='TOF', help="TOF")
    parser.add_argument("--c", type=int, default=1, help="coarseing factor")
    parser.add_argument("--n_ensemble", type=int, default=1, help="Number of processor per simulation")
    parser.add_argument("--mri", type=str, default="./mri/", help="directory with mri data")
    parser.add_argument("--out", type=str, default="./results_dirichlet/", help="output directory")
    parser.add_argument("--xmin", type=float, default=0.0, help="Lower bound x")
    parser.add_argument("--xmax", type=float, default=1000.0, help="Upper bound x")
    parser.add_argument("--ymin", type=float, default=0.0, help="Lower bound y")
    parser.add_argument("--ymax", type=float, default=1000.0, help="Upper bound y")
    parser.add_argument("--zmin", type=float, default=0.0, help="Lower bound z")
    parser.add_argument("--zmax", type=float, default=1000.0, help="Upper bound z")
    parser.add_argument("--threshold_network", type=float, default=250, help="Threshold for network")
    
    args, unknown = parser.parse_known_args()

    

    experiment(args)
