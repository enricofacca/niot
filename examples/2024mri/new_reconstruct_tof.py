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
from build_checkpointfile import setup_h5
#from memory_profiler import profile

from niot.conductivity2image import HeatMap

from firedrake import *
from scipy.ndimage import zoom
import time
from firedrake import VTKFile as File

from firedrake.petsc import PETSc
from firedrake import CheckpointFile

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
    

def setup_btp(brain_mask, inlets, corrupted, kappa, constant_absorption = 1):
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
           gamma,
           wd,
           wr,
           ini,
           confidence,
           tdens2image, 
           method,
           absortion):
    label= [
        f'fem{fem}',
        f'gamma{gamma:.1e}',
        f'wd{wd:.1e}',
        f'wr{wr:.1e}',
        f'ini'+ini.name(),
        f'conf'+confidence.name(),]
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
                 confidence=1.0,
                 tdens2image={
                     'type':'identity',
                     'scaling':1e0
                 },
                 method='tdens_mirror_descent_explicit'  ,
                 ):

    print(method)
   
    mesh = btp.source.function_space().mesh()

    niot_solver = NiotSolver(btp, 
                             corrupted,  
                             confidence=confidence, 
                             spaces = fem,
                             cell2face = 'harmonic_mean',
                             setup=False,
                             ensemble_comm=None,
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
    print(method)
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

def indices_restrict(data_shape, lengths, xyz_bounds):
    """ 
    Assuming that LX=1
    """
    indices_bounds = []
    for axis_index in range(len(data_shape)):
        n_axis = data_shape[axis_index]
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
    if cartesian_mesh.geometric_dimension() == 3:
        hypre_ctrl_3d = {
                        # tuning parameters for the multigrid
                        # https://mooseframework.inl.gov/releases/moose/2021-09-15/application_development/hypre.html
                        "pc_hypre_type": "boomeramg",
                        "pc_hypre_boomeramg_strong_threshold": 0.75,
                        "pc_hypre_boomeramg_max_iter": 1,
                        "pc_hypre_boomeramg_agg_nl": 3,
                        "pc_hypre_boomeramg_interp_type": "ext+i",  # "classic" or "ext+i"
                    }
        petsc_controls.update(hypre_ctrl_3d)


    problem = LinearVariationalProblem(a, L, pot_h, bcs=bcs)
    solver = LinearVariationalSolver(problem, 
                                     solver_parameters=petsc_controls, 
                                     nullspace=nullspace)
    
    # solve problem
    solver.solve()
    #solver.snes.ksp.view()
    ksp_iterations = solver.snes.ksp.getIterationNumber()

    PETSc.Sys.Print(f"Number of iterations: {ksp_iterations}")

    # save solution to pvd
    out_file = File(f'pot.pvd')
    out_file.write(pot_h)


def set_corrupted_network(tof, main_network, external_network, brain_mask, threshold_network):
    DG0 = tof.function_space()
    corrupted = Function(DG0, name="corrupted")
    corrupted.interpolate(tof 
                        * conditional( tof > threshold_network, 1, 0) 
                        * conditional(brain_mask > 1e-10, 1, 0) # only the main brain 
                        * conditional(main_network > 0, 0, 1) # remove main network
                        + tof 
                        * conditional( tof > threshold_network, 1, 0)
                        * conditional(main_network > 0, 1, 0) # restore main network
    )
    
    # remove vessel going and out
    in_and_out = Function(DG0, name="in_and_out")
    in_and_out.interpolate(tof*conditional(external_network>0, 1, 0))
    corrupted -= in_and_out

    return corrupted



#@profile
def experiment(args):
    results = args.out
    threshold_network = args.threshold
    threshold_tof = threshold_network

    PETSc.Sys.Print(f"RUNNING {args.mri} {threshold_network:.1e}")

    # make directories
    mpi_mkdir(results)

    if args.mri[-1] == "/":
        input_name = os.path.basename(os.path.dirname(args.mri))
    else:
        input_name = os.path.basename(args.mri)
    test_case = f"{input_name}_threshold_{args.threshold:.1e}"
    PETSc.Sys.Print(f"RUNNING {test_case}")
    
    
    tof_data = nibabel.load(f"{args.mri}/TOF.nii.gz")
    affine = tof_data.affine
    original_dimensions = tof_data.header.get_data_shape()[:3]
    hx, hy, hz = tof_data.header['pixdim'][1:4]
    lengths = np.array([float(original_dimensions[0]*hx), 
                        float(original_dimensions[1]*hy), 
                        float(original_dimensions[2]*hz)])


    out_directory = results + test_case
    mpi_mkdir(out_directory)
    my_ensemble = Ensemble(COMM_WORLD, args.n_ensemble)
    
    
    # only the first ensemble needs to read the data
    # if my_ensemble.ensemble_comm.rank == 0:
    # check if h5 already exists or build it, but it may run out of memory
    h5_file = f"{args.mri}/inputs_t{args.threshold:.2e}_nproc{args.n_ensemble}.h5"
    PETSc.Sys.Print(f"{h5_file}")
    
    if os.path.exists(h5_file):
        PETSc.Sys.Print(f"Found checkpoint file {h5_file}")
    else:   
        PETSc.Sys.Print(f"Checkpoint not found. Creating it but we may run out of memory")
        if my_ensemble.ensemble_comm.rank == 0:
            setup_h5(args.mri, args.threshold, comm=my_ensemble.comm)
        my_ensemble.ensemble_comm.barrier()
        PETSc.Sys.Print(f"checkpoint created")

    with CheckpointFile(h5_file, 'r',comm=my_ensemble.comm) as afile:
        mesh = afile.load_mesh("mesh")
        PETSc.Sys.Print(f"mesh loaded")
        tof = afile.load_function(mesh, "tof")
        PETSc.Sys.Print(f"tof loaded")
        brain_mask = afile.load_function(mesh, "brain_mask")
        PETSc.Sys.Print(f"brain mask loaded")
        main_network = afile.load_function(mesh, "main_network")
        PETSc.Sys.Print(f"main network loaded")
        t1 = afile.load_function(mesh, "t1")
        PETSc.Sys.Print(f"t1 loaded")
        external_network = afile.load_function(mesh, "external_network")
        PETSc.Sys.Print(f"external network loaded")
        inlets = afile.load_function(mesh, "inlets")
        PETSc.Sys.Print(f"inlets loaded")
    PETSc.Sys.Print(f"Checkpoint loaded")
    my_ensemble.ensemble_comm.barrier()

    mesh.nx = original_dimensions[0]
    mesh.ny = original_dimensions[1]
    mesh.nz = original_dimensions[2]
    mesh.xmin = 0.0
    mesh.xmax = lengths[0]
    mesh.ymin = 0.0
    mesh.ymax = lengths[1]
    mesh.zmin = 0.0
    mesh.zmax = lengths[2]
    


    cartesian_mesh = mesh
    


    # confidence data
    confidence = Function(tof.function_space(), name="confidence")
    confidence.interpolate(1.0 + 9.0 * conditional(main_network > 0, 1, 0))
        
        
    # kappa definition
    kappa = Function(tof.function_space(), name="kappa")
    kappa.interpolate(1.0 + conditional(t1 > 600, 1, 0))

    # define the corrupted network
    corrupted = set_corrupted_network(tof, main_network, external_network, brain_mask, threshold_network)

    space = corrupted.function_space()
    
    # initial guess
    heat_flow = True
    sigma0 = 1.0
    base = 4
    min_tdens = 1e-4

    initial_guess = False
    if initial_guess:
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
        conf = [one, confidence]
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
        btp = setup_btp(brain_mask, inlets, corrupted,  kappa, constant_absorption = 1.0)
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
        btp = setup_btp(brain_mask, inlets, corrupted, kappa, constant_absorption = absortion)

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
        niot_solver = setup_solver( btp, corrupted, *combination[:-1])
        
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
        buffer_saving = min(100,total_iterations)


        def solve_and_save(niot_solver, label_dir):
            ierr = niot_solver.solve()

            # save solution
            pot, tdens, vel = niot_solver.get_otp_solution(niot_solver.sol)

            tdens_np = i2d.firedrake2numpy(tdens)
            tdens = None

            pot_np = i2d.firedrake2numpy(pot)
            pot = None


            # save tdens and pot as npy files
            if my_ensemble.comm.rank == 0:
                filename=f"{label_dir}/tdens.nii.gz"
                nibabel.save(nibabel.Nifti1Image(tdens_np, affine), filename)
                filename=f"{label_dir}/pot.nii.gz"
                nibabel.save(nibabel.Nifti1Image(pot_np, affine), filename)
            
            #np.save(f"{label_dir}/tdens.nii.gz",tdens_np)
            #np.save(f"{label_dir}/pot.nii",pot_np)
        
            save_outputs_as_vtr = False
            if save_outputs_as_vtr:
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
    parser.add_argument("--out", type=str, default="./runs/", help="output directory")
    parser.add_argument("--xmin", type=float, default=0.0, help="Lower bound x")
    parser.add_argument("--xmax", type=float, default=1000.0, help="Upper bound x")
    parser.add_argument("--ymin", type=float, default=0.0, help="Lower bound y")
    parser.add_argument("--ymax", type=float, default=1000.0, help="Upper bound y")
    parser.add_argument("--zmin", type=float, default=0.0, help="Lower bound z")
    parser.add_argument("--zmax", type=float, default=1000.0, help="Upper bound z")
    parser.add_argument("--threshold", type=float, default=250, help="Threshold for network")
    
    args, unknown = parser.parse_known_args()

    

    experiment(args)
