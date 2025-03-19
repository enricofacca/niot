import sys
import glob
import os
from copy import deepcopy as cp
import json 
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

from connected_components_tof import save_main_and_external_network_as_nifti

from niot.conductivity2image import HeatMap
import subprocess
from firedrake import *
from firedrake import COMM_WORLD
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

def product_dict(**kwargs):
    keys = kwargs.keys()
    for instance in itertools.product(*kwargs.values()):
        yield dict(zip(keys, instance))

def save_as_nifti(function, affine, filename):
    function_np = i2d.firedrake2numpy(function)
    nibabel.save(nibabel.Nifti1Image(function_np, affine), filename)
    function_np = None
    gc.collect()
    return


def build_meshes_from_numpy(data, mesh_type="simplicial",lengths=None, comm=COMM_WORLD, label_boundary=False):
    if lengths is None:
        lengths = [1.0,data.shape[1]/data.shape[0],data.shape[2]/data.shape[0]]
    
    mesh = i2d.build_mesh_from_numpy(data, mesh_type=mesh_type,lengths=lengths,comm=comm,label_boundary=label_boundary)
    if mesh_type == "simplicial":
        cartesian_mesh = i2d.cartesian_grid_3d(data.shape,lengths,comm=comm)
    else:
        cartesian_mesh = mesh
    return mesh, cartesian_mesh
    

def set_sink(option="segmented", **kargs):
    """
    Set sink
    """
    
    if option == "segmented":
        
        tof = kargs['tof']
        brain_mask = kargs['brain_mask']
        aseg = kargs['aseg']
        main_network = kargs['main_network'] 
        constant_absorption = kargs['constant_absorption']
        DG0 = tof.function_space()
        
        sink = Function(tof.function_space(), name="sink")
        sink.interpolate(- conditional(aseg > 0, constant_absorption,0)
                        * conditional(main_network > 0, 0, 1) ) # remove blood vessels outside the mask 
    else:
        raise ValueError(f"Unknown sink option {option}")

    return sink



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
    
    threshold_sink = 1e-10
    sink.interpolate(-conditional(brain_mask > threshold_sink, constant_absorption,0)
                     * conditional(corrupted>0,0,1)) # remove blood vessels outside the mask 
    

    mass_source = assemble(source*dx)
    mass_sink = assemble(sink*dx)

    #source /= mass_source
    #figu /= mass_sink
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
    


def labels(**kargs):
    """ 
    Set a list of labels for the experiment
    """
    
    label = []
    try:    
        wd = kargs['wd']
        label.append(f'wd{wd:.1e}')
    except:
        raise ValueError("wd not provided")
    
    try:
        ini = kargs["initial"]
        label.append(f"ini"+ini)
    except:
        raise ValueError("ini not provided")
    
    try:
        confidence = kargs["confidence"]
        label.append(f'conf'+confidence)
    except:
        raise ValueError("conf not provided")
    
    try:
        tdens2image = kargs['map']
        if tdens2image['type'] == 'identity':
            label.append(f'mapidentity')
        elif tdens2image['type'] == 'heat':
            label.append(f"mapheat{tdens2image['sigma']:.1e}")
        elif tdens2image['type'] == 'pm':
            label.append(f"mapipm{tdens2image['sigma']:.1e}")
        else:
            raise ValueError(f'Unknown tdens2image {tdens2image}')
        label.append(f"scaling{tdens2image['scaling']:.1e}")         
    except:
        raise ValueError("tdens2image not provided")

    try:
        absorption = kargs['absorption']
        label.append(f"sink{absorption:.1e}")
    except:
        pass

    try:
        kappa = kargs['kappa']
        label.append(f"kappa{kappa}")
    except:
        pass


    return label



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


def set_corrupted_network(**kargs):
    """
    Set corrupted network
    """
    tof = kargs['tof']
    main_network = kargs['main_network']
    external_network = kargs['external_network']
    brain_mask = kargs['brain_mask']
    threshold_network = kargs['threshold']

        
    DG0 = tof.function_space()
    corrupted = Function(DG0, name="corrupted")
    corrupted.interpolate(tof 
                        * conditional(external_network > 0, 0, 1) # exclude external network
                        * conditional(tof > threshold_network, 1, 0) 
                        * conditional(brain_mask > 1e-10, 1, 0) # only the main brain 
                        * conditional(main_network > 0, 0, 1) # remove main network
                        + tof 
                        * conditional(external_network > 0, 0, 1) # exclude external network
                        * conditional(tof > threshold_network, 1, 0)
                        * conditional(main_network > 0, 1, 0) # restore main network
    )
    
    return corrupted



#@profile
def experiment(args):
    results = args.out
    
    print(f"RUNNING {args.options}")
    # load options from json file
    try:
        with open(args.options, 'r') as f:
            options = json.load(f)
    except:
        options = {
            "threshold": [1e-3],
            "wd": [1e-4],  
            "initial": ["one"],
            "confidence": ["one"],
            "map": [{"type": "identity", "scaling": 10}],
            "kappa": ["one","t1"],
            "absorption": [1e-3]
            }
        # print as example
        for key, value in options.items():
            PETSc.Sys.Print(f"{key} : {value}")
        raise ValueError(f"File {args.options} not found")
        
    if len(options["threshold"]) > 1:
        raise ValueError("Only one threshold is allowed")
    threshold = options["threshold"][0]
    PETSc.Sys.Print(f"RUNNING {args.mri} {threshold:.1e}")

    # make directories
    mpi_mkdir(results)

    if args.mri[-1] == "/":
        input_name = os.path.basename(os.path.dirname(args.mri))
    else:
        input_name = os.path.basename(args.mri)
    test_case = f"{input_name}_threshold_{threshold:.1e}"
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

    #
    # check presence of data required
    #
    if COMM_WORLD.rank == 0:
        files = [
            f"{args.mri}/TOF.nii.gz",
                f"{args.mri}/T1.nii.gz",
                f"{args.mri}/inlets.nii.gz",
                f"{args.mri}/brain_mask.nii.gz",
                ]
        for file in files:
            if not os.path.exists(file):
                raise ValueError(f"File {file} not found")
        
        # threshold dependend files
        main_network_file = f"{args.mri}/main_network_t{threshold:.2e}.nii.gz"
        external_network_file = f"{args.mri}/external_network_t{threshold:.2e}.nii.gz"            
        if not os.path.exists(main_network_file) or not os.path.exists(external_network_file):
            PETSc.Sys.Print(f"Identifty main network and external network",end="")
            # this only use numpy, so we run only on one processor
            save_main_and_external_network_as_nifti(args.mri, threshold)
            PETSc.Sys.Print(f"- done")

    COMM_WORLD.barrier()


    # create ensemble of processors
    if args.n_ensemble == COMM_WORLD.size:
        my_ensemble = None
        comm = COMM_WORLD
        use_ensemble = False
    else:
        my_ensemble = Ensemble(COMM_WORLD, args.n_ensemble)
        comm = my_ensemble.comm
        use_ensemble = True
    
    #
    # check if h5 already exists or build it, but it may run out of memory
    #
    h5_file = f"{args.mri}/inputs_t{threshold:.2e}_nproc{args.n_ensemble}.h5"
    if os.path.exists(h5_file):
        PETSc.Sys.Print(f"Found checkpoint file {h5_file}")
    else:   
        PETSc.Sys.Print(f"Checkpoint not found. Creating it but we may run out of memory.\n"
                        f"Consider running mpiexec -n {args.n_ensemble} python build_checkpointfile.py "
                        )
        if use_ensemble:
            if my_ensemble.ensemble_comm.rank == 0:
                setup_h5(args.mri, threshold, comm=comm)
            my_ensemble.ensemble_comm.barrier()
        else:
            setup_h5(args.mri, threshold, comm=comm)
        PETSc.Sys.Print(f"Checkpoint created")


    #
    # load data
    #
    with CheckpointFile(h5_file, 'r',comm=comm) as afile:
        mesh = afile.load_mesh("mesh")
        PETSc.Sys.Print(f"mesh",end=" ")
        tof = afile.load_function(mesh, "tof")
        PETSc.Sys.Print(f"tof",end=" ")
        aseg = afile.load_function(mesh, "aseg")
        PETSc.Sys.Print(f"aseg",end=" ")
        brain_mask = afile.load_function(mesh, "brain_mask")
        PETSc.Sys.Print(f"brain mask",end=" ")
        t1 = afile.load_function(mesh, "t1")
        PETSc.Sys.Print(f"t1",end=" ")
        inlets = afile.load_function(mesh, "inlets")
        PETSc.Sys.Print(f"inlets",end=" ")
        main_network = afile.load_function(mesh, "main_network")
        PETSc.Sys.Print(f"main network",end="")
        external_network = afile.load_function(mesh, "external_network")
        PETSc.Sys.Print(f"external network",end=" ")
      
    PETSc.Sys.Print(f"Checkpoint loaded")
    if use_ensemble:
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
    
    input_data = { 
        "tof": tof, 
        "aseg": aseg, 
        "brain_mask": brain_mask, 
        "t1": t1, 
        "inlets": inlets, 
        "main_network": main_network, 
        "external_network": external_network,
        "cartesian_mesh": cartesian_mesh,
        "threshold": threshold
    }


    # confidence data
    def set_confidence(option, **kwargs):
        if option == "one":
            return Constant(1.0)            
        elif option == "main_network":
            PETSc.Sys.Print(f"Using main network as confidence")
            # get main network
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")

            confidence = Function(main_network.function_space(), name="confidence")
            confidence.interpolate(10 * conditional(main_network > 0, 1, 0))
            return confidence
        elif option == "main_plus_eps":
            PETSc.Sys.Print(f"Using main network as confidence")
            # get main network
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")

            confidence = Function(main_network.function_space(), name="confidence")
            confidence.interpolate(1e-6 + 10 * conditional(main_network > 0, 1, 0))
            return confidence
        else:
            raise ValueError(f"Unknown confidence option {option}")

    
    def set_kappa(option, **kwargs):
        """
        Set kappa function.
        In the region with high value of kappa the network passage is penalized.
        """
        if option == "one":
            return 1.0
        elif option == "t1":
            try:
                t1 = kwargs['t1']
            except:
                raise ValueError("t1 not provided")
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")

            kappa = Function(t1.function_space(), name="kappa")
            kappa.interpolate(1.0 
                              + conditional(main_network>0,0,1) 
                              * (
                                  conditional(t1 > 400, 5, 0)
                                  + conditional(t1 > 500, 5, 0)
                                ) )
            return kappa
        else:
            raise ValueError(f"Unknown kappa option {option}")
        
    def set_initial_guess(option, **kargs):        
        if option == "one":
            return Constant(1.0)

        if option == "low":
            try:
                corrupted = kargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            heat = HeatMap(space, scaling=1.0, sigma=1e1)
            low = Function(space, name="low", label=f"use heat map with sigma=1e1 and lift by 1e-4")
            low.assign(heat(corrupted+1e-4) + 1e-4)
            return low
        
        if option == "medium":
            try:
                corrupted = kargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            low = set_initial_guess("low", corrupted)
            medium = Function(space,name="MEDIUM")
            medium.assign(heat(low+1e-4) + 1e-4)
            return medium
        
        if option == "high":
            try:
                corrupted = kargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            medium = set_initial_guess("medium", corrupted)
            high = Function(space,name="HIGH")
            high.assign(heat(medium+1e-4) + 1e-4)
            return high

        if option == "low_gaussian":
            try:
                corrupted = kargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            
            corrupted_np = i2d.firedrake2numpy(corrupted)
            low_np = gaussian_filter(corrupted_np, sigma=4, truncate=1e0)
            low = i2d.numpy2firedrake(cartesian_mesh, low_np, name="LOW")
            low += 1e-4
            return low
        
        if option == "medium_gaussian":
            try:
                corrupted = kargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            low = set_initial_guess("low_gaussian", corrupted=corrupted)
            low_np = i2d.firedrake2numpy(low)
            medium_np = gaussian_filter(low_np, sigma=4, truncate=1e0)
            medium = i2d.numpy2firedrake(cartesian_mesh, medium_np, name="MEDIUM")
            medium += 1e-4
            return medium
        
        if option == "high_gaussian":
            try:
                corrupted = kargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            medium = set_initial_guess("medium_gaussian", corrupted=corrupted)
            medium_np = i2d.firedrake2numpy(medium)
            high_np = gaussian_filter(medium_np, sigma=4, truncate=1e0)
            high = i2d.numpy2firedrake(cartesian_mesh, high_np, name="HIGH")
            high += 1e-4
            return high
        
        else:
            raise ValueError(f"Unknown initial guess option {option}")
        



    combinations = list(product_dict(**options))


    test_poisson = False
    if test_poisson:
        btp = setup_btp(brain_mask, inlets, corrupted,  kappa, constant_absorption = 1.0)
        poisson(cartesian_mesh, btp)
        exit()


    
    
    # divide the combinations in the ensemble
    def lol(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    
    if use_ensemble:
        sub_combinations = list(lol(combinations, my_ensemble.ensemble_comm.size))
        todo = sub_combinations[my_ensemble.ensemble_comm.rank]
        for i in range(my_ensemble.ensemble_comm.size):
            if i == my_ensemble.ensemble_comm.rank:
                print(f"ENSEMBLE {i}")
                for j, comb in enumerate(todo):
                    label = "_".join(labels(**comb))
                    if my_ensemble.comm.rank == 0:
                        print(f"{j} {label}")   
            my_ensemble.ensemble_comm.barrier()

        my_ensemble.ensemble_comm.barrier()
        PETSc.Sys.Print(f"{my_ensemble.ensemble_comm.rank=} {len(todo)=}")
    else:
        todo = combinations
        PETSc.Sys.Print(f"TODO {len(todo)=}")
    
    


    
    for i, combination in enumerate(todo):
        # set label and directory
        label = "_".join(labels(**combination))
        if use_ensemble:
            PETSc.Sys.Print(f"{i} {my_ensemble.ensemble_comm.rank=} {label}")
        else:
            PETSc.Sys.Print(f"{i} {label}")

        label_dir = os.path.join(out_directory,label)
        mpi_mkdir(label_dir, comm)
        
        # get git version used 
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        combination4save = cp(combination)
        combination4save["git_hash"] = git_hash
        # save a copy of current combination as json
        with open(f"{label_dir}/option.json", 'w') as f:
            json.dump(combination4save, f, indent=4)

        # print combination
        for key, value in combination.items():
            PETSc.Sys.Print(f"{key} : {value}")


        #
        # set corrupted network
        #
        corrupted = set_corrupted_network(**input_data)
        
        #
        # btp inputs
        #
        absortion = combination["absorption"]
        sink = set_sink(option="segmented", **input_data, constant_absorption = absortion)

        R = FunctionSpace(cartesian_mesh,"R",0)
        source = Function(R, name="source")
        source.assign(0.0)

        kappa = set_kappa(combination["kappa"], **input_data)


        inlet_pressure = Function(inlets.function_space())
        inlet_pressure.assign(0.0)
        weak_Dirichlet = [(inlet_pressure, ds_b, inlets)]
        #strong_Dirichlet = [(source, ds_b, inlets)]
        btp = ot.BranchedTransportProblem(source, sink, 
                                      gamma=0.5, 
                                      Dirichlet = None,
                                      weak_Dirichlet = weak_Dirichlet,
                                      kappa=kappa)

        #
        # set confidence
        #
        confidence = set_confidence(combination["confidence"], **input_data)
        

        # setup solver
        niot_solver = NiotSolver(btp, 
                             corrupted,  
                             confidence=confidence, 
                             spaces = "DG0DG0",
                             cell2face = 'harmonic_mean',
                             setup = False,
                             ensemble_comm=None
                             )
        # inpainting
        wd = combination["wd"]
        niot_solver.ctrl_set('discrepancy_weight', wd)
        niot_solver.ctrl_set('regularization_weight', 0.0)
        tdens2image = combination["map"]
        niot_solver.ctrl_set(['tdens2image'], tdens2image)

        # optimization
        niot_solver.ctrl_set('optimization_tol', 1e-5)
        niot_solver.ctrl_set('constraint_tol', 1e-6)
        try: 
            max_iter = combination["max_iter"]
        except:
            max_iter = 5000
            
        niot_solver.ctrl_set('max_iter', max_iter)
        
        niot_solver.ctrl_set('max_restart', 4)
        niot_solver.ctrl_set('verbose', 1)

        # time discretization
        method = "tdens_mirror_descent_explicit"
        niot_solver.ctrl_set(['dmk','type'], method)

        # time step
        deltat_control = {
        'type': 'adaptive2',
        'lower_bound': 1e-13,
        'upper_bound': 5e-2,
        'expansion': 1.1,
        'contraction': 0.5,
        }
        niot_solver.ctrl_set(['dmk',method,'deltat'], deltat_control)
        
        # setup log file
        log_filename = os.path.join(label_dir,f"niot.log")
        niot_solver.ctrl_set("log_file",log_filename)
        
        # set solvers according to controls
        niot_solver.setup()
        
        # set intial guess
        initial = set_initial_guess(combination["initial"], corrupted=corrupted)
        niot_solver.set_solution(tdens=initial)
        
        save_inputs = True
        if save_inputs:
            filename = f"{label_dir}/corrupted.nii.gz"
            PETSc.Sys.Print(f"Saving {filename}")
            save_as_nifti(corrupted, affine, filename)

            filename = f"{label_dir}/sink.nii.gz"
            PETSc.Sys.Print(f"Saving {filename}")
            save_as_nifti(sink, affine, filename)

            if combination["initial"] != "one":
                filename = f"{label_dir}/initial.nii.gz"
                PETSc.Sys.Print(f"Saving {filename}")
                save_as_nifti(initial, affine, filename)


            if combination["confidence"] != "one":
                filename = f"{label_dir}/confidence.nii.gz"
                PETSc.Sys.Print(f"Saving {filename}")
                confidence_np = i2d.firedrake2numpy(confidence)
                nibabel.save(nibabel.Nifti1Image(confidence_np, affine), filename)
                confidence_np = None

            if combination["kappa"] != "one":
                filename = f"{label_dir}/kappa.nii.gz"
                PETSc.Sys.Print(f"Saving {filename}")
                kappa_np = i2d.firedrake2numpy(btp.kappa)
                nibabel.save(nibabel.Nifti1Image(kappa_np, affine), filename)
                kappa_np = None
                gc.collect()
            
        
        
        
        #
        # run solver, buffering the saving of the solution
        #
        total_iterations = niot_solver.ctrl_get('max_iter')
        buffer_saving = min(500,total_iterations)


        def solve_and_save(niot_solver, label_dir, n_buffer):
            n_iter = niot_solver.ctrl_get('max_iter')
            PETSc.Sys.Print(f"TODO {n_iter}")
            
            # solve
            ierr = niot_solver.solve()

            # save solution
            pot, tdens, vel = niot_solver.get_otp_solution(niot_solver.sol)
            
            tdens_np = i2d.firedrake2numpy(tdens)
            tdens = None
            filename=f"{label_dir}/tdens_{n_buffer}.nii.gz"
            nibabel.save(nibabel.Nifti1Image(tdens_np, affine), filename)
            tdens_np = None
            gc.collect()
            
            pot_np = i2d.firedrake2numpy(pot)
            pot = None
            filename=f"{label_dir}/pot_{n_buffer}.nii.gz"
            PETSc.Sys.Print(f"Saving {filename} ")
            nibabel.save(nibabel.Nifti1Image(pot_np, affine), filename)
            pot_np = None
            gc.collect()
            

        # run and save
        niot_solver.ctrl_set('max_iter', total_iterations%buffer_saving)
        solve_and_save(niot_solver, label_dir, 0)
        PETSc.Sys.Print(f"First {total_iterations%buffer_saving} iterations done")

        # run and save, skipping initial steps
        niot_solver.ctrl_set('max_iter',buffer_saving)
        niot_solver.ctrl_set('restart',True)
        for i in range(total_iterations//buffer_saving):
            interval = [i*buffer_saving,(i+1)*buffer_saving]
            PETSc.Sys.Print(f"Starting {interval[0]} {interval[1]} of {total_iterations:.1f} - {label}")
            solve_and_save(niot_solver, label_dir, i+1)
            

        gc.collect()
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(exit_on_error=True, description='Reconstruct network')
    #parser.add_argument("--field", type=str, default='TOF', help="TOF")
    parser.add_argument("--n_ensemble", type=int, default=1, help="Number of processor per simulation")
    parser.add_argument("--mri", type=str, default="./mri/", help="directory with mri data")
    parser.add_argument("--out", type=str, default="./runs/", help="output directory")
    parser.add_argument("--options", type=str, default="options.json", help="Json file with controls")
    
    args, unknown = parser.parse_known_args()

    

    experiment(args)
