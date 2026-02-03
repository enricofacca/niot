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
from build_checkpointfile import setup_h5, write_h5, nii2firedrake
#from memory_profiler import profile

import warnings
warnings.filterwarnings("ignore")


from connected_components_tof import save_main_and_external_network_as_nifti
from firedrake.__future__ import interpolate
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

def transfer_to_cartesian(f, interpolator, interpolate_fun):
    interpolate_fun.interpolate(f)
    f_cartesian = assemble(interpolator)
    return f_cartesian

class BluringOperator:
    def __init__(self, mesh):
        self.mesh = mesh
        self.R = FunctionSpace(mesh, "R", 0)
        self.sigma = Function(self.R, name="sigma_blur")
        if mesh.ufl_cell().is_simplex():
            cg1 = FunctionSpace(mesh, "CG", 1)
            self.blurred = Function(cg1, name="blurred")
            self.rhs_function = Function(cg1, name="rhs_blur")
            test = TestFunction(cg1)  
            trial = TrialFunction(cg1)
            
            # setup heat equation solver
            self.lhs_form =  inner(test, trial) * dx # mass matrix
            self.lhs_form += self.sigma * inner(grad(test), grad(trial)) * dx # stiffness matrix

            # 1-form for the heat equation
            self.rhs_form = self.rhs_function * test * dx
            self.heat_problem = LinearVariationalProblem(self.lhs_form, 
                                                    self.rhs_form,
                                                    self.blurred)
                    
            self.heat_solver = LinearVariationalSolver(
                self.heat_problem,
                solver_parameters={
                    'ksp_type': 'cg',
                    'ksp_rtol': 1e-10,
                    #'ksp_initial_guess_nonzero': True,
                    #'ksp_monitor_true_residual': None,
                    'pc_type': 'hypre',
                    },
                options_prefix='blur_solver_')
    def __call__(self, function, sigma):
        if sigma <= 0:
            return function
        mesh = function.function_space().mesh()
        if mesh.ufl_cell().is_simplex():
            PETSc.Sys.Print("Using heat equation for simplicial mesh")
            self.sigma.assign(sigma)
            self.rhs_function.interpolate(function)
            self.heat_solver.solve()
            out = assemble(interpolate(self.blurred, function.function_space()))
            return out
        else:
            function_np = i2d.firedrake2numpy(function)
            function_np_blurred = gaussian_filter(function_np, sigma)
            function_blurred = i2d.numpy2firedrake(mesh, function_np_blurred, name=function.name()+"_blurred")
            function_np = None
            function_np_blurred = None
            gc.collect()
            return function_blurred

def save_as_nifti(function, filename, affine, dimensions, lenghts, offset):
    mesh = function.function_space().mesh()
    # We need to interpolate to a cartesian grid
    if mesh.ufl_cell().is_simplex():
        function_np = i2d.anyfiredrake2numpy(function, dimensions, lenghts, offset, fill=-1e30)
    else:
        function_np = i2d.firedrake2numpy(function)
    if mesh.comm.rank == 0:    
        nibabel.save(nibabel.Nifti1Image(function_np, affine), filename)
        function_np = None
        gc.collect()
    mesh.comm.barrier()
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
    

def set_sink(option_type="segmented", **kargs):
    """
    Set sink
    """ 
    
    if option_type == "segmented":    
        tof = kargs['tof']
        brain_mask = kargs['brain_mask']
        aseg = kargs['aseg']
        main_network = kargs['main_network'] 
        absorption = kargs['absorption']
        DG0 = tof.function_space()
        
        def indicator_regions(labels,aseg):
            """
            Return an expression 
            """
            indicator = 0.0
            eps = 1.0e-4
            for label in labels:
                indicator += (conditional(aseg > label-eps, 1, 0) 
                            *conditional(aseg < label+eps, 1, 0))
            return indicator        

        # get the sink
        sink = Function(tof.function_space(), name=f"sink{absorption:.1e}")
        indicator_empty = indicator_regions([4,5,14,15,24,43,44],aseg)
        sink.interpolate(- conditional(aseg > 0, absorption,0)
                        * conditional(main_network > 0, 0, 1)
                        * (1-indicator_empty)) # remove blood vessels outside the mask 
    if option_type == "sink_support":
        sink_support = kargs['sink_support']
        absorption = kargs['absorption']
        sink = Function(sink_support.function_space(), name=f"sink{absorption:.1e}")
        sink.assign(- absorption * sink_support)

    else:
        raise ValueError(f"Unknown sink option {option_type}")

    return sink

    


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
        if isinstance(confidence, str):
            label.append(f'conf'+confidence)
        elif isinstance(confidence, dict):
            confidence_type = confidence['type']
            if confidence_type == "main_plus_eps":
                try:
                    eps = confidence['eps']
                except:
                    raise ValueError("eps_confidence not provided")
                label.append(f'confmain_plus_eps{eps:.1e}')
        else:
            raise ValueError(f"Unknown confidence type {confidence_type}")
        
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


def set_corrupted_network(**kwargs):
    """
    Set corrupted network
    """
    try:
        option = kwargs["corrupted"]
    except:
        raise ValueError("corrupted not provided")

    if isinstance(option, str):
        option_type = option
    elif isinstance(option, dict):
        option_type = option["type"]
    else:
        raise ValueError(f"Unknown confidence type {option}")

    common_name = "OBS"


    tof = kwargs['tof']
    main_network = kwargs['main_network']
    external_network = kwargs['external_network']
    brain_mask = kwargs['brain_mask']
    
    if option_type == "tof":
        try:
            threshold_tof = option['threshold_tof']
        except:
            threshold_tof = 200

        name = f"{common_name}tof_t{threshold_tof:.2e}"

        try: 
            blur = option['blur']
        except:
            blur = 0.0
        
        if blur > 0:
            blurer = kwargs["blurer"]
            hx = kwargs["voxel_size"][0]
            tof4corrupted = blurer(tof, sigma=blur*hx)            
        else:
            tof4corrupted = tof


        DG0 = tof.function_space()
        corrupted = Function(DG0, name=name)
        corrupted.interpolate(
                            tof4corrupted * conditional(main_network > 0, 1, 0) # the main we must fit
                            + 
                            tof4corrupted * conditional(main_network > 0, 0, 1) # the rest
                            * conditional(brain_mask > 1e-10, 1, 0) #within the brain
                            * conditional(tof4corrupted > threshold_tof, 1, 0) # only values above the threshold
                            * conditional(external_network > 0, 0, 1) # exclude external network
                            )
        tof4corrupted = None
        gc.collect()
    
    elif option_type == "support":
        try:
            threshold_tof = option['threshold_tof']
        except:
            threshold_tof = 200

        name = f"{common_name}support_t{threshold_tof:.2e}"

        try: 
            blur = option['blur']
        except:
            blur = 0.0
        
        if blur > 0:
            blurer = kwargs["blurer"]
            hx = kwargs["voxel_size"][0]
            tof4corrupted = blurer(tof, sigma=blur*hx)
        else:
            tof4corrupted = tof


        DG0 = tof.function_space()
        corrupted = Function(DG0, name=name)
        corrupted.interpolate(
                            conditional(tof4corrupted > threshold_tof, 1, 0)
                            * (  conditional(main_network > 0, 1, 0) # the main we must fit
                                + 
                                conditional(main_network > 0, 0, 1) # the rest
                                * conditional(brain_mask > 1e-10, 1, 0) #within the brain
                                * conditional(tof4corrupted > threshold_tof, 1, 0) # only values above the threshold
                                * conditional(external_network > 0, 0, 1) # exclude external network
                                )
                            )   

    elif option_type == "load":
        try:
            path = option['path']
        except:
            raise ValueError("path initial tdens to provided")
        
        try:
            mesh = kwargs["mesh"]
        except:
            raise ValueError("mesh not provided")
        corrupted = nii2firedrake(path, mesh, name=common_name+"load",comm=mesh.comm)

        try:
            scaling = option['scaling']
        except:
            scaling = 1.0
        corrupted *= scaling

        try:
            lift = option['lift']
        except:
            scaling = 0.0
        corrupted += lift


    else:
        raise ValueError(f"Unknown corrupted option {option}")

    return corrupted



#@profile
def experiment(args):
    results = args.out
    
    # load options from json file
    #try:
    with open(args.options, 'r') as f:
        options = json.load(f)
    # except:
    #     options = {
    #         "blur": [0.0],
    #         "threshold": [1e-3],
    #         "wd": [1e-4],  
    #         "initial": ["one"],
    #         "confidence": ["one"],
    #         "map": [{"type": "identity", "scaling": 10}],
    #         "kappa": ["one","t1"],
    #         "absorption": [1e-3]
    #         }
    #     # print as example
    #     for key, value in options.items():
    #         PETSc.Sys.Print(f"{key} : {value}")
    #     raise ValueError(f"File {args.options} not found")
        
    if len(options["threshold"]) > 1:
        raise ValueError("Only one threshold is allowed")
    threshold = options["threshold"][0]

    try:    
        masked_mesh = options["mask_mesh"][0] > 0
    except:
        PETSc.Sys.Print(f"Error reading. Using full grid")
        masked_mesh = False
        options["mask_mesh"] = [masked_mesh]
    
    try:    
        blur = options["blur"][0]
    except:
        blur = 0.0
        options["blur"] = [blur]
    

    PETSc.Sys.Print(f"**** SETUP ****** ")
    PETSc.Sys.Print(f"Inputs: {args.mri}")
    PETSc.Sys.Print(f"Options:")
    for key, value in options.items():
        PETSc.Sys.Print(f"{key} : {value}")
    PETSc.Sys.Print(f"PID: {os.getpid()}")
    git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
    PETSc.Sys.Print(f"GIT has: {git_hash}")
    PETSc.Sys.Print(f"nproc: {COMM_WORLD.size} n_ensemble: {args.n_ensemble}")
    PETSc.Sys.Print(f"********************* ")
    PETSc.Sys.Print(f" ")
    
    # make directories
    mpi_mkdir(results)

    if args.mri[-1] == "/":
        input_name = os.path.basename(os.path.dirname(args.mri))
    else:
        input_name = os.path.basename(args.mri)
    test_case = f"{input_name}_threshold_{threshold:.2e}"
        
    tof_data = nibabel.load(f"{args.mri}/TOF.nii.gz")
    affine = tof_data.affine
    original_dimensions = tof_data.header.get_data_shape()[:3]
    hx, hy, hz = tof_data.header['pixdim'][1:4]
    dimensions = original_dimensions
    
    voxel_size = np.array([hx, hy, hz])
    lengths = np.array([hx, hy, hz]) * np.array(original_dimensions)

    tof_data = None
    gc.collect()



    out_directory = results + test_case
    mpi_mkdir(out_directory)

    


    # create ensemble of processors
    if args.n_ensemble == COMM_WORLD.size:
        my_ensemble = None
        comm = COMM_WORLD
        use_ensemble = False
        color_rank = 0
    else:
        my_ensemble = Ensemble(COMM_WORLD, args.n_ensemble)
        comm = my_ensemble.comm
        use_ensemble = True
        color_rank = my_ensemble.ensemble_comm.rank
    
    #
    # check if h5 already exists or build it, but it may run out of memory
    #
    if args.h5 != "":
        #h5_file = f"{args.mri}/inputs_{label}_nproc{args.n_ensemble}.h5"
        h5_file = args.h5
        if os.path.exists(h5_file):
             PETSc.Sys.Print(f"Found checkpoint file {h5_file}")
        else:   
             PETSc.Sys.Print(f"Checkpoint not found {h5_file}. Creating it but we may run out of memory.\n")
        #                     f"Consider running mpiexec -n {args.n_ensemble} python build_checkpointfile.py "
        #                     )
        #     if use_ensemble:
        #         if my_ensemble.ensemble_comm.rank == 0:
        #             data = setup_h5(args.mri, threshold, blur=blur, masked_mesh=masked_mesh, comm=comm)
        #             write_h5(args.mri, threshold, blur, comm, args.n_ensemble, data=data)
        #         my_ensemble.ensemble_comm.barrier()
        #     else:
        #         data = setup_h5(args.mri, threshold, blur=blur, masked_mesh=masked_mesh, comm=comm)
        #         write_h5(args.mri, threshold, blur, comm, args.n_ensemble, data=data)
        #     PETSc.Sys.Print(f"Checkpoint created h5_file={h5_file}")
        

        #
        # load data
        #
        with CheckpointFile(h5_file, 'r',comm=comm) as afile:
            mesh = afile.load_mesh("relabeled_mesh")
            PETSc.Sys.Print(f"mesh",end=" ")
            tof = afile.load_function(mesh, "tof")
            PETSc.Sys.Print(f"tof",end=" ")
            try:
                aseg = afile.load_function(mesh, "aseg")
                PETSc.Sys.Print(f"aseg",end=" ")
            except:
                PETSc.Sys.Print(f"No aseg found",end=" ")
                aseg = None
            
            brain_mask = afile.load_function(mesh, "brain_mask")
            PETSc.Sys.Print(f"brain mask",end=" ")
            
            t1 = afile.load_function(mesh, "t1")
            PETSc.Sys.Print(f"t1",end=" ")
            
            try:
                inlets = afile.load_function(mesh, "inlets")
                PETSc.Sys.Print(f"inlets",end=" ")
            except:
                PETSc.Sys.Print(f"No inlets found",end=" ")
                inlets = None

            try:
                sink_support = afile.load_function(mesh, "sink_support")
                PETSc.Sys.Print(f"sink_support",end=" ")
            except:
                PETSc.Sys.Print(f"No sink_support found",end=" ")
                sink_support = None

            
            main_network = afile.load_function(mesh, "main_network")
            PETSc.Sys.Print(f"main network",end="")
            skeleton = afile.load_function(mesh, "skeleton")
            PETSc.Sys.Print(f"skeleton",end="")
            thickness = afile.load_function(mesh, "thickness")
            PETSc.Sys.Print(f"thickness",end=" ")
            affine = afile.get_attr("/info/", "affine")
            PETSc.Sys.Print(f" affine ")
            offset = afile.get_attr("/info/", "offset")
            PETSc.Sys.Print(f" offeet")
            voxel_size = afile.get_attr("/info/", "voxel_size")
            PETSc.Sys.Print(f" voxel_size ")
            dimensions = afile.get_attr("/info/", "dimensions")
            PETSc.Sys.Print(f" dimensions ")
            external_network = Function(main_network.function_space(), name="external_network")
            external_network.assign(0.0)
            cartesian_mesh = i2d.cartesian_grid_3d(dimensions, lengths, offset=offset, comm=comm)
        
        PETSc.Sys.Print(f"Checkpoint loaded")
        PETSc.Sys.Print(f"**** Inputs loaded ****")
        PETSc.Sys.Print(f"")
        if use_ensemble:
            my_ensemble.ensemble_comm.barrier()

    else:
        #
        # check presence of data required
        #
        if COMM_WORLD.rank == 0:
            files = [
                f"{args.mri}/TOF.nii.gz",
                    f"{args.mri}/T1.nii.gz",
                    f"{args.mri}/brain_mask_smooth.nii.gz",
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

        PETSc.Sys.Print(f"**** Inputs loading ****")
        if blur > 0:
            label = f"t{threshold:.2e}_blur{blur:.2e}"  
        else: 
            label = f"t{threshold:.2e}"

        COMM_WORLD.barrier()
        data = setup_h5(args.mri, threshold, blur=blur, masked_mesh=masked_mesh, comm=comm)
        cartesian_mesh, tof, aseg, t1, brain_mask, main_network, external_network, inlets, skeleton, thickness, sink_support = data
        
        mesh = cartesian_mesh

    try:
        xmin = mesh.xmin
        offset = [mesh.xmin, mesh.ymin, mesh.zmin]
        PETSc.Sys.Print(f"Mesh dimensions: nx={mesh.nx}, ny={mesh.ny}, nz={mesh.nz}")
        lower, upper = mesh.bounding_box()
        PETSc.Sys.Print(f"Mesh bounds: x[{lower[0]:.2e}, {upper[0]:.2e}], "
                        f"y[{lower[1]:.2e}, {upper[1]:.2e}], "
                        f"z[{lower[2]:.2e}, {upper[2]:.2e}]")
    except:
        mesh.nx = original_dimensions[0]
        mesh.ny = original_dimensions[1]
        mesh.nz = original_dimensions[2]
        mesh.xmin = offset[0]
        mesh.xmax = offset[0]+lengths[0]
        mesh.ymin = offset[1]
        mesh.ymax = offset[1]+lengths[1]
        mesh.zmin = offset[2]
        mesh.zmax = offset[2]+lengths[2]
        mesh.hx = hx
        mesh.hy = hy
        mesh.hz = hz
        mesh.invert_rows_columns = False
    
    blurer = BluringOperator(mesh)

    input_data = { 
        "tof": tof, 
        "aseg": aseg, 
        "brain_mask": brain_mask, 
        "t1": t1, 
        "sink_support": sink_support,
        "inlets": inlets, 
        "main_network": main_network, 
        "external_network": external_network,
        "mesh": mesh,
        "cartesian_mesh": cartesian_mesh,
        "skeleton" : skeleton,
        "thickness": thickness,        
        "affine": affine,
        "offset": offset,
        "voxel_size": voxel_size,
        "dimensions": dimensions,
        "blurer": blurer
    }

    if mesh.ufl_cell().is_simplex():
        defualt_spaces = ["CG1DG0"]
    else:
        defualt_spaces = ["DG0DG0"]

    # check compatibility of mesh and spaces
    spaces = options.get("spaces", defualt_spaces)[0]
    
    if spaces == "CG1DG0" and (not mesh.ufl_cell().is_simplex()):
        raise ValueError(f"Simplicial mesh requires CG1 space, got {spaces}")
    if spaces == "DG0DGO" and mesh.ufl_cell().is_simplex():
        raise ValueError(f"Cartesian mesh requires DG0 space, got {spaces}")

    
    # confidence data
    def set_confidence(**kwargs):
        try:
            option = kwargs["confidence"]
        except:
            raise ValueError("confidence not provided")
    
        if isinstance(option, str):
            option_type = option
        elif isinstance(option, dict):
            option_type = option["type"]
        else:
            raise ValueError(f"Unknown confidence type {option}")
    
        # assign keyword for common name
        common_name = "conf"

        #
        # select option
        #

        
        if option == "one":
            try:
                mesh = kwargs['mesh']
            except:
                raise ValueError("mesh not provided")
            one = Function(FunctionSpace(mesh,"R",0), name=common_name+"one")
            one.assign(1.0)
            return one
        
        elif option_type == "main_network":
            # get main network
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")
            
            # get brain mask
            try:
                brain_mask = kwargs['brain_mask']
            except:
                raise ValueError("brain_mask not provided")

            confidence = assemble(interpolate( # inside, we trust the network
                                    100  * conditional(main_network > 0, 1, 0)
                                   # outside, strong confidence, where we set no network
                                   + 100 * conditional(brain_mask<=1e-16, 1, 0),
                                   main_network.function_space())
                                   )
            confidence.rename(f"{common_name}main_network")
            return confidence
        
        elif option_type == "main_plus_eps":
            # get main network
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")
            
            # get brain mask
            try:
                brain_mask = kwargs['brain_mask']
            except:
                raise ValueError("brain_mask not provided")
            
            # get brain mask
            try:
                eps = option['eps']
            except:
                raise ValueError("eps_confidence provided")
            
            try:
                main_confidence = option['main_confidence']
            except:
                main_confidence = 100

            
            confidence = assemble(
                interpolate( # inside, we trust the network plus a small value
                            conditional(brain_mask>1e-10, 1, 0)
                            * (eps + main_confidence  * conditional(main_network > 0, 1, 0) )
                            # outside, strong cce, where we set no network
                            + main_confidence * conditional(brain_mask<=1e-10, 1, 0), 
                            main_network.function_space()
                            )
                        )
            confidence.rename(f"confmain_{main_confidence:.2e}_eps{eps:.2e}")
            
            return confidence
        elif option_type == "main_tof_eps":
            # get main network
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")
            
            try:
                tof = kwargs['tof']
            except:
                raise ValueError("main_network not provided")
            
            # get brain mask
            try:
                brain_mask = kwargs['brain_mask']
            except:
                raise ValueError("brain_mask not provided")
            
            # get eps lift
            try:
                eps = option['eps']
            except:
                raise ValueError("eps not provided")
            

            # get eps lift
            try:
                scale_tof = option['scale_tof']
            except:
                scale_tof = 1.0
            

            try:
                main_confidence = option['main_confidence']
            except:
                main_confidence = 100

            
            confidence = assemble(
                interpolate( # inside, we trust the network plus a small value
                            conditional(brain_mask>1e-10, 1, 0)
                            * (
                                main_confidence  * conditional(main_network > 0, 1, 0) 
                                + scale_tof * tof 
                                + eps 
                                )
                                # outside, strong cce, where we set no network
                            + main_confidence * conditional(brain_mask<=1e-10, 1, 0), 
                            main_network.function_space()
                            )
                        )
            confidence.rename(f"CONFmain_{main_confidence:.2e}_tof_eps{eps:.2e}")
            
            return confidence



        else:
            raise ValueError(f"Unknown confidence option {option}")

    
    def set_kappa(**kwargs):
        """
        Set kappa function.
        In the region with high value of kappa the network passage is penalized.
        """
        try:
            option = kwargs["kappa"]
        except:
            raise ValueError("kappa not provided")
            
        if isinstance(option, str):
            option_type = option
        elif isinstance(option, dict):
            option_type = option["type"]
        else:
            raise ValueError(f"Unknown kappa type {option}")
        
        common_name = "kappa"

        if option_type == "one":
            try:
                mesh = kwargs['mesh']
            except:
                raise ValueError("mesh not provided")
            one = Function(FunctionSpace(mesh,"R",0), name=common_name+"one")
            one.assign(1.0)
            return one
        elif option_type == "t1":
            PETSc.Sys.Print(option)
            try:
                t1 = kwargs['t1']
            except:
                raise ValueError("t1 not provided")
            
            try:
                brain_mask = kwargs['brain_mask']
            except:
                raise ValueError("brain_mask not provided")
            
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")
            
            try:
                level1 = option["level1"]
            except:
                level1 = 200
            
            try: 
                level2 = option["level2"]
            except:
                level2 = 500


            try:
                kappa1 = option["kappa1"]
            except:
                kappa1 = 2
            
            try: 
                kappa2 = option["kappa2"]
            except:
                kappa2 = 4
            

            name = f"{common_name}t1_{level1:.0f}k{kappa1:.1e}_{level2:.0f}k{kappa2:.1e}"
            kappa = Function(t1.function_space(), name=name)
            kappa.interpolate(# base value is value (Euclidean distace)
                              1.0
                              # outsise the main network, we penalize the passage 
                              # but only in the region where t1 is high
                              # or outside the brain domain 
                              + conditional(main_network > 0, 0, 1) 
                              * (
                                  kappa1 * conditional(t1 > level1, 1, 0) * conditional(t1 < level2, 1, 0)
                                  + kappa2 * conditional(t1 > level2, 1, 0)
                                + 10*conditional(brain_mask < 1e-10, 1, 0 ) 
                                )
                              
                        )
            with kappa.dat.vec_ro as kappa_vec:
                PETSc.Sys.Print(utilities.msg_bounds(kappa_vec, "kappa function"))
            return kappa
               
        elif option_type == "sink_support":
            PETSc.Sys.Print(option)
            try:
                sink_support = kwargs['sink_support']
            except:
                raise ValueError("sink not provided")
            
            try:
                brain_mask = kwargs['brain_mask']
            except:
                raise ValueError("brain_mask not provided")
            
            try:
                corrupted_fun = kwargs['corrupted_fun']
            except:
                raise ValueError("corrupted")
            
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")
            
            try:
                kappa_support = option["kappa_support"]
            except:
                kappa_support = 1.5

            try:
                kappa_outside = option["kappa_outside"]
            except:
                kappa_outside = 5
            
            
            name = f"{common_name}sink_support"
            kappa = Function(corrupted_fun.function_space(), name=name)
            kappa.interpolate(# base value is value (Euclidean distace)
                              1.0
                              # outsise the main network and where there are data
                              # we penalize the passage
                              + conditional(main_network > 0, 0, 1)
                              * conditional(corrupted_fun > 1e-10, 0, 1) 
                              * sink_support * kappa_support
                                + conditional(main_network > 0, 0, 1)
                                * conditional(brain_mask < 1e-10, 1, 0 )
                                * kappa_outside
                              )
            
            with kappa.dat.vec_ro as kappa_vec:
                PETSc.Sys.Print(utilities.msg_bounds(kappa_vec, "kappa function"))
            return kappa
        elif option_type == "t1white":
            try:
                t1 = kwargs['t1']
            except:
                raise ValueError("t1 not provided")
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")

            kappa = Function(t1.function_space(), name=common_name+"t1white")
            kappa.interpolate(# base value is value (Euclidean distace)
                              1.0
                              # outsise the main network, we penalize the passage 
                              + conditional(main_network > 0, 0, 1) 
                              # but only in the region where t1 is high
                              * conditional(t1 > 500, 10, 0) )
            return kappa
        elif option_type == "white":
            try:
                aseg = kwargs['aseg']
            except:
                raise ValueError("aseg not provided")
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")

            kappa = Function(aseg.function_space(), name=common_name+"white")
            kappa.interpolate(# base value is value (Euclidean distace)
                              1.0
                              # outsise the main network, we penalize the passage 
                              + conditional(main_network > 0, 0, 1) 
                              # but only in the region where t1 is high
                              * ( 
                                  # left hemisphere white matter is labeled 2
                                  10 * conditional( aseg > 1.9, 1, 0) * conditional( aseg < 2.1, 1, 0)
                                  # right hemisphere white matter is labeled 41
                                + 10 * conditional( aseg > 40.9, 1, 0) * conditional( aseg < 41.1, 1, 0)
                                ) )
            return kappa
        else:
            raise ValueError(f"Unknown kappa option {option}")
        
    def set_initial_guess(option, **kwargs):
        common_name = "ini"

        if isinstance(option, str):
            option_type = option
        elif isinstance(option, dict):
            option_type = option["type"]
        else:
            raise ValueError(f"Unknown initial {option}")

        
        if option_type == "one":
            try:
                mesh = kwargs['mesh']
            except:
                raise ValueError("mesh not provided")
            one = Function(FunctionSpace(mesh,"R",0), name=common_name+"one")
            one.assign(1.0)
            return one

        if option_type == "low":
            try:
                corrupted = kwargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            heat = HeatMap(corrupted.function_space(), scaling=1.0, sigma=1e1)
            low = Function(corrupted.function_space(), name=common_name+"low_heat", label=f"use heat map with sigma=1e1 and lift by 1e-4")
            low.assign(heat(corrupted+1e-4) + 1e-4)
            return low
        
        if option_type == "medium":
            try:
                corrupted = kwargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            low = set_initial_guess("low", corrupted)
            medium = Function(corrupted.function_space(),name=common_name+"medium_heat")
            medium.assign(heat(low+1e-4) + 1e-4)
            return medium
        
        if option_type == "high":
            try:
                corrupted = kwargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            medium = set_initial_guess("medium", corrupted)
            high = Function(corrupted.function_space(),name=common_name+"low_heat")
            high.assign(heat(medium+1e-4) + 1e-4)
            return high

        if option_type == "low_gaussian":
            try:
                corrupted = kwargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            try: 
                map = kwargs['map']
                scaling = map['scaling']
            except:
                raise ValueError("map not provided")
            
            blurer = kwargs["blurer"]
            low = blurer(corrupted, sigma=4)/scaling
            low += 1e-2
            return low
        
        if option_type == "medium_gaussian":
            try:
                corrupted = kwargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            low = set_initial_guess("low_gaussian", **kwargs)
            blurer = kwargs["blurer"]
            medium = blurer(low, sigma=4)
            medium += 1e-4
            return medium
         
        if option_type == "high_gaussian":
            try:
                corrupted = kwargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            medium = set_initial_guess("medium_gaussian", corrupted=corrupted)
            blurer = kwargs["blurer"]
            high = blurer(medium, sigma=4)
            high += 1e-4
            return high
        
        if option_type == "tof_on_main_network":
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")
            
            try: 
                scaling_tof = options['scaling_tof']
            except:
                scaling_tof = 1.0
            
            try:
                tof = kwargs['tof']
            except:
                raise ValueError("tof not provided")
            
            try:
                brain_mask = kwargs['brain_mask']
            except:
                raise ValueError("brain mask not provided")
            
            try:
                lift_brain = option["lift_brain"]
            except:
                lift_brain = 1e-4
            
            try:
                lift = option["lift"]
            except:
                lift = 1e-4
            


            initial = Function(main_network.function_space(), name=common_name+"tof_on_main_network")
            initial.interpolate(tof * scaling_tof * conditional(main_network > 0, 1, 0)
                                + lift_brain * brain_mask
                                + lift
                                )
            return initial
        
        if option_type == "load":
            try:
                path = option['path']
            except:
                raise ValueError("path initial tdens to provided")
            
            try:
                mesh = kwargs["mesh"]
            except:
                raise ValueError("mesh not provided")
            initial = nii2firedrake(path, mesh, name=common_name+"load",comm=mesh.comm)

            try:
                scaling = option['scaling']
            except:
                scaling = 1.0
            initial *= scaling

            try:
                lift = option['lift']
            except:
                scaling = 0.0
            initial += lift
            


            return initial
        
        if option_type == "corrupted":
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")
            
            try:
                brain_mask = kwargs['brain_mask']
            except:
                raise ValueError("brain mask not provided")
            
            try:
                corrupted = kwargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            try:
                lift = option["lift"]
            except:
                lift = 1e-4

            name = common_name+f"corrupted_lift{lift:.2e}"
            support = conditional(brain_mask > 1e-10, 1, 0) + conditional(brain_mask > 1e-10, 0, 1) * conditional(main_network > 0, 1, 0)
            initial = Function(main_network.function_space(), name=name)
            initial.interpolate(corrupted*support + lift)

            try:
                sigma_heat = option["sigma_heat"]
                heat = HeatMap(initial.function_space(), scaling=1.0, sigma=sigma_heat)
                initial.assign(heat(initial))
                name += f"_heat{sigma_heat:.2e}"
                initial.rename(name)
            except:
                pass


            return initial
        
        if option_type == "support":
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")
            
            try:
                brain_mask = kwargs['brain_mask']
            except:
                raise ValueError("brain not provided")
            
            try:
                lift = option["lift"]
            except:
                lift = 1e-4

            name = common_name+f"support_lift{lift:.2e}"
            support = conditional(brain_mask > 1e-10, 1, 0) + conditional(brain_mask > 1e-10, 0, 1) * conditional(main_network > 0, 1, 0)
            initial = Function(main_network.function_space(), name=name)
            initial.interpolate(support + lift)

            try:
                sigma_heat = option["sigma_heat"]
                heat = HeatMap(initial.function_space(), scaling=1.0, sigma=sigma_heat)
                initial.assign(heat(initial))
                name += f"_heat{sigma_heat:.2e}"
                initial.rename(name)
            except:
                pass


            return initial
        
        if option_type == "skeleton_thickness":
            try:
                skeleton = kwargs["skeleton"]
            except:
                raise ValueError("skeleton not provided")
            
            try:
                thickness = kwargs["thickness"]
            except:
                raise ValueError("thickness not provided")
            
            try:
                mu0 = option["mu0"]
            except:
                raise ValueError("mu0 not provided")
            
            try:
                mesh = kwargs['mesh']
            except:
                raise ValueError("mesh not provided")
            
            try:
                voxel_size = kwargs['voxel_size']
                h = min(voxel_size)
            except:
                raise ValueError("voxel_size not provided")
            
            try:
                lift = option["lift"]
            except:
                lift = 1e-15


            # get brain mask
            try:
                brain_mask = kwargs['brain_mask']
            except:
                raise ValueError("brain_mask not provided")

            
           
            dim = mesh.geometric_dimension()
            exponent_p = 4.0 if dim == 3 else 3.0
            name = common_name+f"pou_{mu0:.2e}_lift{lift:.2e}"
            initial = Function(skeleton.function_space(), name=name)
            initial.interpolate(mu0*skeleton * (thickness/2) ** exponent_p / h ** (dim-1) + lift * conditional(brain_mask > 1e-10, 1, 0)+ 1e-8)
            try:
                sigma_heat = option["sigma_heat"]
                heat = HeatMap(initial.function_space(), scaling=1.0, sigma=sigma_heat)
                initial.assign(heat(initial))
                name += f"_heat{sigma_heat:.2e}"
                initial.rename(name)
            except:
                pass

            return initial



        else:
            raise ValueError(f"Unknown initial guess option {option}")
        



    combinations = list(product_dict(**options))
    PETSc.Sys.Print(combinations)

    
    
    # split the combinations in the ensemble
    def lol(a, n):
        k, m = divmod(len(a), n)
        return (a[i*k+min(i, m):(i+1)*k+min(i+1, m)] for i in range(n))
    
    PETSc.Sys.Print(f"**** RUNNING ****** ")
    if use_ensemble:
        sub_combinations = list(lol(combinations, my_ensemble.ensemble_comm.size))
        todo = sub_combinations[my_ensemble.ensemble_comm.rank]
        my_ensemble.ensemble_comm.barrier()
        PETSc.Sys.Print(f"color {color_rank} TODO :{len(todo)=}",comm=comm)
    else:
        todo = combinations
        PETSc.Sys.Print(f"color {color_rank} TODO {len(todo)=}",comm=comm)
    
    DG0_cartesian = FunctionSpace(cartesian_mesh, "DG", 0)
    DG0 = FunctionSpace(mesh, "DG", 0)
    interpolate_fun = Function(DG0, name="interpolatator_fun")
    interpolator = interpolate(interpolate_fun, DG0_cartesian, 
                                 allow_missing_dofs=True,  
                                 default_missing_val=-1e30)
    
    
    
    for i, combination in enumerate(todo):
        labels = []
        try:
            spaces = combination["spaces"]
            if spaces != "DG0DG0":
                labels.append(f"spaces{spaces}")
        except:
            spaces = "DG0DG0"
        PETSc.Sys.Print(f"{spaces}= {spaces}")
        
        #
        # set corrupted network
        #
        corrupted = set_corrupted_network(**combination, **input_data)
        
        #
        # btp inputs
        #
        sink = set_sink(option_type="sink_support",**combination, **input_data)

        R = FunctionSpace(mesh,"R",0)
        source = Function(R, name="source")
        source.assign(0.0)

        kappa = set_kappa(**combination, corrupted_fun=corrupted, **input_data)
                
        
        if spaces == "DG0DG0":
            inlet_pressure = Function(inlets.function_space())
            inlet_pressure.assign(0.0)
            if mesh.extruded:
                PETSc.Sys.Print("Using bottom boundary for inlet pressure")
                # we need to use the bottom boundary only
                weak_Dirichlet = [(inlet_pressure, ds_b, inlets)]
            else:
                PETSc.Sys.Print("Using all boundary for inlet pressure")
                # we use all the boundary
                weak_Dirichlet = [(inlet_pressure, ds, inlets)]

            strong_Dirichlet = None
        elif spaces == "CG1DG0" or spaces == "CR1DG0":
            weak_Dirichlet = None
            strong_Dirichlet = [(99, 0.0)]
        
        
        btp = ot.BranchedTransportProblem(source, sink, 
                                      gamma=0.5, 
                                      Dirichlet = strong_Dirichlet,
                                      weak_Dirichlet = weak_Dirichlet,
                                      kappa=kappa)

        #
        # set confidence
        #
        confidence = set_confidence(**combination, **input_data)
        

        #
        # set initial guess
        # 
        initial = set_initial_guess(combination["initial"], map=combination["map"], corrupted=corrupted, **input_data)
        
        # 
        # set labels defining the experiment
        # 
        labels += [
            f"wd{combination['wd']:.1e}",
            initial.name(),
            confidence.name(),
            ]

        tdens2image = combination["map"]
        if tdens2image['type'] == 'identity':
            labels.append(f'mapidentity')
        elif tdens2image['type'] == 'heat':
            labels.append(f"mapheat{tdens2image['sigma']:.1e}")
        elif tdens2image['type'] == 'pm':
            try:
                labels.append(f"mapipm"
                              +f"_cond{tdens2image['cond_zero']:.2e}"
                              +f"_scale{tdens2image['scaling']:.2e}")
            except:
                labels.append(f"mappm")
        else:
            raise ValueError(f'Unknown tdens2image {tdens2image}')
        
        labels.append(corrupted.name())
        labels.append(sink.name())
        labels.append(kappa.name())

        try:
            discrepancy_norm = combination["discrepancy_norm"]
        except: 
            discrepancy_norm = "l2"

        if discrepancy_norm == "dual_h1":
            try: 
                discrepancy_dual_h1_sigma = combination["discrepancy_dual_h1_sigma"]
            except:
                discrepancy_dual_h1_sigma = 1.0

            


        labels.append(f"DISC{discrepancy_norm}")
        if discrepancy_norm == "dual_h1":
            labels.append(f"sigma{discrepancy_dual_h1_sigma:.1e}")

        
        label = "_".join(labels)
        label_dir = os.path.join(out_directory,label)
        mpi_mkdir(label_dir, comm)


        # get git version used 
        git_hash = subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()
        combination4save = cp(combination)
        combination4save["git_hash"] = git_hash
        # save a copy of current combination as json
        with open(f"{label_dir}/option.json", 'w') as f:
            json.dump(combination4save, f, indent=4)
        
        
        

        # setup solver
        mesh.comm.Barrier()
        niot_solver = NiotSolver(btp, 
                             corrupted,  
                             confidence=confidence, 
                             spaces = spaces,
                             cell2face = 'harmonic_mean',
                             setup = False,
                             ensemble_comm=None
                             )
        # inpainting
        wd = combination["wd"]
        niot_solver.ctrl_set('discrepancy_weight', wd)
        niot_solver.ctrl_set('regularization_weight', 0.0)
        
        niot_solver.ctrl_set("min_tdens", 1e-7)


        tdens2image = combination["map"]
        niot_solver.ctrl_set(['tdens2image'], tdens2image)

        use_adjoint = combination.get("use_adjoint", 0) == 1
        niot_solver.ctrl_set(['use_adjoint'], use_adjoint)
        if tdens2image['type'] == 'pm':
            # we need to use the adjoint to compute the gradient of the discrepancy
            niot_solver.ctrl_set(['use_adjoint'], True)
        
        niot_solver.ctrl_set("discrepancy_norm", discrepancy_norm)

        if discrepancy_norm == "dual_h1":
            niot_solver.ctrl_set("discrepancy_dual_h1_sigma", discrepancy_dual_h1_sigma)


        # optimization
        niot_solver.ctrl_set('optimization_tol', 1e-5)
        niot_solver.ctrl_set('constraint_tol', 1e-6)
        try: 
            max_iter = combination["max_iter"]
        except:
            max_iter = 5000
            
        niot_solver.ctrl_set('max_iter', max_iter)
        
        niot_solver.ctrl_set('max_restart', 4)
        niot_solver.ctrl_set('verbose', 3)

        




        # time discretization
        method = "tdens_mirror_descent_explicit"
        niot_solver.ctrl_set(['dmk','type'], method)

        # time step
        default_deltat_control = {
        'type': 'adaptive2',
        'lower_bound': 1e-13,
        'upper_bound': 5e-2,
        'expansion': 1.1,
        'contraction': 0.5,
        'order_down': -0.6,
        'order_up': 0.6,
        }
        deltat_control = combination.get("deltat_control", default_deltat_control) 
        niot_solver.ctrl_set(['dmk',method,'deltat'], deltat_control)
        
        # setup log file
        log_filename = os.path.join(label_dir,f"niot.log")
        niot_solver.ctrl_set("log_file",log_filename)
        
        # set solvers according to controls
        niot_solver.setup()
        
        # set intial guess
        niot_solver.set_solution(tdens=initial)

        try:
            file_nii_pot = combination["initial_pot"]
            pot = nii2firedrake(file_nii_pot, mesh, name="initial_pot",comm=mesh.comm)
            niot_solver.set_solution(pot=pot)
        except:
            pass
        

        save_inputs = combination.get("save_inputs", 0) == 1

        save_h5 = combination.get("save_h5", 0) == 1
        save_nifti = combination.get("save_nifti", 0) == 1
        save_pvd = combination.get("save_pvd", 0) == 1
        if save_inputs:
            if spaces == "DG0DG0":
                filename = f"{label_dir}/corrupted.nii.gz"
                save_as_nifti(corrupted, filename,  affine, dimensions, lengths, offset)

                filename = f"{label_dir}/sink.nii.gz"
                save_as_nifti(sink,  filename,  affine, dimensions, lengths, offset)

                if combination["initial"] != "one":
                    filename = f"{label_dir}/initial.nii.gz"
                    save_as_nifti(initial, filename,  affine, dimensions, lengths, offset)


                if combination["confidence"] != "one":
                    filename = f"{label_dir}/confidence.nii.gz"
                    save_as_nifti(confidence, filename,  affine, dimensions, lengths, offset)

                if combination["kappa"] != "one":
                    filename = f"{label_dir}/kappa.nii.gz"
                    save_as_nifti(btp.kappa, filename,  affine, dimensions, lengths, offset)
                    with kappa.dat.vec_ro as kappa_vec:
                        PETSc.Sys.Print(utilities.msg_bounds(kappa_vec, "kappa function"))
                
                filename = f"{label_dir}/main_network.nii.gz"
                save_as_nifti(main_network, filename,  affine, dimensions, lengths, offset)

                filename = f"{label_dir}/external_network.nii.gz"
                save_as_nifti(external_network, filename,  affine, dimensions, lengths, offset)
            else:
                if save_h5:
                    h5_file_inputs = os.path.join(label_dir, "inputs.h5")
                    with CheckpointFile(h5_file_inputs, 'w',comm=comm) as afile:
                        afile.save_function(corrupted, "corrupted")
                        afile.save_function(sink, "sink")
                        if combination["initial"] != "one":
                            afile.save_function(initial, "initial")
                        if combination["confidence"] != "one":
                            afile.save_function(confidence, "confidence")
                        if combination["kappa"] != "one":
                            afile.save_function(kappa, "kappa")
                        afile.save_function(main_network, "main_network")
                if save_pvd:
                    filename = f"{label_dir}/inputs.pvd"
                    data = [corrupted, sink]
                    if combination["initial"] != "one":
                        data.append(initial)
                    if combination["confidence"] != "one":
                        data.append(confidence)
                    if combination["kappa"] != "one":
                        data.append(kappa)
                    data.append(main_network)
                    VTKFile(filename).write(*data)

        
        #
        # run solver, buffering the saving of the solution
        #
        total_iterations = niot_solver.ctrl_get('max_iter')
        try: 
            buffer = combination["buffer"]
        except:
            buffer = 500
        
        buffer_saving = max(1,min(buffer,total_iterations))


        def solve_and_save(niot_solver, label_dir, n_buffer):
            gc.collect()

            # select if we just want the latest solution 
            update_solution = True
            if update_solution:
                file_label = "final"
            else:
                file_label = f"{n_buffer:02}"

                
            
            n_iter = niot_solver.ctrl_get('max_iter')
            # solve
            ierr = niot_solver.solve()
            PETSc.Sys.Print(f"Solved done {ierr=}",comm=comm)
            if ierr < 0:
                return ierr

            # save solution
            pot, tdens, vel = niot_solver.get_otp_solution(niot_solver.sol)
            PETSc.Sys.Print(f"extract otp solution")

            save_h5 = combination.get("save_h5", 0) == 1
            save_nifti = combination.get("save_nifti", 0) == 1
            save_pvd = combination.get("save_pvd", 0) == 1
            if save_nifti:
                filename=f"{label_dir}/tdens_{file_label}.nii.gz"
                save_as_nifti(tdens, filename,  affine, dimensions, lengths, offset)
            
                filename=f"{label_dir}/pot_{file_label}.nii.gz"
                save_as_nifti(pot, filename,  affine, dimensions, lengths, offset)

                flux_component = Function(niot_solver.mesh,"DG",0)
                for i in range(niot_solver.mesh.geometric_dimension() ):
                    flux_component.interpolate(vel[i])
                    filename=f"{label_dir}/flux{i}_{file_label}.nii.gz"
                    save_as_nifti(flux_component, filename,  affine, dimensions, lengths, offset)

                if combination["map"]['type'] == 'pm':
                    filename = f"{label_dir}/image_reconstruction_{file_label}.nii.gz"
                    save_as_nifti(niot_solver.image_h, 
                                  filename,  affine, dimensions, lengths, offset)

                    actual_rec = niot_solver.tdens2image_map(tdens)
                    filename = f"{label_dir}/real_image_{file_label}.nii.gz"
                    save_as_nifti(actual_rec, 
                                  filename,  affine, dimensions, lengths, offset)

            if save_h5:
                h5_file = os.path.join(label_dir, f"solution_{file_label}.h5")
                # if file exists, remove it
                PETSc.Sys.Print(f"Start h5 - {label}",comm=comm)
                with CheckpointFile(h5_file, 'w',comm=comm) as afile:
                    tic = time.time()
                    PETSc.Sys.Print(f"Start tdens - {label}",comm=comm)
                    afile.save_function(tdens)
                    cpu = time.time() - tic
                    PETSc.Sys.Print(f"color {color_rank} - Saved tdens cpu {cpu:.1f} s - {label}",comm=comm)
                    PETSc.Sys.Print(f"Start pot - {label}",comm=comm)
                    afile.save_function(pot)
                    if combination["map"]['type'] == 'pm':
                        afile.save_function(niot_solver.image_h)
                    PETSc.Sys.Print(f" Include info ")
                    afile.require_group("/info/")
                    afile.set_attr("/info/", "affine", affine)
                    PETSc.Sys.Print(f" affine ")
                    afile.set_attr("/info/", "offset", offset)
                    PETSc.Sys.Print(f" offeset")
                    afile.set_attr("/info/", "voxel_size", voxel_size)
                    PETSc.Sys.Print(f" voxel_size ")
                    afile.set_attr("/info/", "dimensions", dimensions)
                    PETSc.Sys.Print(f" dimensions ")
                    PETSc.Sys.Print(f" - done")
            if save_pvd:
                filename=f"{label_dir}/solution_{file_label}.pvd"
                data = [tdens, pot]
                start = time.time() 
                    
                if combination["map"]['type'] == 'pm':
                    data.append(niot_solver.image_h)
                VTKFile(filename).write(*data)
                PETSc.Sys.Print(f"color {color_rank} - Saved as pvd {time.time()-start:.1f} s - {label}",comm=comm)

            save_intermediate = False
            if save_intermediate and combination["map"]["type"] != "identity":
                for i, img in enumerate(niot_solver.tdens2image_map.intermediate_images):
                    filename = f"{label_dir}/image_intermediate_{file_label}_{i}.nii.gz"
                    save_as_nifti(img, affine, filename, interpolator, interpolate_fun)

            return ierr
                
                
            

        # run and save
        niot_solver.ctrl_set('max_iter', total_iterations%buffer_saving)
        solve_and_save(niot_solver, label_dir, 0)
        PETSc.Sys.Print(f"color {color_rank} - First {total_iterations%buffer_saving} iterations done - {label}",comm=comm)
        

        # run and save, skipping initial steps
        niot_solver.ctrl_set('max_iter',buffer_saving)
        niot_solver.ctrl_set('restart',True)
        for i in range(total_iterations//buffer_saving):
            interval = [i*buffer_saving,(i+1)*buffer_saving]
            tic = time.time()
            ierr = solve_and_save(niot_solver, label_dir, i+1)
            cpu = time.time() - tic
            if ierr < 0:
                PETSc.Sys.Print(f"color {color_rank} - Stopping at {interval[0]} iterations - {label}",comm=comm)
                break
            PETSc.Sys.Print(f"color {color_rank} - Done {interval[1]/total_iterations*100:.1f}% of {total_iterations}- avg cpu {cpu/buffer_saving:.1f} s - {label}",comm=comm)


        niot_solver = None

        gc.collect()
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(exit_on_error=True, description='Reconstruct network')
    #parser.add_argument("--field", type=str, default='TOF', help="TOF")
    parser.add_argument("--n_ensemble", type=int, default=1, help="Number of processor per simulation")
    parser.add_argument("--mri", type=str, default="./mri/", help="directory with mri data")
    parser.add_argument("--out", type=str, default="./runs/", help="output directory")
    parser.add_argument("--options", type=str, default="options.json", help="Json file with controls")
    parser.add_argument("--h5", type=str, default="", help="h5 file with inputs")

    args, unknown = parser.parse_known_args()

    

    experiment(args)
