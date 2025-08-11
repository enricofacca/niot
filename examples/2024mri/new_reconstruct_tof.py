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



    tof = kwargs['tof']
    main_network = kwargs['main_network']
    external_network = kwargs['external_network']
    brain_mask = kwargs['brain_mask']
    
    if option_type == "tof":
        try:
            threshold_tof = option['threshold_tof']
        except:
            threshold_tof = 200

        name = f"OBSsupport_t{threshold_tof:.2e}"

        try: 
            blur = option['blur']
        except:
            blur = 0.0
        
        if blur > 0:
            tof_np = i2d.firedrake2numpy(tof)
            mesh = kwargs["cartesian_mesh"]
            hx = mesh.hx
            tof_np = gaussian_filter(tof_np, sigma=blur*hx)
            name += f"_blur{blur:.2e}"
            tof4corrupted = i2d.numpy2firedrake(mesh, tof_np, name="tof_blurred")
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
    
    elif option_type == "support":
        try:
            threshold_tof = option['threshold_tof']
        except:
            threshold_tof = 200

        name = f"OBSsupport_t{threshold_tof:.2e}"

        try: 
            blur = option['blur']
        except:
            blur = 0.0
        
        if blur > 0:
            tof_np = i2d.firedrake2numpy(tof)
            mesh = kwargs["cartesian_mesh"]
            hx = mesh.hx
            tof_np = gaussian_filter(tof_np, sigma=blur*hx)
            name += f"_blur{blur:.2e}"
            tof4corrupted = i2d.numpy2firedrake(mesh, tof_np, name="tof_blurred")
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
        color_rank = 0
    else:
        my_ensemble = Ensemble(COMM_WORLD, args.n_ensemble)
        comm = my_ensemble.comm
        use_ensemble = True
        color_rank = my_ensemble.ensemble_comm.rank
    
    #
    # check if h5 already exists or build it, but it may run out of memory
    #
    PETSc.Sys.Print(f"**** Inputs loading ****")
    if blur > 0:
        label = f"t{threshold:.2e}_blur{blur:.2e}"  
    else: 
        label = f"t{threshold:.2e}"

    if args.reuse_h5:
        h5_file = f"{args.mri}/inputs_{label}_nproc{args.n_ensemble}.h5"
        if os.path.exists(h5_file):
            PETSc.Sys.Print(f"Found checkpoint file {h5_file}")
        else:   
            PETSc.Sys.Print(f"Checkpoint not found. Creating it but we may run out of memory.\n"
                            f"Consider running mpiexec -n {args.n_ensemble} python build_checkpointfile.py "
                            )
            if use_ensemble:
                if my_ensemble.ensemble_comm.rank == 0:
                    data = setup_h5(args.mri, threshold, blur, comm=comm)
                    write_h5(args.mri, threshold, blur, comm, args.n_ensemble, data=data)
                my_ensemble.ensemble_comm.barrier()
            else:
                data = setup_h5(args.mri, threshold, blur, comm=comm)
                write_h5(args.mri, threshold, blur, comm, args.n_ensemble, data=data)
            PETSc.Sys.Print(f"Checkpoint created h5_file={h5_file}")
        


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
            skeleton = afile.load_function(mesh, "skeleton")
            PETSc.Sys.Print(f"skeleton",end="")
            thickness = afile.load_function(mesh, "thickness")
            PETSc.Sys.Print(f"thickness",end=" ")
        
        PETSc.Sys.Print(f"Checkpoint loaded")
        PETSc.Sys.Print(f"**** Inputs loaded ****")
        PETSc.Sys.Print(f"")
        if use_ensemble:
            my_ensemble.ensemble_comm.barrier()

    else:
        data = setup_h5(args.mri, threshold, blur, comm=comm)
        cartesian_mesh, tof, aseg, t1, brain_mask, main_network, external_network, inlets, skeleton, thickness = data
        
        mesh = cartesian_mesh

    mesh.nx = original_dimensions[0]
    mesh.ny = original_dimensions[1]
    mesh.nz = original_dimensions[2]
    mesh.xmin = 0.0
    mesh.xmax = lengths[0]
    mesh.ymin = 0.0
    mesh.ymax = lengths[1]
    mesh.zmin = 0.0
    mesh.zmax = lengths[2]
    mesh.hx = hx
    mesh.hy = hy
    mesh.hz = hz


    
    input_data = { 
        "tof": tof, 
        "aseg": aseg, 
        "brain_mask": brain_mask, 
        "t1": t1, 
        "inlets": inlets, 
        "main_network": main_network, 
        "external_network": external_network,
        "cartesian_mesh": cartesian_mesh,
        "skeleton" : skeleton,
        "thickness": thickness,
    }


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
                mesh = kwargs['cartesian_mesh']
            except:
                raise ValueError("cartesian_mesh not provided")
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
                main_confidence = kwargs['main_confidence']
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
                mesh = kwargs['cartesian_mesh']
            except:
                raise ValueError("cartesian_mesh not provided")
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
                              + conditional(main_network > 0, 0, 1) 
                              # but only in the region where t1 is high
                              * (
                                  kappa1 * conditional(t1 > level1, 1, 0) * conditional(t1 < level2, 1, 0)
                                  + kappa2 * conditional(t1 > level2, 1, 0)
                                ) 
                        )
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
                mesh = kwargs['cartesian_mesh']
            except:
                raise ValueError("cartesian_mesh not provided")
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
            
            corrupted_np = i2d.firedrake2numpy(corrupted)/scaling
            low_np = gaussian_filter(corrupted_np, sigma=4, truncate=1e0)
            low = i2d.numpy2firedrake(cartesian_mesh, low_np, name=common_name+"low_gaussian")
            low += 1e-2
            return low
        
        if option_type == "medium_gaussian":
            try:
                corrupted = kwargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            low = set_initial_guess("low_gaussian", **kwargs)
            low_np = i2d.firedrake2numpy(low)
            medium_np = gaussian_filter(low_np, sigma=4, truncate=1e0)
            medium = i2d.numpy2firedrake(cartesian_mesh, medium_np, name=common_name+"medium_gaussian")
            medium += 1e-4
            return medium
        
        if option_type == "high_gaussian":
            try:
                corrupted = kwargs['corrupted']
            except:
                raise ValueError("corrupted not provided")
            
            medium = set_initial_guess("medium_gaussian", corrupted=corrupted)
            medium_np = i2d.firedrake2numpy(medium)
            high_np = gaussian_filter(medium_np, sigma=4, truncate=1e0)
            high = i2d.numpy2firedrake(cartesian_mesh, high_np, name=common_name+"high_gaussian")
            high += 1e-4
            return high
        
        if option_type == "main_network":
            try:
                main_network = kwargs['main_network']
            except:
                raise ValueError("main_network not provided")
            
            try: 
                map = kwargs['map']
                scaling = map['scaling']
            except:
                raise ValueError("map not provided")
            
            try:
                tof = kwargs['tof']
            except:
                raise ValueError("tof not provided")
            
            initial = Function(main_network.function_space(), name=common_name+"main_network")
            initial.interpolate(tof/scaling*conditional(main_network > 0, 1, 0))
            return initial
        
        if option_type == "load":
            try:
                path = option['path']
            except:
                raise ValueError("path initial tdens to provided")
            
            try:
                cartesian_mesh = kwargs["cartesian_mesh"]
            except:
                raise ValueError("cartesian_mesh not provided")
            initial = nii2firedrake(path,cartesian_mesh,name=common_name+"load",comm=cartesian_mesh.comm)
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
                mesh = kwargs['cartesian_mesh']
            except:
                raise ValueError("cartesian_mesh not provided")
            
            h = ( mesh.zmax - mesh.zmin ) / mesh.nz
            
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
    
    


    
    for i, combination in enumerate(todo):
        #
        # set corrupted network
        #
        corrupted = set_corrupted_network(**combination, **input_data)
        
        #
        # btp inputs
        #
        sink = set_sink(option_type="segmented",**combination, **input_data)

        R = FunctionSpace(cartesian_mesh,"R",0)
        source = Function(R, name="source")
        source.assign(0.0)

        kappa = set_kappa(**combination, **input_data)


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
        confidence = set_confidence(**combination, **input_data)
        
        #
        # set initial guess
        # 
        initial = set_initial_guess(combination["initial"], map=combination["map"], corrupted=corrupted, **input_data)

        
        # 
        # set labels defining the experiment
        # 
        labels = [f"wd{combination['wd']:.1e}",
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
        labels.append(corrupted.name())

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
        if tdens2image['type'] == 'pm':
            # we need to use the adjoint to compute the gradient of the discrepancy
            niot_solver.ctrl_set(['use_adjoint'], True)


        # optimization
        niot_solver.ctrl_set('optimization_tol', 1e-5)
        niot_solver.ctrl_set('constraint_tol', 1e-6)
        try: 
            max_iter = combination["max_iter"]
        except:
            max_iter = 5000
            
        niot_solver.ctrl_set('max_iter', max_iter)
        
        niot_solver.ctrl_set('max_restart', 4)
        niot_solver.ctrl_set('verbose', 0)

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
        niot_solver.set_solution(tdens=initial)

        try:
            file_nii_pot = combination["initial_pot"]
            pot = nii2firedrake(file_nii_pot, cartesian_mesh, name="initial_pot",comm=cartesian_mesh.comm)
            niot_solver.set_solution(pot=pot)
        except:
            pass
        
        save_inputs = True
        if save_inputs:
            filename = f"{label_dir}/corrupted.nii.gz"
            save_as_nifti(corrupted, affine, filename)

            filename = f"{label_dir}/sink.nii.gz"
            save_as_nifti(sink, affine, filename)

            if combination["initial"] != "one":
                filename = f"{label_dir}/initial.nii.gz"
                save_as_nifti(initial, affine, filename)


            if combination["confidence"] != "one":
                filename = f"{label_dir}/confidence.nii.gz"
                save_as_nifti(confidence, affine, filename)

            if combination["kappa"] != "one":
                filename = f"{label_dir}/kappa.nii.gz"
                save_as_nifti(kappa, affine, filename)
        
        
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
            
            # select if we just want the latest solution 
            update_solution = False
            if update_solution:
                file_label = "final"
            else:
                file_label = f"{n_buffer:02}"
                
            
            n_iter = niot_solver.ctrl_get('max_iter')
            # solve
            ierr = niot_solver.solve()

            # save solution
            pot, tdens, vel = niot_solver.get_otp_solution(niot_solver.sol)
            
            
            filename=f"{label_dir}/tdens_{file_label}.nii.gz"
            save_as_nifti(tdens, affine, filename)
            
            filename=f"{label_dir}/pot_{file_label}.nii.gz"
            save_as_nifti(pot, affine, filename)
        
            filename = f"{label_dir}/image_reconstruction_{file_label}.nii.gz"
            save_as_nifti(niot_solver.reconstruction, affine, filename)
                

            save_intermediate = False
            if save_intermediate and combination["map"]["type"] != "identity":
                for i, img in enumerate(niot_solver.tdens2image_map.intermediate_images):
                    filename = f"{label_dir}/image_intermediate_{file_label}_{i}.nii.gz"
                    save_as_nifti(img, affine, filename)
                
                
            

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
            solve_and_save(niot_solver, label_dir, i+1)
            cpu = time.time() - tic
            PETSc.Sys.Print(f"color {color_rank} - Done {interval[1]/total_iterations*100:.1f}% of {total_iterations}- avg cpu {cpu/buffer_saving:.1f} s - {label}",comm=comm)

        gc.collect()
        
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(exit_on_error=True, description='Reconstruct network')
    #parser.add_argument("--field", type=str, default='TOF', help="TOF")
    parser.add_argument("--n_ensemble", type=int, default=1, help="Number of processor per simulation")
    parser.add_argument("--mri", type=str, default="./mri/", help="directory with mri data")
    parser.add_argument("--out", type=str, default="./runs/", help="output directory")
    parser.add_argument("--options", type=str, default="options.json", help="Json file with controls")
    parser.add_argument("--reuse_h5", type=bool, default=False, help="Reuse h5 file with inputs")

    args, unknown = parser.parse_known_args()

    

    experiment(args)
