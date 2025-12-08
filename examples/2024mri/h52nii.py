from firedrake import *
from firedrake.petsc import PETSc
from time import time
import os
import nibabel
import numpy as np
from niot import image2dat as i2d
import gc
import argparse


comm = COMM_WORLD
funs = []

def h52nii(h5_file, di_name="./"):
    # Load the mesh from the HDF5 file
    with CheckpointFile(h5_file, 'r',comm=comm) as afile:
        # get mesh 
        start = time()
        PETSc.Sys.Print(f"Start mesh ")
        mesh = afile.load_mesh("relabeled_mesh")

        # get info of original image
        affine = afile.get_attr("/info/", "affine")
        PETSc.Sys.Print(f" affine ")
        offset = afile.get_attr("/info/", "offset")
        PETSc.Sys.Print(f" offeet")
        voxel_size = afile.get_attr("/info/", "voxel_size")
        PETSc.Sys.Print(f" voxel_size ")
        dimensions = afile.get_attr("/info/", "dimensions")
        PETSc.Sys.Print(f" dimensions ")
        lenghts = voxel_size*dimensions


    # create cartesian mesh for interpolation
    cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,
                                            lengths=lenghts,
                                            offset=offset)

    DG0_cartesian = FunctionSpace(cartesian_mesh, "DG", 0)
    DG0 = FunctionSpace(mesh, "DG", 0)
    interpolate_fun = Function(DG0, name="interpolatator_fun")
    interpolator = interpolate(interpolate_fun, DG0_cartesian, 
                                allow_missing_dofs=True,  
                                default_missing_val=1e30)


    with CheckpointFile(h5_file, 'r',comm=comm) as afile:
        # get all functions in the file
        scalar_functions_name = afile._get_function_name_function_space_name_map(afile._get_mesh_name_topology_name_map()[mesh.name], mesh.name)

        # read all functions
        for fname in scalar_functions_name.keys():
            start = time()  
            fun = afile.load_function(mesh, fname)
            PETSc.Sys.Print(f" Loaded {fname} in {time()-start:.2f} seconds")

            # interpolate to cartesian grid
            PETSc.Sys.Print(f" Interpolating {fun.name()} to cartesian grid")
            start = time()
            fun_cartesian = Function(DG0_cartesian, name=fun.name())
            fun_cartesian.interpolate(fun, 
                                    allow_missing_dofs=True,  
                                    default_missing_val=+1e30)
            PETSc.Sys.Print(f" Interpolated {fun.name()} in {time()-start:.2f} seconds")

            # convert in numpy array
            function_np = i2d.firedrake2numpy(fun_cartesian)

            # save as nii
            filename = os.path.join(di_name, f"{fun.name()}.nii.gz")
            nibabel.save(nibabel.Nifti1Image(function_np, affine), filename)
            function_np = None
            gc.collect()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 mesh and functions to NIfTI format.")
    parser.add_argument("--h5", type=str, help="Path to the input HDF5 file.")
    parser.add_argument("--out", type=str, default="./", help="Directory to save the output NIfTI files.")
    args = parser.parse_args()

    h52nii(args.h5, args.out)