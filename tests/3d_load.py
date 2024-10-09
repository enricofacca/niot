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
from firedrake import File

from firedrake.petsc import PETSc



field = 'QSM'

coarseness = 3

load_and_convert = True
if load_and_convert:

    data = np.load(f'mri/{field}.npy')



    PETSc.Sys.Print(f"Data shape: {data.shape}") 


    if coarseness > 1:
        PETSc.Sys.Print('coarsening image')
        data = zoom(data, (1/coarseness,1/coarseness,1/coarseness), order=0)
        PETSc.Sys.Print(data.shape)

    # create mesh
    PETSc.Sys.Print('building mesh')
    start = time.time()
    mesh = i2d.build_mesh_from_numpy(data, mesh_type='cartesian')
    mesh.name = 'mesh'
    end = time.time()
    PETSc.Sys.Print(f'mesh built {end-start}s')

    # convert to firedrake
    PETSc.Sys.Print('converting image to firedrake')
    start = time.time()
    data_fire = i2d.numpy2firedrake(mesh, data, name=field)
    end = time.time()
    PETSc.Sys.Print(f'image converted {end-start}s')

    out_file = File(f'{field}_c{coarseness}.pvd')
    out_file.write(data_fire)

    PETSc.Sys.Print('saving to file')
    start = time.time()
    fname = f'{field}_c{coarseness}.h5'
    with CheckpointFile(fname, 'w') as afile:
        afile.save_mesh(mesh)  # optional
        afile.save_function(data_fire)
    end = time.time()
    PETSc.Sys.Print(f'saved to file {end-start}s')

PETSc.Sys.Print('loading from file')
fname = f'{field}_c{coarseness}.h5'
start = time.time()
with CheckpointFile(fname, 'r') as afile:
    mesh = afile.load_mesh("mesh")
    field_read = afile.load_function(mesh, f"{field}")
end = time.time()
PETSc.Sys.Print(f'loaded from file {end-start}s')