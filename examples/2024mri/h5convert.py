from firedrake import *
from firedrake.petsc import PETSc
from time import time
import sys

#
# load data
#
comm = COMM_WORLD
n_proc = COMM_WORLD.size
h5_file = sys.argv[1]
PETSc.Sys.Print(f"Loading data from {h5_file}")
with CheckpointFile(h5_file, 'r',comm=comm) as afile:
    start = time()
    PETSc.Sys.Print(f"Start mesh ")
    mesh = afile.load_mesh("relabeled_mesh")
    PETSc.Sys.Print(f" Loaded mesh in {time()-start:.2f} seconds")
    
    start = time()
    tdens = afile.load_function(mesh, "tdens")
    PETSc.Sys.Print(f" Loaded tdens in {time()-start:.2f} seconds")
    
    start = time()
    pot = afile.load_function(mesh, "pot")
    PETSc.Sys.Print(f" Loaded pot in {time()-start:.2f} seconds")

out_file = sys.argv[2]
# save as pvd
PETSc.Sys.Print(f"Saving data to {out_file}")
VTKFile(out_file).write(tdens, pot)