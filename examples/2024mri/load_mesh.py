from firedrake import *
from firedrake.petsc import PETSc
from time import time
import os
import sys
import nibabel
import numpy as np
from mesh2h5 import bounding_box
from niot import image2dat as i2d


#
# load data
#
out_directory = sys.argv[1]
comm = COMM_WORLD
n_proc = COMM_WORLD.size
h5_file = os.path.join(out_directory, f"inputs_nproc{n_proc}.h5")
PETSc.Sys.Print(f"Loading data from {h5_file}")
with CheckpointFile(h5_file, 'r',comm=comm) as afile:
    start = time()
    PETSc.Sys.Print(f"Start mesh ")
    mesh = afile.load_mesh("relabeled_mesh")
    PETSc.Sys.Print(f" Loaded mesh in {time()-start:.2f} seconds")
    start = time()
    tof = afile.load_function(mesh, "tof")
    PETSc.Sys.Print(f" Loaded tof in {time()-start:.2f} seconds")
    start = time()
    t1 = afile.load_function(mesh, "t1")
    PETSc.Sys.Print(f" Loaded t1 in {time()-start:.2f} seconds")

    start = time()
    main_network = afile.load_function(mesh, "main_network")
    PETSc.Sys.Print(f" Loaded main_network in {time()-start:.2f} seconds")
    start = time()
    sink_support = afile.load_function(mesh, "sink_support")
    PETSc.Sys.Print(f" Loaded sink in {time()-start:.2f} seconds")

    
main_file = os.path.join(out_directory, f"preprocessed/main_network.nii.gz")
main_data = nibabel.load(main_file)
dimensions = main_data.header.get_data_shape()[:3]

print(f"Data shape: {dimensions=}")
hx, hy, hz = main_data.header['pixdim'][1:4]
lengths = np.array([float(dimensions[0]*hx), 
                    float(dimensions[1]*hy), 
                    float(dimensions[2]*hz)])
voxel_size = (hx, hy, hz)
affine = main_data.affine
offset = affine[:3, 3]


cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths)
PETSc.Sys.Print("Shifting coordinates of relabeled mesh")
mesh.coordinates.dat.data[:, 0] -= offset[0]
mesh.coordinates.dat.data[:, 1] -= offset[1]
mesh.coordinates.dat.data[:, 2] -= offset[2]
PETSc.Sys.Print("Offset completed")
lower, upper = bounding_box(mesh)
PETSc.Sys.Print(f" {lower[0]:.2e}<= x <>{upper[0]:.2e}. Lx={lengths[0]:.2e}")
PETSc.Sys.Print(f" {lower[1]:.2e}<= y <>{upper[1]:.2e}. Ly={lengths[1]:.2e}")
PETSc.Sys.Print(f" {lower[2]:.2e}<= z <>{upper[2]:.2e}. Lz={lengths[2]:.2e}")


V = FunctionSpace(mesh, "CG", 1)
R = FunctionSpace(mesh, "R", 0)
test = TestFunction(V)
trial = TrialFunction(V)
const = Function(R, name="const")
const.assign(0.0)
a = (1+const*main_network)*inner(grad(trial), grad(test)) * dx
L = sink_support * test * dx

solution = Function(V,name="solution")
problem = LinearVariationalProblem(a, L, solution, bcs=[DirichletBC(V, 0.0, 99)])
solver = LinearVariationalSolver(problem,
                        solver_parameters={
                            "ksp_type": "cg",
                            "ksp_rtol": 1e-6,
                            "pc_type": "hypre",
                            # tuning parameters for the multigrid 
                            # https://mooseframework.inl.gov/releases/moose/2021-09-15/application_development/hypre.html
                            "pc_hypre_type": "boomeramg",
                            "pc_hypre_boomeramg_strong_threshold": 0.7,
                            "pc_hypre_boomeramg_max_iter": 1,
                            "pc_hypre_boomeramg_agg_nl": 4,
                            "pc_hypre_boomeramg_agg_num_paths": 2
                            "pc_hypre_boomeramg_interp_type": "ext+i",  # "classic" or "ext+i"
                                "ksp_monitor_true_residual": None})



for i in range(4):
    start = time()
    const.assign(4**i)
    solver.solve()
    PETSc.Sys.Print(f"Dirichlet BC test solve completed in {time()-start:.2e} s")

DG0_Cartesian = FunctionSpace(cartesian_mesh, "DG", 0)
solution_cartesian = assemble(interpolate(solution, DG0_Cartesian, allow_missing_dofs=True,  default_missing_val=-99))
solution_np = i2d.firedrake2numpy(solution_cartesian)
if cartesian_mesh.comm.rank == 0:
    outfilename = os.path.join(out_directory,f"solution_dirichlet.nii.gz")
    nibabel.save(nibabel.Nifti1Image(solution_np, affine), outfilename)
cartesian_mesh.comm.barrier()
