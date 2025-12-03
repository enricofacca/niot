from firedrake import *
from firedrake.petsc import PETSc
from time import time
import os
import sys
import nibabel
import numpy as np
#
# load data
#
out_directory = sys.argv[1]
comm = COMM_WORLD
n_proc = COMM_WORLD.size
h5_file = os.path.join(out_directory, f"inputs_{n_proc:04d}.h5")

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

    
V = FunctionSpace(mesh, "CG", 1)
test = TestFunction(V)
trial = TrialFunction(V)
a = inner(grad(trial), grad(test)) * dx
L = 1e-3 * sink_support * test * dx

solution = Function(V,name="solution")
problem = LinearVariationalProblem(a, L, solution, bcs=[DirichletBC(V, 0.0,\
                                                                    99)])
solver = LinearVariationalSolver(problem,
                                 solver_parameters={
                                     "ksp_type": "cg",
                                     "ksp_rtol": 1e-6,
                                    "pc_type": "hypre",
                                    "ksp_monitor_true_residual": None})
solver.solve()

main_file = os.path.join(out_directory, f"main_network.nii.gz")
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


cartesian_mesh = i2d.create_cartesian_mesh(dimensions, lengths, comm=comm)
PETSc.Sys.Print("Shifting coordinates of relabeled mesh")
mesh.coordinates.dat.data[:, 0] -= offset[0]
mesh.coordinates.dat.data[:, 1] -= offset[1]
mesh.coordinates.dat.data[:, 2] -= offset[2]
PETSc.Sys.Print("Offset completed")


V = FunctionSpace(mesh, "CG", 1)
test = TestFunction(V)
trial = TrialFunction(V)
a = inner(grad(trial), grad(test)) * dx
L = sink_support_mesh * test * dx

start = time.time()
solution = Function(V,name="solution")
problem = LinearVariationalProblem(a, L, solution, bcs=[DirichletBC(V, 0.0, 99)])
solver = LinearVariationalSolver(problem,
                        solver_parameters={
                            "ksp_type": "cg",
                            "ksp_rtol": 1e-6,
                            "pc_type": "hypre",
                            "ksp_monitor_true_residual": None})

solver.solve()
PETSc.Sys.Print(f"Dirichlet BC test solve completed in {time.time()-start:.2e} s")
data4pvd.append(solution)
DG0_Cartesian = FunctionSpace(cartesian_mesh, "DG", 0)
solution_cartesian = assemble(interpolate(solution, DG0_Cartesian, allow_missing_dofs=True,  default_missing_val=-99))
solution_np = i2d.firedrake2numpy(solution_cartesian)
if cartesian_mesh.comm.rank == 0:
    outfilename = os.path.join(out_directory,f"solution_dirichlet.nii.gz")
    nibabel.save(nibabel.Nifti1Image(solution_np, affine), outfilename)
cartesian_mesh.comm.barrier()
