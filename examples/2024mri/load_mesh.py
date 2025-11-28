from firedrake import *
from firedrake.petsc import PETSc
from time import time
#
# load data
#
comm = COMM_WORLD
n_proc = COMM_WORLD.size
h5_file = f"inputs_nproc{n_proc}.h5"

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
L = - 1e-3 * sink_support * test * dx

solution = Function(V,name="solution")
problem = LinearVariationalProblem(a, L, solution, bcs=[DirichletBC(V, 0.0,\
                                                                    99)])
solver = LinearVariationalSolver(problem,
                                 solver_parameters={
                                     "ksp_type": "cg",
                                     "ksp_rtol": 1e-6,
                                    "pc_type": "hypre"})
solver.solve()

start = time()
VTKFile("direchlet_h5.pvd").write(solution,tof, t1, sink_support, main_network)
PETSc.Sys.Print(f"Saved {time()-start:.2f} seconds")
