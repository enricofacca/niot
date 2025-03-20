import time
start = time.time()
from firedrake import *
from firedrake import PETSc
import argparse
load_packages = time.time() - start
PETSc.Sys.Print(f"Load packages {load_packages}")
ndiv0 = 32

def define_problem_inputs(test_case_number, nref, mesh_type="cartesian", comm=COMM_WORLD):

    if mesh_type == "cartesian":
        quadrilateral = True
        hexahedral = True
    else:
        quadrilateral = False
        hexahedral = False

    if test_case_number == 0:
        #
        # description = {
        #    "domain" : "[0,1]\times[0,1]",
        #    "f": "-\Delta u = 8\pi^2\sin(2\pi x)\cos(2\pi y)",
        #    "bc": {
        #        "bc_left": {
        #            "type" : "dirichlet"
        #             "value": 0,
        #             "domain": "x=0"
        #             },
        #         "bc_right" : {
        #             "type" : "dirichlet"
        #             "value": 0,
        #             "domain": "x=1"
        #         }   
        #     },
        # }
        #
        ndiv = ndiv0 * 2**nref
        mesh = RectangleMesh(ndiv, 4*ndiv, Lx=1.0, Ly=1.0, quadrilateral=quadrilateral, comm=comm)

        x,y = SpatialCoordinate(mesh)
        u_exact = u_exact = sin(2 * pi * x) * cos(2 * pi *y)
        f = -div(grad(u_exact))

        strong_Dirichlet = [
            (u_exact, 1),
            (u_exact, 2),
        ]
        weak_Dirichlet = [
            (u_exact, ds(1), 1.0),
            (u_exact, ds(2), 1.0),
        ]
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet
    
    if test_case_number == 1:
        # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        ndiv = ndiv0 * 2**nref
        mesh1d = UnitIntervalMesh(ndiv, comm=comm)
        ndivy = 2*ndiv
        mesh = ExtrudedMesh(mesh1d, ndivy, layer_height=1.0/ndivy)

        x,y = SpatialCoordinate(mesh)
        u_exact = u_exact = sin(2 * pi * x) * cos(2 * pi *y)
        f = -div(grad(u_exact))

        strong_Dirichlet = [
            (u_exact, "bottom"),
            (u_exact, "top"),
        ]
        weak_Dirichlet = [
            (u_exact, ds_v(1),1.0),
            (u_exact, ds_v(2),1.0),
        ]
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet
    
    if test_case_number == 2:
        # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        ndiv = ndiv0 * 2**nref
        mesh = RectangleMesh(ndiv, ndiv, Lx=1.0, Ly=1.0, quadrilateral=quadrilateral, comm=comm)

        x,y = SpatialCoordinate(mesh)
        u_exact = cos(2 * pi * x) * cos(2 * pi *y)
        f = -div(grad(u_exact))
        
        strong_Dirichlet = []
        weak_Dirichlet = []
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet
    
    if test_case_number == 3:
        # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        ndiv = ndiv0 * 2**nref
        mesh = RectangleMesh(ndiv, ndiv, Lx=1.0, Ly=1.0, quadrilateral=quadrilateral)

        x,y = SpatialCoordinate(mesh)
        u_exact = x**2/2 - x**3/3 - 1/12 + y**2/2 - y**3/3 - 1/12
        f = -div(grad(u_exact))
        
        strong_Dirichlet = []
        weak_Dirichlet = []
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet
    
    if test_case_number == 4:
        if mesh_type == "simplicial":
            return None
        
        # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        ndiv = 4 * 2**nref
        mesh2d = UnitSquareMesh(ndiv, ndiv, quadrilateral=True, comm=comm)
        mesh = ExtrudedMesh(mesh2d, ndiv, layer_height=1.0/ndiv)

        x,y,z = SpatialCoordinate(mesh)
        u_exact = x**2/2 - x**3/3 - 1/12 + y**2/2 - y**3/3 - 1/12 + z**2/2 - z**3/3 - 1/12
        f = -div(grad(u_exact))
        
        strong_Dirichlet = []
        weak_Dirichlet = []
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet
    
    if test_case_number == 5:
        # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        if mesh_type == "simplicial":
            return None
        ndiv = 4 * 2**nref
        mesh2d = UnitSquareMesh(ndiv, ndiv, quadrilateral=quadrilateral)
        mesh = ExtrudedMesh(mesh2d, ndiv, layer_height=1.0/ndiv)
        #mesh = BoxMesh(ndiv, ndiv, ndiv, Lx=1.0, Ly=1.0, Lz=1.0, hexahedral=True)
        mesh.nx = ndiv
        mesh.ny = ndiv
        mesh.nz = ndiv

        x,y,z = SpatialCoordinate(mesh)
        u_exact = x**2/2 - x**3/3 - 1/12 + y**2/2 - y**3/3 - 1/12 + z**2/2 - z**3/3 - 1/12
        f = -div(grad(u_exact))
        
        strong_Dirichlet = []
        weak_Dirichlet = []
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet
    
    
    
    if test_case_number == 6:
        # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        ndiv = 4 * 2**nref
        mesh = BoxMesh(ndiv, ndiv, ndiv, Lx=1.0, Ly=1.0, Lz=1.0, hexahedral=hexahedral, comm=comm)


        x,y,z = SpatialCoordinate(mesh)
        u_exact = x**2/2 - x**3/3 - 1/12 + y**2/2 - y**3/3 - 1/12 + z**2/2 - z**3/3 - 1/12
        f = -div(grad(u_exact))
        
        strong_Dirichlet = []
        weak_Dirichlet = []
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet
    
    if test_case_number == 7:
        # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        if mesh_type == "simplicial":
            return None
        ndiv = 4 * 2**nref
        mesh2d = UnitSquareMesh(ndiv, ndiv, quadrilateral=quadrilateral,comm=comm)
        mesh = ExtrudedMesh(mesh2d, ndiv, layer_height=1.0/ndiv)
        mesh.nx = ndiv
        mesh.ny = ndiv
        mesh.nz = ndiv


        x,y,z = SpatialCoordinate(mesh)
        u_exact = z
        V = FunctionSpace(mesh, "DG", 0)
        f = -div(grad(u_exact))#100*conditional(z>0.4,1,0)*conditional(z<0.6,100,0)

        DG0 = FunctionSpace(mesh, "DG", 0)
        
        strong_Dirichlet = [(u_exact, "bottom"), (u_exact, "top")]
        weak_Dirichlet = [(Constant(0.0), ds_b, 1.0), (1.0, ds_t, 1.0)]
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet

    if test_case_number == 8:
        # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        if mesh_type == "simplicial":
            return None
        ndiv = 200
        mesh2d = UnitSquareMesh(ndiv, ndiv, quadrilateral=quadrilateral, comm=comm)
        mesh = ExtrudedMesh(mesh2d, 5, layer_height=1.0/ndiv)
        #mesh = BoxMesh(ndiv, ndiv, ndiv, Lx=1.0, Ly=1.0, Lz=1.0, hexahedral=True)

        x,y,z = SpatialCoordinate(mesh)
        u_exact = z #- (x-0.5)*(1-z)
        V = FunctionSpace(mesh, "DG", 0)
        # PCG64 random number generator
        pcg = np.random.PCG64(seed=123456789)
        rg = RandomGenerator(pcg)
        # beta distribution
        f = rg.beta(V, 1.0, 2.0)
        #f = -div(grad(u_exact))

        DG0 = FunctionSpace(mesh, "DG", 0)
        marker = Function(DG0)
        marker.interpolate(conditional(x<0.5, 1, 0) * conditional(y<0.2, 1, 0) * conditional(x>0.2, 1, 0))

        print(type(ds_b))
        strong_Dirichlet = [(u_exact, "bottom"), (1, "top")]
        weak_Dirichlet = [(u_exact, ds_b, marker), (1.0, ds_t, 1.0)]
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet


def test_ensemble(nref, n_ensemble, mode="split", slot=0):    
    n_mpi = COMM_WORLD.size
    if mode == "split":
        color = COMM_WORLD.rank % (n_mpi//n_ensemble)
        comm = COMM_WORLD.Split(color)
    if mode == "ensemble":
        my_ensemble = Ensemble(COMM_WORLD, n_ensemble)
        comm = my_ensemble.comm
        color = my_ensemble.ensemble_comm.rank
    
    start_setup = time.time()
    # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
    ndiv = ndiv0 * 2**nref
    mesh = RectangleMesh(ndiv, ndiv, Lx=1.0, Ly=1.0, comm=comm)
    x,y = SpatialCoordinate(mesh)

    R = FunctionSpace(mesh, "R", 0)
    scaling = Function(R)
    if mode == "split":
        scaling.assign(1.0 + comm.rank%n_ensemble)
    if mode == "ensemble":
        scaling.assign(1.0 + my_ensemble.ensemble_comm.rank)
    
    u_exact = (x**2/2 - x**3/3 - 1/12 + y**2/2 - y**3/3 - 1/12)
    f = -div(grad(u_exact))
    
    DG0 = FunctionSpace(mesh, "DG", 0)
    f_h = Function(DG0)
    f_h.interpolate(f)

    extra_u_exact = cos(2 * pi * x) * cos(2 * pi *y)
    extra_f = -div(grad(extra_u_exact))


    space = FunctionSpace(mesh, "CG", 1)
    sol = Function(space, name="u")
    trial = TrialFunction(space)
    test = TestFunction(space)


    A = inner(grad(trial), grad(test)) * dx
    L = f_h * test * dx

    # setup solver
    solver_parameters={ "ksp_type": "cg",
                        "ksp_max_it": 1000,
                        "ksp_rtol": 1e-08, 
                        "ksp_atol": 1e-12,
                        "pc_type": "hypre",
                        #"ksp_monitor_true_residual": None,
                        }
    
    nullspace = VectorSpaceBasis(constant=True,comm=mesh.comm)
    problem = LinearVariationalProblem(A, L, sol)
    solver = LinearVariationalSolver(problem, 
                                    solver_parameters = solver_parameters,
                                    nullspace=nullspace)
    
    setup_time = time.time() - start_setup
    
    # solve problem
    cpu = 0.0
    for i in range(100):
        f_h.interpolate(f + (i+slot)*extra_f)
        tic = time.time()
        solver.solve()
        cpu += time.time()-tic
        PETSc.Sys.Print(f"Color {color} | {i} | CPU {cpu} ", comm=comm)
    ksp_iterations = solver.snes.ksp.getIterationNumber()

    ndof = sol.function_space().dim()
    PETSc.Sys.Print(f"Color {color} | Ndiv {ndiv} - Nref {nref}  Ndofs {ndof} - Iterations {ksp_iterations} | CPU {cpu} | setup_time {setup_time}", comm=comm)


    
    
    if nullspace is not None:
        nullspace.orthogonalize(sol)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nref", type=int, default=2, help="Refinement level")
    parser.add_argument("--ne", type=int, default=1, help="Number of ensembles")
    parser.add_argument("--mode", type=str, default="split", help="split or ensemble")
    parser.add_argument("--slot", type=int, default=0, help="slot number")
    args = parser.parse_args()
    
    test_ensemble(args.nref, args.ne, args.mode, args.slot)

