from firedrake import *
from firedrake.__future__ import interpolate
from firedrake.petsc import PETSc
from niot import SpaceDiscretization
from niot import utilities
import numpy as np
import pytest
import itertools
import gc


penalty = 1e1
ndiv0 = 8

def define_problem_inputs(test_case_number, nref, mesh_type="cartesian",comm=COMM_WORLD):
    if mesh_type == "cartesian":
        quadrilateral = True
        hexahedral = True
    else:
        quadrilateral = False
        hexahedral = False

    if test_case_number == 0:
        #description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        ndiv = ndiv0 * 2**nref
        mesh = RectangleMesh(ndiv, 4*ndiv, Lx=1.0, Ly=1.0, quadrilateral=quadrilateral)
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
        mesh1d = UnitIntervalMesh(ndiv)
        mesh = ExtrudedMesh(mesh1d, ndiv, layer_height=1.0/ndiv)

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
        mesh = RectangleMesh(ndiv, ndiv, Lx=1.0, Ly=1.0, quadrilateral=quadrilateral)

        x,y = SpatialCoordinate(mesh)
        u_exact = cos(2 * pi * x) * cos(2 * pi *y)
        f = -div(grad(u_exact))
        
        strong_Dirichlet = []
        weak_Dirichlet = []
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet
    
    if test_case_number == 3:
        # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        ndiv = ndiv0 * 2**nref
        #mesh = RectangleMesh(ndiv, ndiv, Lx=1.0, Ly=1.0, quadrilateral=quadrilateral)

        mesh1d = UnitIntervalMesh(ndiv)
        mesh = ExtrudedMesh(mesh1d, ndiv, layer_height=1.0/ndiv)


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
        mesh2d = UnitSquareMesh(ndiv, ndiv, quadrilateral=True)
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

        x,y,z = SpatialCoordinate(mesh)
        u_exact = x**2/2 - x**3/3 - 1/12 + y**2/2 - y**3/3 - 1/12 + z**2/2 - z**3/3 - 1/12
        f = -div(grad(u_exact))
        
        strong_Dirichlet = []
        weak_Dirichlet = []
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet
    
    if test_case_number == 6:
        # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        ndiv = 4 * 2**nref
        mesh = BoxMesh(ndiv, ndiv, ndiv, Lx=1.0, Ly=1.0, Lz=1.0, hexahedral=hexahedral)

        x,y,z = SpatialCoordinate(mesh)
        u_exact = x**2/2 - x**3/3 - 1/12 + y**2/2 - y**3/3 - 1/12 + z**2/2 - z**3/3 - 1/12
        f = -div(grad(u_exact))
        
        strong_Dirichlet = []
        weak_Dirichlet = []
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet
    

    if test_case_number == 7:
        # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        ndiv = 32 * 2**nref

        # create size partions 
        # as a list of number of vertical layers for each partition
        size_partitions = [ndiv // comm.size]*comm.size
        size_partitions[0] += ndiv % comm.size
        size_partitions = [ndiv-4,4]


        if comm.rank == 0:
            partitions = (size_partitions, list(range(ndiv)))  
        else:
            partitions = (None, None) 
        
        PETSc.Sys.Print(f"{ndiv=} - size_partitions: {size_partitions}")

        distribution_parameters = {
            "partition": partitions,
            "overlap_type": (DistributedMeshOverlapType.FACET, 1)}
        
        #if comm.size == 1:
        #distribution_parameters = None

        mesh1d = UnitIntervalMesh(ndiv, distribution_parameters=distribution_parameters)
        mesh = ExtrudedMesh(mesh1d, ndiv, layer_height=1.0/ndiv)

        # x,y = SpatialCoordinate(mesh)
        # u_exact = u_exact = sin(2 * pi * x) * cos(2 * pi *y)
        # f = -div(grad(u_exact))

        # strong_Dirichlet = [
        #     (u_exact, "bottom"),
        #     (u_exact, "top"),
        # ]
        # weak_Dirichlet = [
        #     (u_exact, ds_v(1),1.0),
        #     (u_exact, ds_v(2),1.0),
        # ]

        x,y = SpatialCoordinate(mesh)
        u_exact = x**2/2 - x**3/3 - 1/12 + y**2/2 - y**3/3 - 1/12
        f = -div(grad(u_exact))
        
        strong_Dirichlet = []
        weak_Dirichlet = []
        
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet





def check_convergences(hs,errors,expected_rate):
    hs = np.array(hs)
    errors = np.array(errors)

    if (errors<1e-12).all():
        return

    # compute convergence rate fitting a line to the log-log plot
    log_hs = np.log(hs)
    log_errors = np.log(errors)
    slope, intercept = np.polyfit(log_hs, log_errors, 1)
    PETSc.Sys.Print(f"Convergence rate: {slope:.2f}")
    
    assert slope > expected_rate - 0.1
    
verbose = 0
mesh_types = ["cartesian", "simplicial"]
pot_fems = [("DG",0), ("CG",1), ("CR",1), ]
test_cases = list(range(6))
save_output = False
@pytest.mark.parametrize("mesh_type", mesh_types)
@pytest.mark.parametrize("pot_fem", pot_fems)
@pytest.mark.parametrize("test_case_number", test_cases) 
def test_case(mesh_type, pot_fem, test_case_number,total_ref=4):
    beta = penalty*10**(2)
    PETSc.Sys.Print(f"Test case number: {test_case_number} - Mesh type: {mesh_type} - Space: {pot_fem}")
    #PETSc.Sys.Print("beta:", beta)
    hs=[]
    errorsL2 = []
    for nref in range(total_ref):
        hs.append(1.0/(ndiv0 * 2**nref))
        

        mesh = None
        gc.collect()

        # define problem inputs and reference solution
        inputs = define_problem_inputs(test_case_number, nref, mesh_type=mesh_type)
        if inputs is None:
            break
        else:
            mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet = inputs
        
        # define space discretization
        space, degree = pot_fem
        if degree > 0 and hasattr(mesh,"extruded"):
            break
        
        if mesh.ufl_cell().is_simplex():
            if degree==0:
                break
        else:
            if degree>0:
                break
        SD = SpaceDiscretization(mesh, space, degree)
        

        # set Poisson problem
        u = Function(SD.pot_space, name="u")
        a = SD.Laplacian_form(SD.pot_space)
        test = TestFunction(SD.pot_space)
        L = f * test * dx

        # fix boundary conditions
        if space == "DG":
            if verbose > 0:
                PETSc.Sys.Print(f"Imposing Dirichlet waekly : penalty={beta}")
            # impose weak Dirichlet BCs
            a = SD.apply_weak_Dirichlet_lhs(weak_Dirichlet, a, penalty=beta)
            L = SD.apply_weak_Dirichlet_rhs(weak_Dirichlet, L, penalty=beta)
            bcs = None
        else:
            # impose strong Dirichlet BCs
            if verbose > 0:
                PETSc.Sys.Print(f"Imposing Dirichlet strongly")
            bcs = []
            for u_D, id_boundary in strong_Dirichlet:
                bc = DirichletBC(SD.pot_space, u_D, id_boundary)
                bcs.append(bc)
            

        # set nullspace (if no Dirichlet BCs)
        nullspace = None
        if len(strong_Dirichlet) == 0 and len(weak_Dirichlet) == 0:
            if verbose > 0:
                PETSc.Sys.Print(f"Setting Nullspace for Pure Neumann problem") 
            bcs = None
            nullspace = VectorSpaceBasis(constant=True,comm=mesh.comm)
        

        # setup solver
        solver_parameters={ "ksp_type": "cg",
                            "ksp_max_it": 1000,
                            "ksp_rtol": 1e-13, 
                            "ksp_atol": 1e-13,
                            "pc_type": "hypre",
                            }
        problem = LinearVariationalProblem(a, L, u, bcs=bcs)
        solver = LinearVariationalSolver(problem, solver_parameters=solver_parameters, nullspace=nullspace)
        
        # solve problem
        solver.solve()
        ksp_iterations = solver.snes.ksp.getIterationNumber()
        if nullspace is not None:
            nullspace.orthogonalize(u)


        # check error
        error_function = Function(SD.pot_space, name="error")
        error_function.interpolate(u - u_exact)
            
        error = np.sqrt(assemble(error_function**2 * dx))
        PETSc.Sys.Print(f"Number of cell {mesh.num_cells():07} -L^2 error: {error:.2e} - KSP iterations: {ksp_iterations}")
        errorsL2.append(error)

        if save_output:
            f_h = Function(SD.pot_space, name="forcing")
            f_h.interpolate(f)


            exact_function = Function(SD.pot_space, name="exact")
            exact_function.interpolate(u_exact)
            
            out_file_name  = f"poisson_test/output_nref{nref:02}.pvd"
            utilities.save2pvd([u, exact_function, error_function,f_h], out_file_name)

    hs = np.array(hs)
    errorsL2 = np.array(errorsL2)

    check_convergences(hs,errorsL2,expected_rate=1.0)


if __name__ == "__main__":
    # parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default="cartesian", help="mesh type: cartesian, simplicial")
    parser.add_argument("--nref", type=int, default=0, help="number of uniform refinements")
    parser.add_argument("--fems", type=str, default="all", help="finite element spaces to test: all, DG0, CG1, CR1")
    parser.add_argument("--test", type=int, default=0, help="test cases to run: all, 0,1,2,...")
    args = parser.parse_args()

    mesh_type = args.mesh
    #nref = args.nref
    fems = args.fems
    # split fems
    pot_fem = (fems[0:2],int(fems[2]))
    print(pot_fem)
    test_cases_number = args.test
    

    #combinations = itertools.product(mesh_types, pot_fems, test_cases)
    #for mesh_type, pot_fem, test_case_number in combinations:
    test_case(mesh_type, pot_fem, test_cases_number,total_ref=args.nref+1)



class DG0Laplacian(AuxiliaryOperatorPC):
    def form(self, pc, test, trial):
        context = pc.get_appctx()
        h_len = context["h_len"]
        h_size = context["h_size"]
        d_interior = context["d_interior"]
        a = jump(trial) * jump(test) / h_len * d_interior

        # impose weak Dirichlet BCs
        beta = context["penalty"]
        weak_Dirichlet = context["weak_Dirichlet"]
        
        for u_D, dface in weak_Dirichlet:
            a += beta / h_size * test * trial  * dface

        return a, None

mixed = False
if mixed:
    RT= FunctionSpace(mesh, 'RT', 1)
    u_grad = Function(RT)

    M = RT * SD.pot_space
    test = TestFunction(M)
    trial = TrialFunction(M)
    vel_test, pot_test = test
    vel_trial, pot_trial = trial

    a = (inner(vel_trial, vel_test) 
        - div(vel_trial) * pot_test 
        + div(vel_test) * pot_trial) * dx
    L = forcing * pot_test * dx


local = False
if local:
    #h = FacetArea(mesh) / CellVolume(mesh)
    #h_size = Function(SD.pot_space, name="h_size")
    #cellsize_mesh = CellSize(mesh)
    #h_size = 1.0/ (FacetArea(mesh) / CellVolume(mesh))
    
    #h_size.interpolate(cellsize_mesh)
    #h_len = h_length(SD.pot_space,'cell_distance')
    
    

    #dface_interior = d_face_interior(mesh)
    #dface_exterior = d_face_exterior(mesh)

    
    
    u = Function(SD.pot_space, name="u")
    test = TestFunction(SD.pot_space)
    trial = TrialFunction(SD.pot_space)
    h_len = SD.delta_h
    h_len = 1.0 / avg(FacetArea(mesh) / CellVolume(mesh))
    #a = jump(trial) * jump(test) / h_len * dface_interior
    
    L = f * test * dx

    A = assemble(a)


    R = FunctionSpace(mesh, 'R', 0)
    one = Function(R).assign(1)
