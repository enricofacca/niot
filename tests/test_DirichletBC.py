from firedrake import *
from firedrake.__future__ import interpolate
from firedrake.petsc import PETSc
from niot import SpaceDiscretization
from niot import utilities
from niot import image2dat as i2d
import numpy as np
import pytest
import itertools



penalty = 1e1
ndiv0 = 8

def define_problem_inputs(test_case_number, nref, mesh_type="cartesian"):
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
        if mesh_type == "simplicial":
            return None
        ndiv = 8 * 2**nref
        mesh2d = UnitSquareMesh(ndiv, ndiv, quadrilateral=quadrilateral)
        mesh = ExtrudedMesh(mesh2d, ndiv, layer_height=1.0/ndiv)
        
        x,y,z = SpatialCoordinate(mesh)
        u_exact = z
        V = FunctionSpace(mesh, "DG", 0)
        f = -div(grad(u_exact))

        DG0 = FunctionSpace(mesh, "DG", 0)
        strong_Dirichlet = [(u_exact, "bottom"), (u_exact, "top")]
        weak_Dirichlet = [(u_exact, ds_b, 1.0), (1.0, ds_t, 1.0)]
        
        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet

    if test_case_number == 8:
        # description = "Domain=[0,1]\time[0,1], zero Dirichlet BCs on x=0, x=1"
        if mesh_type == "simplicial":
            return None
        ndiv = 200
        mesh2d = UnitSquareMesh(ndiv, ndiv, quadrilateral=quadrilateral)
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
    
    if test_case_number == 9:
        tof_full = np.load(f'../examples/2024mri/data/TOF.npy')
        tof_np = tof_full[:,0:160,:10]
        tof_np[tof_np<100] = 0.0
        mesh = i2d.build_mesh_from_numpy(tof_np, mesh_type="cartesian")
        tof = i2d.numpy2firedrake(mesh, tof_np)
        f = tof
        u_exact = Constant(0.0)

        strong_Dirichlet = [(0, "bottom")]
        weak_Dirichlet = [(u_exact, ds_b, tof)]

        return mesh, u_exact, f, strong_Dirichlet, weak_Dirichlet
    
    
    
    if test_case_number == 10:
        #
        # test case for fixing Dirichlet BCs on portion of the boundary
        #
        if mesh_type == "simplicial":
            return None
        ndiv = 16 * 2**nref
        mesh2d = UnitSquareMesh(ndiv, ndiv, quadrilateral=quadrilateral)
        mesh = ExtrudedMesh(mesh2d, ndiv, layer_height=1.0/ndiv)
        
        x,y,z = SpatialCoordinate(mesh)
        u_exact = Constant(0.0)
        V = FunctionSpace(mesh, "DG", 0)
        f = conditional(z>0.6, 1.0, 0.0)*conditional(z<0.8, 1.0, 0.0)
        marker = conditional(x>0.4, 1.0, 0.0)*conditional(x<0.5, 1.0, 0.0) * conditional(y>0.4, 1.0, 0.0)*conditional(y<0.5, 1.0, 0.0)

        DG0 = FunctionSpace(mesh, "DG", 0)
        strong_Dirichlet = [(u_exact, "bottom"),]
        weak_Dirichlet = [(u_exact, ds_b, marker)]
        
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
    
verbose = 1
mesh_types = ["cartesian", "simplicial"]
pot_fems = [("DG",0), ("CG",1), ("CR",1), ]
test_cases = list(range(8))
save_output = True
@pytest.mark.parametrize("mesh_type", mesh_types)
@pytest.mark.parametrize("pot_fem", pot_fems)
@pytest.mark.parametrize("test_case_number", test_cases) 
def test_case(mesh_type, pot_fem, test_case_number):
    PETSc.Sys.Print(f"Test case number: {test_case_number} - Mesh type: {mesh_type} - Space: {pot_fem}")
    hs=[]
    errorsL2 = []
    for nref in range(4):
        hs.append(1.0/(ndiv0 * 2**nref))
        
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

        h_mode = "cell_over_facet"
        SD = SpaceDiscretization(mesh, space, degree, h_mode=h_mode)
        

        # set Poisson problem
        u = Function(SD.pot_space, name="u")
        a = SD.Laplacian_form(SD.pot_space)
        energy_form = SD.Laplacian_Lagrangian(u)
        PDE = derivative(energy_form, u)
        a = derivative(PDE, u)

        test = TestFunction(SD.pot_space)
        L = f * test * dx

        # fix boundary conditions
        if space == "DG":
            if verbose > 0:
                PETSc.Sys.Print(f"Imposing Dirichlet weakly :")
            # impose weak Dirichlet BCs
            a = SD.apply_weak_Dirichlet_lhs(weak_Dirichlet, a)# penalty=1e6)
            L = SD.apply_weak_Dirichlet_rhs(weak_Dirichlet, L)#, penalty=1e6)
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
                            "ksp_monitor_true_residual": None,
                            }
        problem = LinearVariationalProblem(a, L, u, bcs=bcs)
        solver = LinearVariationalSolver(problem, 
                                         solver_parameters=solver_parameters,
                                        nullspace=nullspace)
        
        # solve problem
        solver.solve()
        #solver.snes.ksp.view()
        ksp_iterations = solver.snes.ksp.getIterationNumber()
        if nullspace is not None:
            nullspace.orthogonalize(u)


        # check error
        error_function = Function(SD.pot_space, name="error")
        error_function.interpolate(u - u_exact)
            
        error = np.sqrt(assemble(error_function**2 * dx))
        with error_function.dat.vec_ro as x:
            ndof = x.getSize()
        PETSc.Sys.Print(f"Number DOF {ndof:07} -L^2 error: {error:.2e} - KSP iterations: {ksp_iterations}")
        errorsL2.append(error)

        f_h = Function(SD.pot_space, name="forcing")
        f_h.interpolate(f)


        exact_function = Function(SD.pot_space, name="exact")
        exact_function.interpolate(u_exact)

        if save_output:    
            out_file_name  = f"dirichlet_test/output_nref{nref:02}.pvd"
            utilities.save2pvd([u, exact_function, error_function,f_h], out_file_name)

    hs = np.array(hs)
    errorsL2 = np.array(errorsL2)

    check_convergences(hs,errorsL2,expected_rate=1.0)


if __name__ == "__main__":
    mesh_types = ["cartesian"]
    pot_fems = [("DG",0)]
    test_cases = [10]
    combinations = itertools.product(mesh_types, pot_fems, test_cases)
    for mesh_type, pot_fem, test_case_number in combinations:
        test_case(mesh_type, pot_fem, test_case_number)
