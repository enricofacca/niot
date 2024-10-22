from firedrake import *
from firedrake.__future__ import interpolate
import time
from firedrake import File as VTKFile

# fit the error
import numpy as np
from scipy.stats import linregress



def test_case(cubic_mesh, test_case_name):
    if test_case_name == "neumann_1":
        x, y, z = SpatialCoordinate(cubic_mesh)
        f = 1.5 - (x + y + z)
        bc_Dirichlet = None
        bc_Neumann = None#[as_vector(np.zeros(mesh.topological_dimension()-1)),"on_boundary"]
        u_exact = (x**3 + y**3 +z**3 - 3 * 1/4 )/ 6 - (x**2+y**2+z**2-3*1/3)/4
        return f, bc_Dirichlet, bc_Neumann, u_exact
    elif test_case_name == "dirichlet":
        raise NotImplementedError

def build_mesh(ndiv_x, ndiv_y, ndiv_z,
               L_x=1, L_y=1, L_z=1):
    mesh2d = RectangleMesh(ndiv_x, ndiv_y, L_x, L_y, quadrilateral=True)
    mesh = ExtrudedMesh(mesh2d, ndiv_z, layer_height=L_z/ndiv_z)
    return mesh

ndiv0 = 8
nref = 3

l2_errors = []
for level in range(nref+1):
    print("")
    print(f"nref: {nref}")

    ndiv = ndiv0*2**level
    ndivx = ndiv*1
    ndivy = ndiv*2
    ndivz = ndiv*4
    tic = time.time()
    mesh = build_mesh(ndivx, ndivy, ndivz)
    stop = time.time()
    #print(f"mesh3d: {stop-tic}")

    f, bc_Dirichlet, bc_Neumann, u_exact = test_case(mesh, "neumann_1")


    # piecewise constant function space
    DG0 = FunctionSpace(mesh, "DG", 0)
    # solution
    u_h = Function(DG0,name="pot")
    test = TestFunction(DG0)
    trial = TrialFunction(DG0)
    
    # delta_h is required for scale
    x, y, z = SpatialCoordinate(mesh)
    x_func = assemble(interpolate(x, DG0))
    y_func = assemble(interpolate(y, DG0))
    z_func = assemble(interpolate(z, DG0))
    delta_h = sqrt(jump(x_func)**2 
                        + jump(y_func)**2 
                        + jump(z_func)**2)

    # integration on the interior facets
    dS = dS_v + dS_h



    tic = time.time()
    a = jump(test) * jump(trial) / delta_h * dS
    L = f * test * dx

    nullspace = VectorSpaceBasis(constant=True, comm=mesh.comm)
    solver_parameters = {
        "ksp_type": "cg",
        "pc_type": "hypre",
    }
    solve(a == L, u_h,
        solver_parameters=solver_parameters,
            nullspace=nullspace)


    l2_error = errornorm(u_exact, u_h, norm_type="L2")
    l2_errors.append(l2_error)
    print(f"l2_error: {l2_error:.2e}")
    #h1_error = errornorm(u_exact, u_h, norm_type="H1")
    #print(f"h1_error: {h1_error:.2e}")


    #f_h = Function(DG0,name="forcing").interpolate(1.5-z-y-x)

    #out_file = VTKFile("test.pvd")
    #out_file.write(u_h,f_h)



hs = np.array([1/ndiv0*2**i for i in range(nref+1)])
errors = np.array(l2_errors)

# get the slope of the log-log plot


slope, intercept, r_value, p_value, std_err = linregress(np.log(hs), np.log(errors))
print(f"slope: {slope}")