# Author: Enrico Facca
# Date: 2023-09-15T15:45:04+02:00
# -----
#
# Description:
#   This file contains the implementation of the smoothing operator
#   to transform a conductivity field into an image. 
#   Both are implemented as pyadjoint blocks.
# """

from firedrake import FunctionSpace
from firedrake import Function, TestFunction, TrialFunction
from firedrake import assemble
from firedrake import dx

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

import firedrake.adjoint as fire_adj


def smooth(func, LaplacianSmoother):
    """
    Return a smoothed version of the function `func` 
    by solving, for one time step, the heat equation 
    dt u - Laplace(u) = 0 
    u(0) = func
    This is applying the inverse of the matrix
    (I - dt Laplace)
    args:
        func: a Function 
        LaplacianSmoother: LinSol
    """
    # TODO we should check if func is a Function of 
    # LaplacianSmoother.function_space()    
    smoothed = Function(func.function_space())
    test = TestFunction(func.function_space())
    rhs = assemble(func * test * dx)
    with rhs.dat.vec as y_vec, smoothed.dat.vec as out:
        LaplacianSmoother.solve(y_vec, out)
    print('SMOOTHING INFO',LaplacianSmoother.info())
    return smoothed
    
backend_smooth = smooth


class SmootherBlock(Block):
    def __init__(self, func, LaplacianSmoother, **kwargs):
        """
        Smoothing with (I+delta_t Laplace)^{-1}.
        The variable LaplacianSmoother is a PETSc solver
        """
        super(SmootherBlock, self).__init__()
        self.kwargs = kwargs
        self.add_dependency(func)
        self.V = func.function_space()
        self.LaplacianSmoother = LaplacianSmoother
        print('init SmootherBlock')
    def __str__(self):
        return "SmootherBlock"

    def evaluate_adj_component(self, inputs, adj_inputs, block_variable, idx, prepared=None):
        """
        We must return the adjoint of the smoothing operator, that is 
        a dual form
        """
        # This maybe a 1-form or a cofunction, but we return a function
        out = Function(self.V)
        with adj_inputs[0].dat.vec as rhs, out.dat.vec as out_vec:
            self.LaplacianSmoother.solve(rhs, out_vec)
            print('ADJ application',self.LaplacianSmoother.info())

        test = TestFunction(self.V)
        v = assemble(out*test*dx)

        print('utils',type(v))
        #with v.dat.vec as v_vec:
        #return v_vec
        return v

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_smooth(inputs[0], self.LaplacianSmoother)

LaplacianSmoothing = overload_function(smooth, SmootherBlock)


######################################################################
# Nonlinear smoothing via "porous medium equation" (porrous media for the mathematician)
######################################################################
def spread(density, tau=0.1, m_exponent=1, nsteps=1):
    '''
    Spread a density using the porous medium equation,
    :math:`\partial_t \rho - \nabla \cdot (\rho^m-1 \nabla \rho)=0`.
    '''
    # create the function space
    mesh = density.function_space().mesh()
    space = FunctionSpace(mesh, 'CG', 1)
    log_density_initial = Function(space)
    log_density_initial.interpolate(ln(density))
    log_density = cp(log_density_initial)

    nstep = 1
    while nstep <= nsteps:
        # define the PDE
        # m=1 is the heat equation 
        test = TestFunction(space)
        PDE = ( 
            1/tau * (exp(log_density) - exp(log_density_initial)) * test  * dx 
            + m_exponent * exp(m_exponent * log_density_initial) * inner(grad(log_density), grad(test)) * dx
            )
        problem = NonlinearVariationalProblem(PDE, log_density)

        # set solver. One Newton step is required for m=1
        ctrl={
                'snes_rtol': 1e-16,
                'snes_atol': 1e-4,
                'snes_stol': 1e-16,
                'snes_type': 'newtonls',
                'snes_linesearch_type':'bt',
                'snes_max_it': 20,
                'snes_monitor': None,
                # krylov solver controls
                'ksp_type': 'gmres',
                'ksp_atol': 1e-30,
                'ksp_rtol': 1e-8,
                'ksp_divtol': 1e4,
                'ksp_max_it' : 100,
                # preconditioner controls
                'pc_type': 'hypre',
            }
        snes_solver = NonlinearVariationalSolver(problem, solver_parameters=ctrl)
        
        # solve the problem
        try:
            snes_solver.solve()
            ierr = 0
        except:
            ierr = snes_solver.snes.getConvergedReason()

        if (ierr != 0):
            print('ierr',ierr)
            print(f' Failure in due to {SNESReasons[ierr]}')
            raise ValueError('Newton solver failed')
        
        nstep += 1
        log_density_initial.assign(log_density)

    # return the solution in the same function space of the input
    smooth_density = Function(density.function_space())
    smooth_density.interpolate(exp(log_density))

    return smooth_density