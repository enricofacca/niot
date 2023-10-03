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
