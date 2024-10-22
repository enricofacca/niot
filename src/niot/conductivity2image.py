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
from firedrake import Function, TestFunction, TrialFunction # this are methods
from firedrake import assemble, inner, grad, jump, dx, dS
#from firedrake import function # this is the class
from firedrake import derivative

from firedrake import LinearVariationalProblem, LinearVariationalSolver
from firedrake import NonlinearVariationalProblem, NonlinearVariationalSolver
from firedrake import File

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

import firedrake.adjoint as fire_adj
#fire_adj.continue_annotation()

from firedrake.petsc import PETSc

from firedrake.__future__ import interpolate

from . import utilities
from . import linear_algebra_utilities as linalg

import numpy as np
from scipy.integrate import quad
from scipy.special import gamma
from scipy.optimize import newton

class Conductivity2ImageMap:
    """
    Base class for defining the interface of a map from conductivity to image
    """
    def __init__(self, space, scaling=1.0, **kargs) -> None:
        raise NotImplementedError('Conductivity2Image is not implemented')
    def __call__(self, conductivity, **kargs) -> Function:
        raise NotImplementedError('Conductivity2Image is not implemented')
    
class IdentityMap(Conductivity2ImageMap):
    """
    Simply scales the conductivity by a factor
    """
    def __init__(self, space, scaling=1.0) -> None:
        self.space = space
        self.scaling = scaling
        self.image_h = Function(space)
        self.cd = {self.image_h: self.scaling}
    def __call__(self, conductivity, **kargs) -> Function:
        #self.image_h = assemble(interpolate(self.scaling * conductivity, self.space))
        self.image_h.assign(self.scaling * conductivity)#
        return self.image_h
        
class HeatMap(Conductivity2ImageMap):
    """
    The image is the solution of one time step 
    the heat equation
    (u - u_0) /sigma  - \Delta u = 0
    with u_0 = conductivity 

    plus a scaling factor multiplication.
    """
    def __init__(self, space, scaling=1.0, sigma=1e-2) -> None:
        self.space = space
        self.scaling = scaling
        self.sigma = sigma

        self.image_h = Function(space)
        self.tdens4transform = Function(space)
        
        test = TestFunction(space)  
        trial = TrialFunction(space)
        form =  inner(test, trial) * dx # mass matrix
        if space.ufl_element().degree() > 0:
            form += sigma * grad(test) * grad(trial) * dx
        else:
            if space.mesh().ufl_cell().is_simplex():
                raise NotImplementedError('Laplacian with DG0 simplices is not implemented')
            delta_h = utilities.delta_h(space)
            form += sigma * jump(test) * jump(trial) / delta_h * dS

        # 1-form for the heat equation
        self.rhs_heat = self.tdens4transform * test * dx
        self.heat_problem = LinearVariationalProblem(form, 
                                                self.rhs_heat,
                                                self.image_h, 
                                                constant_jacobian=True)
                
        self.heat_solver = LinearVariationalSolver(
            self.heat_problem,
            solver_parameters={
                'ksp_type': 'cg',
                'ksp_rtol': 1e-10,
                'ksp_initial_guess_nonzero': True,
                #'ksp_monitor_true_residual': None,
                'pc_type': 'hypre',
                },
            options_prefix='heat_solver_')

    def __call__(self, conductivity, **kargs) -> Function:
        # image and tdens are expected to be close (up to scaling)
        self.image_h = assemble(interpolate(conductivity / self.scaling, self.space))
                
        # image = M * tdens
        self.tdens4transform = assemble(interpolate(conductivity, self.space))
                
        # invoce the solver
        self.heat_solver.solve()
                
        # we scale here so we return a function 
        # otherwise (scaling * image) is an expression
        self.image_h *= self.scaling

        return self.image_h


class Barenblatt():
    """
    Some formulas for the Barenblatt solution from
    @book{fasano2006problems,
        title={Problems in Nonlinear Diffusion: Lectures Given at the 2nd 1985 Session of the Centro Internazionale Matematico Estivo (CIME) Held at Montecatini Terme, Italy, June 10-June 18, 1985},
        author={Fasano, Antonio and Primicerio, Mario},
        volume={1224},
        year={2006},
        publisher={Springer}
    }
    """
    def __init__(self,exponent_m, dim) -> None:
        self.exponent_m = exponent_m
        self.dim = dim

        self.beta = self.eval_beta(self.exponent_m, self.dim)
        self.alpha = self.eval_alpha(self.exponent_m, self.dim)
        self.B = self.eval_B(self.exponent_m, self.dim)
        self.K_md = self.eval_K_md(self.exponent_m, self.dim)


    def eval_alpha(self,exponent_m, dim):
        return self.eval_beta(exponent_m, dim) * dim

    def eval_beta(self, exponent_m, dim):
        return 1 / (2 + (exponent_m - 1 ) * dim)

    def eval_B(self, exponent_m, dim):
        return self.eval_beta(exponent_m, dim) * (exponent_m - 1 )/ (2 * exponent_m)
    
    def eval_K_md(self, exponent_m, dim):
        """+
        Constant depending on m and d only in eq .5 pag.3 such that
        K_md A^{1/(2\beta(m-1))} = M
        with M the mass of the initial condition u(x,0)=M \delta(x)
        """
        beta = self.eval_beta(exponent_m, dim)
        B = self.eval_B(exponent_m, dim)
        # sphere volume
        wd_d = np.pi**(dim/2)/gamma(dim/2 + 1)
        
        # compute the integral numerically
        def integrand(theta,m,d):
            return np.cos(theta)**((m+1)/(m-1)) * np.sin(theta)**(d-1)
            
        integral, err_estimate = quad(integrand, 0, np.pi/2, args=(float(exponent_m),float(dim))) 
        if err_estimate > 1e-6:
            raise ValueError('Integral not computed with sufficient accuracy')
        return (wd_d * ( B ** (-dim/2) )  * integral) ** (-beta*(exponent_m-1))
    
    
    def radius(self, t, M):
        """
        Radius of the support of the solution at time t
        """
        return t**(1/self.beta) * self.K_md**(1/2) * self.B**(1/2) * M**(self.beta*(self.exponent_m-1))
    
    def height(self, t, M):
        """
        Radius of the support of the solution at time t
        """
        return t**(-self.alpha) * self.K_md**(2/(self.exponent_m-1)) * M**(2*self.beta)

    def sigma(self, k, exponent_p):
        """
        find sigma such that
        M = (Dirac value at t=0) ~ k * (r(\sigma))**p 
        """
        return (k**(-1/exponent_p) * self.K_md ** (-1/2) * self.B **(1/2))**(1/self.beta)


class PorousMediaMap(Conductivity2ImageMap):
    """
    We I(\mu) as the solution u of the porous media PDE
    (u-u_0)/sigma - Div u^{m-1} \Grad u = 0
    with u_0 = \mu 
    """
    def __init__(self, space, 
                 scaling=1.0, 
                 exponent_m=2.0, 
                 sigma=1e-2,
                 nsteps=1,
                 solver_parameters=None) -> None:
        self.space = space
        self.scaling = scaling
        self.sigma = sigma
        self.exponent_m = exponent_m
        self.nsteps = nsteps

        self.verbose = 0
        """ Verbosity level. 0 is silent, 1 print some information"""

        self.store_images = False
        """ Store the images at each time step"""

        
        # we pass the time step as a constant function
        # to being able the PDE
        self.R = FunctionSpace(space.mesh(), 'R', 0)
        self.dt = Function(self.R)


        if nsteps==1:
            self.dt.assign(sigma)
        else:
            # we adaptive the time step to the number of steps
            self.dt.assign(sigma / nsteps)
        
        self.image_h = Function(space)
        self.tdens4transform = Function(space)
        
        
        min_image = 1e-14 # a minimim value for the image
        
        # define PDE 
        test = TestFunction(space)
        permeability = self.exponent_m * (self.image_h ) ** (self.exponent_m - 1) #+ min_image
        if space.ufl_element().degree() > 0:
            pm_Laplacian_PDE = permeability * inner(grad(self.image_h) ,grad(test)) * dx  
        else:
            if space.mesh().ufl_cell().is_simplex():
                raise NotImplementedError('Laplacian with DG0 simplices is not implemented')
            facet_image = utilities.cell2face_map(permeability, approach='arithmetic_mean') # harmonic mean does not work
            # delta_h is an expression, is light
            delta_h = utilities.delta_h(space)
            pm_Laplacian_PDE = facet_image * jump(self.image_h) * jump(test) / delta_h * dS
        self.pm_PDE = ( 
            (self.image_h - self.tdens4transform) / self.dt * test * dx 
            + pm_Laplacian_PDE )

        # relaxed Jacobian
        relaxed_permeability = self.exponent_m * (self.image_h + 1e-8) ** (self.exponent_m - 1) + 1e-8
        if space.ufl_element().degree() > 0:
            relaxed_pm = relaxed_permeability * inner(grad(self.image_h) ,grad(test)) * dx  
        else:
            if space.mesh().ufl_cell().is_simplex():
                raise NotImplementedError('Laplacian with DG0 simplices is not implemented')
            relaxed_facet = utilities.cell2face_map(relaxed_permeability, approach='arithmetic_mean') # harmonic mean does not work
            # delta_h is an expression, is light
            delta_h = utilities.delta_h(space)
            relaxed_pm = relaxed_facet * jump(self.image_h) * jump(test) / delta_h * dS
          
        relaxed_pm_PDE = ( 
                (self.image_h - self.tdens4transform) / self.dt * test * dx 
                + relaxed_pm)
        Jac = derivative(relaxed_pm_PDE, self.image_h)
        
        # set porous media problem
        self.pm_problem = NonlinearVariationalProblem(
            self.pm_PDE, self.image_h, J=Jac)

        if solver_parameters is None:
            solver_parameters={
                'snes_type': 'newtonls',
                'snes_rtol': 1e-10,
                'snes_atol': 1e-10,
                'snes_stol': 1e-10,
                'snes_linesearch_type':'bt',
                #'snes_monitor': None,
                'ksp_type': 'gmres',
                'ksp_rtol': 1e-6,
                'ksp_atol': 1e-6,
                #'ksp_monitor': None,
                'pc_type': 'hypre',
                }

        self.images =[]
        
        
        self.pm_solver = NonlinearVariationalSolver(
            self.pm_problem,
            solver_parameters=solver_parameters,
            options_prefix='porous_solver_',
            )
    
    
    def __call__(self, conductivity):#-> Function:
        """
        Apply the porous media map to the conductivity

        Args:
            conductivity: a Function with the conductivity field
        Returns:
            a Function with the image
        """
        
        # this command will inject the fun in the pde
        # (I - conductivity) / sigma  - \Delta^m I = 0
        assemble(interpolate(conductivity,self.space), tensor=self.tdens4transform)

        # estimate for initial time step
        with self.tdens4transform.dat.vec as cond_vec:
            _, min_cond = cond_vec.min()
            if min_cond < 0:
                raise ValueError('Negative conductivity')
            _, max_cond = cond_vec.max()
            dt0 = min(1e-6, 1e-6 /(max_cond))
            
        
        # find optimal expansion
        n = self.nsteps + 1
        def f(r):
            return self.sigma - dt0 * ( 1 - r ** (n -1) ) / (1 - r)
        
        def df(r):
            value = dt0 * ( (n - 1) * r ** (n -2) / (1 - r) 
                           + ( 1 - r ** (n -1) ) / (1 - r)**2 )
            return value
            
        rate = newton(f, 2, fprime=df)
        if self.verbose > 0:
            print('sigma',self.sigma,'rate=',rate,'steps=',self.nsteps,'dt0=',dt0,'f',f(rate))


        self.images = []
        total_time = 0.0
        self.steps_done = 0
        for i in range(self.nsteps):
            if i > 0:
                self.tdens4transform.assign(self.image_h)
            

         
            # assign self.dt, change the expression in the PDE
            dt = dt0*rate**(i)
            total_time += dt
            self.dt.assign(dt)
            # invoke the solver
            self.pm_solver.solve()
            
            if self.verbose > 0:
                print(f'{i=} dt={dt:.1e} t={total_time:.1e} {self.image_h.dat.data_ro.min():.1e}<=IMG<={self.image_h.dat.data_ro.max():.1e}')
            
            if self.store_images:
                img = self.image_h.copy(deepcopy=True)
                img.rename(f'img_{i}')
                self.images.append(img)
            
            self.steps_done += 1
        
        
        # we scale here so we return a function 
        # otherwise (scaling * image) is an expression
        min_img = self.image_h.dat.data_ro.min()
        if min_img < 0:
            self.image_h -= min_img
        self.image_h *= self.scaling
        if self.verbose > 0:
            print(f'{self.image_h.dat.data_ro.min():.1e}<=IMG<={self.image_h.dat.data_ro.max():.1e}')

        return self.image_h


######################################################################
# The following code is not used in the current version of the code
# I kept it for reference on how to implement an operator
# as a pyadjoint block
######################################################################


# We defined the smoothing operator addapting from the example in:
# https://www.dolfin-adjoint.org/en/latest/documentation/custom_functions.html

def smooth(func, LaplacianSmoother):
    """
    Return a smoothed version of the function `func` 
    by solving, for one time step dt, the heat equation 
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

        test = TestFunction(self.V)
        v = assemble(out*test*dx)
        return v

    def recompute_component(self, inputs, block_variable, idx, prepared):
        return backend_smooth(inputs[0], self.LaplacianSmoother)

LaplacianSmoothing = overload_function(smooth, SmootherBlock)

class MyHeatMap(Conductivity2ImageMap):
    """
    The image is the solution of one time step
    the heat equation
    (u - u_0) /sigma  - \Delta u = 0
    with u_0 = conductivity

    plus a scaling factor multiplication.
    
    This is the implementation of the smoothing operator
    """
    def __init__(self, space, scaling=1.0, sigma=1e-2) -> None:
        self.space = space
        self.scaling = scaling
        self.sigma = sigma

        self.image_h = Function(space)
        
        test = TestFunction(space)
        trial = TrialFunction(space)
        form =  inner(test, trial) * dx
        if space.ufl_element().degree() > 0:
            form += sigma * grad(test) * grad(trial) * dx
        else:
            if space.mesh().ufl_cell().is_simplex():
                raise NotImplementedError('Laplacian with DG0 simplices is not implemented')
            delta_h = utilities.delta_h(space.mesh())
            form += sigma * jump(test) * jump(trial) / delta_h * dS

        self.HeatMatrix = assemble(form).M.handle
        self.HeatSmoother = linalg.LinSolMatrix(self.HeatMatrix,
                                        self.fems.tdens_space, 
                                        solver_parameters={
                                            'ksp_monitor': None,                
                                            'ksp_type': 'cg',
                                            'ksp_rtol': 1e-10,
                                            'pc_type': 'hypre'},
                                            options_prefix='heat_smoother_')

    def __call__(self, conductivity) -> Function:        
        self.image_h = LaplacianSmoothing(conductivity,self.HeatSmoother)
        self.image_h *= self.scaling
        return self.image_h