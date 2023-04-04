# import Solver 
#from rcis import Solver
from copy import deepcopy as cp

import sys

import numpy as np
import scipy as sp
import scipy.sparse.linalg as splinalg
from scipy.linalg import norm 
import time as cputiming
import os
import utilities
#from .linear_solvers import info_linalg


#from ufl import FiniteElement
#from ufl import FunctionSpace
#from firedrake import TrialFunction
#from firedrake import TestFunction
#from ufl import MixedFunctionSpace

from firedrake import Function


# function operations
from firedrake import *


# integration
from ufl import dx
from firedrake import assemble


#from linear_algebra_firedrake import transpose

from petsc4py import PETSc as p4pyPETSc

class SpaceDiscretization:
    """
    Class containg fem discretization variables
    """
    def __init__(self,mesh):
       #tdens_fem='DG0',pot_fem='P1'):
       """
       Initialize FEM spaces used to discretized the problem
       """  
       # For Pot unknow, create fem, function space, trial and test functions 
       self.pot_fem = FiniteElement('Crouzeix-Raviart', mesh.ufl_cell(), 1)
       self.pot_fem_space = FunctionSpace(mesh, self.pot_fem)
       self.pot_trial = TrialFunction(self.pot_fem_space)
       self.pot_test = TestFunction(self.pot_fem_space)
       
       
       # For Tdens unknow, create fem, function space, test and trial
       # space
       self.tdens_fem = FiniteElement('DG', mesh.ufl_cell(), 0)
       self.tdens_fem_space = FunctionSpace(mesh, self.tdens_fem)
       self.tdens_trial = TrialFunction(self.tdens_fem_space)
       self.tdens_test = TestFunction(self.tdens_fem_space)

      
       # create mass matrix $M_i,j=\int_{\xhi_l,\xhi_m}$ with
       # $\xhi_l$ funciton of Tdens
       M = inner(self.tdens_trial,self.tdens_test)*dx
       self.tdens_mass_matrix = assemble( M ,mat_type="aij")

       # create the mixed function space
       self.pot_tdens_fem_space = FunctionSpace(mesh, self.tdens_fem * self.tdens_fem)
       self.pot_tdens_trial = TrialFunction(self.pot_tdens_fem_space)
       self.pot_tdens_test = TestFunction(self.pot_tdens_fem_space)

    def create_solution():
        sol = Function(self.pot_tdens_fem_space)
        sol.sub(0).vector()[:] = 0.0
        sol.sub(1).vector()[:] = 1.0



    def build_stiff(self,conductivity):
        """
        Internal procedure to assembly stifness matrix 
        A(tdens)=-Div(\Tdens \Grad)

        Args:
        conductivity: non-negative function describing conductivity

        Returns:
        stiff: PETSC matrix
        """

        u = self.pot_trial
        v = self.pot_test
        a = conductivity * dot( grad(u), grad(v) ) * dx

        stiff=assemble(a)
        
        return stiff;


class TdensPotential:
    """ This class contains problem solution tdens and pot such that
    pot solves the P-laplacian solution and 
    $\Tdens=|\Grad \Pot|^{\plapl-2}$
    """
    def __init__(self, problem, tdens0=None, pot0=None):
        """
         Constructor of TdensPotential class, containing unkonws
         tdens, pot
         
         Args:
            
        
        Raise:
        ValueError

        Example:
       
        
        """
        #: Tdens function
        self.tdens = Function(problem.tdens_fem_space)
        self.tdens.vector()[:]=1.0
        self.tdens.rename("tdens","Optimal Transport Tdens")
        
        #: Potential function
        self.pot = Function(problem.pot_fem_space)
        self.pot.vector()[:]=0.0
        self.pot.rename("Pot","Kantorovich Potential")
        
        #: int: Number of tdens variable
        self.n_tdens = self.tdens.vector().size()
        #: int: Number of pot variable 
        self.n_pot = self.pot.vector().size()

        # define implicitely the velocity
        self.velocity = self.tdens * grad(self.pot) 
        

class InpaitingProblem:
    """
    This class contains the inputs of the inpaiting problem
    """
    def __init__(self, observed, source, sink, force_balance=False):
        """
        Constructor of problem setup
        """
        self.observed  = observed
        self.source = source
        self.sink = sink
        self.forcing = self.source - self.sink
        imbalance = assemble(self.forcing*dx)
        if (force_balance):
            mass_source = assemble(self.source*dx)
            mass_sink = assemble(self.sink*dx)

        
class Controls:
    """
    Class with Dmk Solver 
    """
    def __init__(self,
                 deltat=0.5,
                 opt_max_iter=100,
                 time_discretization_method='mirrow_descent',
                 linear_max_iter=1000,
                 linear_tol=1e-6,
                 nonlinear_max_iter=30,
                 nonlinear_tol=1e-6):
        """
        Set the controls of the Dmk algorithm
        """
        #: character: time discretization approach
        self.time_discretization_method = time_discretization_method

        #: real: time step size
        self.deltat = deltat

        # variables for set and reset procedure
        self.deltat_control = 0
        self.min_deltat = 1e-2
        self.max_deltat = 1e+2
        self.expansion_deltat = 2
        
        #: int: max number of Krylov solver iterations
        self.linear_max_iter = linear_max_iter
        #: str: Krylov solver approach
        self.approach_linear_solver = 'cg'
        #: real: Krylov solver tolerance
        self.linear_tol = linear_tol
        
        #: real: nonlinear solver iteration
        self.nonlinear_tol = nonlinear_tol

        #: int: Max number of nonlinear solver iterations 
        self.nonlinear_max_iter = nonlinear_max_iter
        
       
        #: info on standard output
        self.verbose=0
        #: info on log file
        self.save_log=0
        self.file_log='admk.log'

# Create a class to store solver info 
class InfoDmkSolver():
    def __init__(self):
        self.linear_solver_iterations = 0
        # non linear solver
        self.nonlinear_solver_iterations = 0
        self.nonlinear_solver_residum = 0.0

 
class NiotSolver:
    """
    Solver for the network inpaiting problem 
    """
    def __init__(self, fems):
        """
        Initialize solver with passed controls (or default)
        and initialize structure to store info on solver application
        """
        # create pointe to FEM spaces
        self.fems = fems

        # init infos
        self.info = InfoDmkSolver()

        # set niot parameter to default
        self.set_parameters()
        
    def set_parameters(self,
                 gamma=1.5,
                 confidence = 1.0,
                 weights=[1.0,1.0,0.0],
                 tdens2image= lambda x: x ):
        """
        Set all niot parameter
        """
        self.gamma = gamma
        self.confidence  = confidence
        self.weights = weights
        self.tdens2image = tdens2image
        
        
    def print_info(self, msg, priority):
        """
        Print messagge to stdout and to log 
        file according to priority passed
        """
        
        if (self.ctrl.verbose > priority):
            print(msg)        
            
    def discrepancy(self, problem, parameters, pot, tdens ):
        """
        Measure the discrepancy between I(tdens) and the observed data.
        """
        niot = self.niot_parameters 
        dis = niot.confidence * ( problem.obs - niot.tdens2image(tdens))*dx
        return dis

    def penalty(self, problem,  pot, tdens): 
        """ 
        Definition of the branch transport penalization
        """        
        otp_pen = ( problem.forcing *pot - 0.5 * tdens * outer(grad(pot), grad(pot))
                    + 0.5 * tdens ** parameters.gamma /  parameters.gamma ) * dx
        return otp_pen

    def penalty(self, problem, pot, tdens):
        """ 
        Definition of the regularization term
        """
        return 0*dx
    
    def energy(self, problem, parameters, pot, tdens):
        """ 
        Definition of energy minimizated by the niot solver
        """
        niot = self.niot_parameters 
        ene= ( niot.weight[0] * self.discrepancy(problem,tdens,pot)
               + niot.weight[1] * self.penalty(problem,tdens,pot)
               + niot.weight[2] * self.regularization(problem,tdens,pot))
        return ene

    
    def pot_PDE(self, problem,tdpot):
        F = ( tdpot.tdens * inner(grad(tdpot.pot),grad(self.pot_test))
              - problem.forcing * self.pot_test)* dx
        bc = None

                                    
    def syncronize(self, problem, sol, ctrl, ierr):
        """        
        Args:
         tdpot: Class with unkowns (tdens, pot in this case)
         problem: Class with inputs  (rhs, q_exponent)
         ctrl:  Class with controls how we solve

        Returns:
         tdpot : syncronized to fill contraint S(tdens) pot = rhs
         info  : control flag (=0 if everthing worked)
        """
        # pot0 is the initial guess
        # tden0 is the given data
        pot0, tdens0 = sol.split()
        # we need to write the unknown as a function in a single space
        # not a component of a mixed space 
        pot = Function(self.pot_fem_space)

        # fix tdens and solve with Euler-Lagrange eqs. w.r.t to pot
        # (the resulting PDE is linear)
        PDE_pot = derivate(self.energy(problem, pot, tdens), pot)
        u_prob = NonlinearVariationalProblem(PDE_pot, pot)

        # the context dictoniary can be used to pass 
        # variable for building preconditioners
        snes_ctrl={
            # global controls
            #"ksp_monitor": None,
            #"ksp_view": None,
            "snes_rtol": ctrl.nonlinear_tol,
            "snes_atol": 1e-16,
            "snes_stol": 1e-16,
            "snes_type": 'newtonls',
            "snes_max_it": ctrl.nonlinear_max_iter,
            # krylov solver controls
            'ksp_type': 'cg',
            'ksp_atol': 1e-30,
            'ksp_rtol':  ctrl.nonlinear_tol,
            'ksp_divtol': 1e4,
            'ksp_max_it' : 100,
            # preconditioner controls
            'pc_type': 'hypre',
        }

        context ={} 
        nullspace = VectorSpaceBasis(constant=True)
        snes_solver = NonlinearVariationalSolver(u_prob,
                                                solver_parameters=snes_ctrl,
                                                nullspace=nullspace,
                                                appctx=context)
        snes_solver.snes.setConvergenceHistory()        
        self.info.linear_solver_iterations = [solver.ksp.getIterationNumber()]
        
        utilities.my_solve(snes_solver)
        ierr = snes_solver.snes.getConvergedReason()
        
        # pass the result
        sol.sub(0).assign(pot)
        
    
    def iterate(self, problem, tdpot, ctrl, ierr):
        """
        Procedure overriding update of parent class(Problem)
        
        Args:
        problem: Class with inputs  (rhs, q_exponent)
        tdpot  : Class with unkowns (tdens, pot in this case)

        Returns:
         tdpot : update tdpot from time t^k to t^{k+1} 

        """
        if (self.ctrl.time_discretization_method == 'mirrow_descent'):
            # the new tdens is computed as
            #
            # tdens_new  = tdens - deltat * tdens(\nabla_tdens Ene)
            #
            # and pot solves the PDE witht the new 
            direction = assemble(tdens * derivate(self.energy(problem, parameter, pot, tdens),tdens))

            d = self.tdens_mass_matrix.getLeftVec()
            with direction.dat.vec as rhs, tdens.dat.vec as td:
                self.tdens_inv_mass_matrix(rhs,d)
                td.axpy(- ctrl. deltat,d)
                     
            # compute pot associated to new tdens
            self.syncronize(problem,pot, tdens, ierr)
            
            #return tdpot, ierr, self;            
        else:
            print('value: self.ctrl.time_discretization_method not supported. Passed:',self.ctrl.time_discretization_method )
            ierr = 1
            #return tdpot, ierr, self;
