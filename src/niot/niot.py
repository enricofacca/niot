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
import linear_algebra_utilities as linalg
from petsc4py import PETSc

SNESReasons = utilities._make_reasons(PETSc.SNES.ConvergedReason())



#from .linear_solvers import info_linalg


#from ufl import FiniteElement
#from ufl import FunctionSpace
#from firedrake import TrialFunction
#from firedrake import TestFunction
#from ufl import MixedFunctionSpace



# function operations
from firedrake import *


# integration
from ufl import dx
from firedrake import assemble


#from linear_algebra_firedrake import transpose

from petsc4py import PETSc as p4pyPETSc


def msg_bounds(vec,label):
    min = vec.min()[1]
    max = vec.max()[1]
    return "".join([f'{min:2.1e}','<=',label,'<=',f'{max:2.1e}'])

def get_step_lenght(x,increment,x_lower_bound=0.0,step_lower_bound=1e-16):
    """
    Get the step lenght to ensure that the new iterate is above as the lower bound
    """
    np_x = x.array
    np_increment = increment.array
    #print(utilities.msg_bounds(x,'x'))
    #print(utilities.msg_bounds(increment,'increment'))
    negative = (np_increment<0).any()
    #print("negative",negative)
    if negative:
        negative_indeces  = np.where(np_increment<0)[0]
        #print('negative',len(negative_indeces),'size',len(np_increment))
        #print(np.min(np_increment[negative_indeces]))
        #print(np.min(np_x[negative_indeces]))
        step = np.min((x_lower_bound-np_x[negative_indeces])/np_increment[negative_indeces])
        #print("step lenght", step)
        if step<0:
            #print("x",np_x)
            #print("increment",np_increment)
            raise ValueError("step lenght is negative")
        if (step<step_lower_bound):
            step = 0
        return step
    else:
        return 1




class SpaceDiscretization:
    """
    Class containg fem discretization variables
    """
    def __init__(self, mesh):
       #tdens_fem='DG0',pot_fem='P1'):
       """
       Initialize FEM spaces used to discretized the problem
       """  
       # For Pot unknow, create fem, function space, trial and test functions
       #meshes = MeshHierarchy(mesh, 1)
       #print("meshes",meshes)
       #print("meshes",len(meshes))
       self.pot_fem = FiniteElement('CR', mesh.ufl_cell(), 1)
       self.pot_space = FunctionSpace(mesh, self.pot_fem)
       self.pot_trial = TrialFunction(self.pot_space)
       self.pot_test = TestFunction(self.pot_space)
       
       
       # For Tdens unknow, create fem, function space, test and trial
       # space
       self.tdens_fem = FiniteElement('DG', mesh.ufl_cell(), 0)
       self.tdens_space = FunctionSpace(mesh, self.tdens_fem)
       self.tdens_trial = TrialFunction(self.tdens_space)
       self.tdens_test = TestFunction(self.tdens_space)

      
       # create mass matrix $M_i,j=\int_{\xhi_l,\xhi_m}$ with
       # $\xhi_l$ funciton of Tdens
       self.tdens_mass_matrix = assemble( inner(self.tdens_trial,self.tdens_test)*dx ,mat_type="aij").M.handle
       self.tdens_inv_mass_matrix = linalg.LinSolMatrix(self.tdens_mass_matrix, self.tdens_space,
                    solver_parameters={
                        'ksp_type':'cg',
                        'ksp_rtol': 1e-6,
                        'pc_type':'jacobi'})


       # create the mixed function space
       self.pot_tdens_space = FunctionSpace(mesh, self.pot_fem * self.tdens_fem)
       self.pot_tdens_trial = TrialFunction(self.pot_tdens_space)
       self.pot_tdens_test = TestFunction(self.pot_tdens_space)

       self.pot_is, self.tdens_is = self.pot_tdens_space.dof_dset.field_ises

    def create_solution(self):
        """
        Create a mixed function sol=(pot,tdens) 
        defined on the mixed function space.
        """
        sol = Function(self.pot_tdens_space,name=('pot','tdens'))
        sol.sub(0).vector()[:] = 0.0
        sol.sub(1).vector()[:] = 1.0

        return sol

    


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
        img_space = observed.function_space()
        self.observed  = observed
        self.observed.rename("obs","ObservedData")
        self.source = source
        self.sink = sink
        self.forcing = interpolate(self.source - self.sink, img_space)
        imbalance = assemble(self.forcing*dx)
        print(f"Mass balance {imbalance:.2e}")
        if (force_balance):
            mass_source = assemble(self.source*dx)
            mass_sink = assemble(self.sink*dx)
            self.forcing = interpolate(self.source - self.sink * mass_source / mass_sink, img_space)
            imbalance = assemble(self.forcing*dx)
            print(f"New mass balance  {imbalance:.2e}")
        self.forcing.rename("forcing","Forcing")
    
    def save_inputs(self, filename):
        """
        Save the inputs of the problem
        """
        outfile = File(filename)
        outfile.write(self.observed, self.forcing)
        
        
class Controls:
    """
    Class with Dmk Solver 
    """
    def __init__(self,
                tol=1e-2,
                time_discretization_method='mirrow_descent',
                deltat=0.5,
                max_iter=100,
                nonlinear_tol=1e-6,
                linear_tol=1e-6,
                nonlinear_max_iter=30,
                linear_max_iter=1000,
                ):
        """
        Set the controls of the Dmk algorithm
        """
        #: float: stop tolerance
        self.tol = tol

        #: character: time discretization approach
        self.time_discretization_method = time_discretization_method
        
        #: int: max number of time steps
        self.max_iter = max_iter

        #: int: max number of update restarts
        self.max_restart = 2
        #: real : time step size for contraction in case of failure 
        self.deltat_contraction = 0.5

        #: real: time step size
        self.deltat = deltat

        # variables for set and reset procedure
        self.deltat_control = 'adaptive'
        self.deltat_min = 1e-2
        self.deltat_max = 0.5
        self.deltat_expansion = 2

        #: real : lowerb bound for tdens
        self.tdens_min = 1e-8
        

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
        self.verbose = 0
        
        #: info on log file
        self.save_log = 0
        self.file_log = 'niot.log'

 
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
        self.linear_iter = 0
        self.nonlinear_iter = 0
        self.nonlinear_res = 0.0

        # set niot parameter to default
        self.set_parameters()
        
    def set_parameters(self,
                 gamma=0.8,
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

    def save_solution(self, sol, file_name):
        """
        Save solution to file
        """
        pot, tdens = sol.split()
        pot.rename("pot","Kantorovich Potential")
        tdens.rename("tdens","Optimal Transport Tdens")
        out_file = File(file_name)
        out_file.write(pot, tdens)
        
        
    def print_info(self, msg, priority):
        """
        Print messagge to stdout and to log 
        file according to priority passed
        """
        
        if (self.ctrl.verbose > priority):
            print(msg)        
            
    def discrepancy(self, problem, pot, tdens ):
        """
        Measure the discrepancy between I(tdens) and the observed data.
        """
        dis = self.confidence * ( problem.observed - self.tdens2image(tdens))**2 * dx
        return dis

    def penalty(self, forcing, gamma, pot, tdens): 
        """ 
        Definition of the branch transport penalization
        """        
        otp_pen = ( forcing * pot - 0.5 * tdens * dot(grad(pot), grad(pot))
                    + 0.5 * tdens ** gamma /  gamma ) * dx
        return otp_pen

    def regularization(self, problem, pot, tdens):
        """ penalty
        Definition of the regularization term
        """
        return 0*dx(domain=self.fems.pot_space.mesh())
    
    def energy(self, problem, pot, tdens):
        """ 
        Definition of energy minimizated by the niot solver
        """
        ene = ( self.weights[0] * self.discrepancy(problem,pot,tdens)
               + self.weights[1] * self.penalty(problem.forcing,self.gamma,pot,tdens)
               + self.weights[2] * self.regularization(problem,pot,tdens)
               )
        return ene

    def step4positive(self, tdens_vec, increment_vec, lower_bound):
        """
        Return the step that ensure tdens and slack are positive
        """        
        # get the step length that ensure positiveness of slack and tdens
        step = get_step_lenght(tdens_vec,increment_vec,lower_bound)
        
        return step

    
    def pot_PDE(self, problem, tdpot):
        F = ( tdpot.tdens * inner(grad(tdpot.pot),grad(self.pot_test))
              - problem.forcing * self.pot_test)* dx
        bc = None

                                    
    def syncronize(self, problem, sol, ctrl):
        """        
        Args:
         tdpot: Class with unkowns (tdens, pot in this case)
         problem: Class with inputs  (rhs, q_exponent)
         ctrl:  Class with controls how we solve

        Returns:
         tdpot : syncronized to fill contraint S(tdens) pot = rhs
         ierr  : control flag (=0 if everthing worked)
        """
        # pot0 is the initial guess
        # tden0 is the given data
        pot0, tdens0 = sol.split()
        # we need to write the unknown as a function in a single space
        # not a component of a mixed space 
        pot = Function(self.fems.pot_space)

        # current state as the initial solution
        # TODO: faster via to copy the values of pot0
        pot.interpolate(pot0) 
        
        # fix tdens and solve with Euler-Lagrange eqs. w.r.t to pot
        # (the resulting PDE is linear and is given by
        # PDE_pot = (problem.forcing*test-tdens0*dot(grad(pot),grad(test)))*dx
        PDE_pot = derivative(self.energy(problem, pot, tdens0), pot)
        test = self.fems.pot_test
        
        PDE_pot = tdens0 * inner(grad(pot),grad(test)) * dx - problem.forcing * test * dx

        
        u_prob = NonlinearVariationalProblem(PDE_pot, pot)

        # the context dictionary can be used to pass 
        # variable for building preconditioners
        snes_ctrl={
            "snes_rtol": ctrl.nonlinear_tol,
            "snes_atol": 1e-6,
            "snes_stol": 1e-6,
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
        if ctrl.verbose >= 1:
            snes_ctrl["snes_monitor"] = None
        if  ctrl.verbose >= 2:
                snes_ctrl["ksp_monitor"] = None
        
        context ={} 
        nullspace = VectorSpaceBasis(constant=True)
        snes_solver = NonlinearVariationalSolver(u_prob,
                                                solver_parameters=snes_ctrl,
                                                nullspace=nullspace,
                                                appctx=context)
        snes_solver.snes.setConvergenceHistory()        
        
        # solve the problem
        utilities.my_solve(snes_solver)
        ierr = 0 if snes_solver.snes.getConvergedReason()>0 else snes_solver.snes.getConvergedReason()
        self.linear_iter = snes_solver.snes.getLinearSolveIterations()
        
        # move the pot solution in sol
        sol.sub(0).assign(pot)

        return ierr
        
    
    def iterate(self, problem, sol, ctrl):
        """
        Procedure overriding update of parent class(Problem)
        
        Args:
        problem: Class with inputs  (rhs, q_exponent)
        sol  : Class with unkowns (pot,tdens, in this case)

        Returns:
         sol : solution updated from time t^k to t^{k+1} 

        """
        if (ctrl.time_discretization_method == 'mirrow_descent'):
            # the new tdens is computed as
            #
            # tdens_new  = tdens - deltat * tdens(\nabla_tdens Ene)
            #
            # and pot solves the PDE witht the new 

            pot_k, tdens_k = sol.split()


            tdens = Function(self.fems.tdens_space)
            tdens.interpolate(tdens_k)
            # test_tdens = TestFunction(self.fems.tdens_space)
            # # TODO: faster via to copy the values of pot0
            
            # direction = assemble(derivative(self.energy(problem, pot_k, tdens), tdens))

            test_tdens = TestFunction(self.fems.tdens_space)
            gamma = self.gamma
            direction = assemble(-(
                self.weights[0] * (
                    tdens_k ** (2 - gamma) * (
                        - dot(grad(pot_k), grad(pot_k))
                        + tdens_k**(gamma-1) )
                )
                + self.weights[1] * self.confidence * (tdens_k - problem.observed) 
                )*test_tdens*dx)

            d = self.fems.tdens_mass_matrix.createVecLeft()
            with direction.dat.vec as rhs, tdens_k.dat.vec as td:
                self.fems.tdens_inv_mass_matrix.solve(rhs,d)
                print(msg_bounds(d,'increment tdens'))

                if (ctrl.deltat_control == 'adaptive'):
                    _,d_max = d.max()
                    if (d_max < 0):
                        step = ctrl.deltat_max
                    else:
                        step = max(min(ctrl.deltat / d_max, ctrl.deltat_max),ctrl.deltat_min)
                    ctrl.deltat = step
                elif (ctrl.deltat_control == 'expansive'):
                    step = max(min(ctrl.deltat * ctrl.deltat_expansion, ctrl.deltat_max),ctrl.deltat_min)
                elif (ctrl.deltat_control == 'fixed'):
                    step = ctrl.deltat 
                print('step',step)
               
                # update the tdens
                td.axpy(step,d)
            
            lift_tdens = interpolate(
                tdens_k * conditional(tdens_k > ctrl.tdens_min,1,0)
                + Constant(ctrl.tdens_min) * conditional(tdens_k>ctrl.tdens_min,0,1), 
                self.fems.tdens_space) 
            sol.sub(1).assign(lift_tdens)

            pot_old, tdens_new = sol.split()
         
            # compute pot associated to new tdens
            ierr = self.syncronize(problem, sol, ctrl)
            
            self.nonlinear_iter = 1
            return ierr            
        else:
            print('value: self.ctrl.time_discretization_method not supported. Passed:',self.ctrl.time_discretization_method )
            ierr = 1
            #return tdpot, ierr, self;

    def solve(self, inputs, sol, ctrl):
        """
        Args:
        problem: Class with inputs  (rhs, q_exponent)
        sol  : Class with unkowns (pot,tdens, in this case)

        Returns:
         sol : solution updated from time t^k to t^{k+1} 

        """

        ierr = self.syncronize(inputs, sol, ctrl)
        iter = 0
        ierr_dmk = 0
        while ierr_dmk == 0:
            # get inputs

            # update with restarts
            sol_old = cp(sol)
            nrestart = 0 
            while nrestart < ctrl.max_restart:
                ierr = self.iterate(inputs, sol, ctrl)
                if ierr == 0:
                    break
                else:
                    nrestart +=1
                    
                    # reset controls after failure
                    ctrl.deltat = ctrl.deltat_contraction * ctrl.deltat
                    print(f' Failure in due to {SNESReasons[ierr]}. Restarting with deltat = {ctrl.deltat:.2e}')

                    # restore old solution
                    sol = cp(sol_old)

            if (ierr != 0):
                print('ierr',ierr)
                break

            # study state of convergence
            iter += 1
            pot, tdens = sol.split()
            pot_old, tdens_old = sol_old.split()
            #var_tdens = (
            #    assemble(dot(tdens-tdens_old,tdens-tdens_old)*dx(degree=0)) 
            #    / ( assemble(dot(tdens_old,tdens_old)*dx(degree=0)) * ctrl.deltat))
            gamma = self.gamma
            test_tdens = TestFunction(self.fems.tdens_space)
            steady_state = assemble(tdens**(2-gamma)*(
                inner(grad(pot),grad(pot))-tdens**(gamma-1))*test_tdens*dx)
            d = self.fems.tdens_mass_matrix.createVecLeft()
            with steady_state.dat.vec_ro as v:
                self.fems.tdens_inv_mass_matrix.solve(v,d)
                var_tdens = d.norm(PETSc.NormType.NORM_INFINITY)

            avg_outer = self.linear_iter / self.nonlinear_iter
            print(f"It: {iter} "+
                    f" deltat: {ctrl.deltat:.2e}"+
                    f" var:{var_tdens:.2e}"
                    f" nsym: {self.nonlinear_iter:.2f}"+
                    f" avgkrylov: {avg_outer:.2f}")
            with tdens.dat.vec_ro as v:
                print(msg_bounds(v,'tdens'))

            # check convergence
            if (var_tdens < ctrl.tol ):
                ierr_dmk = 0
                break

            # break if max iter is reached
            if (iter == ctrl.max_iter):
                ierr_dmk = 1
        
        return ierr_dmk
