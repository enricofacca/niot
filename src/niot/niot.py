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
    def __init__(self, mesh, pot_space='CR', pot_deg=1, tdens_space='DG', tdens_deg=0):
       #tdens_fem='DG0',pot_fem='P1'):
       """
       Initialize FEM spaces used to discretized the problem
       """  
       # For Pot unknow, create fem, function space, trial and test functions
       #meshes = MeshHierarchy(mesh, 1)
       #print("meshes",meshes)
       #print("meshes",len(meshes))

       self.pot_fem = FiniteElement(pot_space, mesh.ufl_cell(), pot_deg)
       self.pot_space = FunctionSpace(mesh, self.pot_fem)
       self.pot_trial = TrialFunction(self.pot_space)
       self.pot_test = TestFunction(self.pot_space)

       if (pot_space=='DG') and (pot_deg == 0):
            x,y = mesh.coordinates
            x_func = interpolate(x, self.pot_space)
            y_func = interpolate(y, self.pot_space)
            self.delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)
       
       
       # For Tdens unknow, create fem, function space, test and trial
       # space
       self.tdens_fem = FiniteElement(tdens_space, mesh.ufl_cell(), tdens_deg)
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

    def save_solution(self, sol, file_name):
        """
        Save solution to file
        """
        pot, tdens = sol.split()
        pot.rename("pot","Kantorovich Potential")
        tdens.rename("tdens","Optimal Transport Tdens")
        out_file = File(file_name)
        out_file.write(pot, tdens)
        

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
    Solver for the network inpaiting problem-
    Args:
    spaces: strings denoting the discretization scheme used
    numpy_corruped: numpy 2d/3d array describing network to be reconstructed
    numpy_source: numpy 2d/3d array describing inlet flow
    numpy_sink: numpy 2d/3d array describing outlet flow
    force_balance: boolean to force balancing sink to ensure problem to be well posed
    """
    def __init__(self, spaces, numpy_corrupted, numpy_source, numpy_sink, force_balance=False):
        """
        Initialize solver (spatial discretization) from numpy_image (2d or 3d data)

        """
        # create mesh from image
        self.spaces = spaces
        print('spaces', spaces)
        if spaces == 'CR1DG0':
            self.mesh = self.build_mesh_from_numpy(numpy_corrupted, mesh_type='simplicial')
            self.mesh_type = 'simplicial'
            # create FEM spaces
            self.fems = SpaceDiscretization(self.mesh,'CR',1, 'DG',0)
        elif spaces == 'DG0DG0':
            self.mesh = self.build_mesh_from_numpy(numpy_corrupted, mesh_type='cartesian')
            self.mesh_type = 'cartesian'
            # create FEM spaces
            self.fems = SpaceDiscretization(self.mesh,'DG',0,'DG',0)
        else:
            raise ValueError("Wrong spaces only (pot,tdens) in (CR1,DG0) or (DG0,DG0) implemented")

        self.DG0 = FunctionSpace(self.mesh, "DG", 0)

        # set function 
        self.observed = self.numpy2function(numpy_corrupted)

        # set source and sink 
        self.set_inout_flow(numpy_source, numpy_sink, force_balance=force_balance)    
    
        # set niot parameter to default
        self.set_parameters(
            gamma=0.6,
            weights=[1.0,1.0,0.0],
            confidence=Function(self.DG0).assign(1.0),
            tdens2image=lambda x: x)

        # init infos
        self.linear_iter = 0
        self.nonlinear_iter = 0
        self.nonlinear_res = 0.0


    def build_mesh_from_numpy(self, np_image, mesh_type='simplicial'): 
        """
        Create a mesh (first axis size=1) from a numpy array
        """
        if (mesh_type == 'simplicial'):
            quadrilateral = False
        elif (mesh_type == 'cartesian'):
            quadrilateral = True
        print(f"Mesh type {mesh_type} quadrilateral {quadrilateral}")
        
        if (np_image.ndim == 2):
            height, width  = np_image.shape
            mesh = RectangleMesh(width,height,1,height/width, 
                quadrilateral = quadrilateral,
                reorder=False)
                
        elif (np_image.ndim == 3):
            raise NotImplementedError("3D not implemented")
        else:
            raise ValueError("Wrong dimension of image")
        return mesh

        
    def set_parameters(self,
                 gamma=None,
                 weights=None,
                 confidence=None,
                 tdens2image=None):
        """
        Set all niot parameter
        """
        if gamma is not None:
            if gamma<0:
                raise ValueError(f"Gamma must be greater than zero")
            self.gamma = gamma
        
        if weights is not None:
            if len(weights)!=3:
                raise ValueError(f"3 weights are required, Transport Energy, Discrepancy, Penalization")
            self.weights = weights

        if confidence is not None:
            self.confidence  = self.convert_data(confidence)
            self.confidence.rename('confidence','confidence')
        
        if tdens2image is not None:
            self.tdens2image = tdens2image

    def numpy2function(self, value, name=None):
        """
        Convert np array (2d o 3d) into a function compatible with the mesh solver.
        Args:
        
        value: numpy array (2d or 3d) with images values

        returns: piecewise constant firedake function 
        """ 
        
        # we flip vertically because images are read from left, right, top to bottom
        value = np.flip(value,0)

        if (self.mesh_type == 'simplicial'):
            # Each pixel is splitted in two triangles.
            if (self.mesh.geometric_dimension() == 2):
                # double the value to copy the pixel value to the triangles
                double_value = np.zeros([2,value.shape[0],value.shape[1]])
                double_value[0,:,:] = value[:,:]
                double_value[1,:,:] = value[:,:]
                triangles_image = double_value.swapaxes(0,2).flatten()
                DG0 = FunctionSpace(self.mesh,'DG',0)
                
                img_function = Function(DG0)
                with img_function.dat.vec as d:
                    d.array = triangles_image
            else:
                raise ValueError("3d data not implemented yet")
            if (name is not None):
                img_function.rename(name,name)
            return img_function

        if (self.mesh_type == 'cartesian'):
            DG0 = FunctionSpace(self.mesh,'DG',0)
            img_function = Function(DG0)
            with img_function.dat.vec as d:
                d.array = value.flatten('F')
            if (name is not None):
                img_function.rename(name,name)
            return img_function

    def convert_data(self, data):
        print(type(data))
        if isinstance(data, np.ndarray):
            return self.numpy2function(data)
        elif isinstance(data, functionspaceimpl.FunctionSpace) or isinstance(data, function.Function):
            # function must be defined on the same mesh
            if data.function_space().mesh() == self.mesh:
                return data
            else:
                raise ValueError("Data not defined on the same mesh of the solver")
        elif isinstance(data, constant.Constant):
            return data 
        else:
            raise ValueError("Type "+str(type(data))+" not supported")

    def set_inout_flow(self, source, sink, force_balance=False, tolerance=1e-10):
        """
        Set source and sink member of niot_solver class

        Args:
        source: (numpy 2d-3d array or function) inlet ratio
        sink:  (numpy 2d-3d array or function) outlet ratio
        force balance: (bolean) balance sink term 
        """

        # Ensure to store source and sink as functions
        self.source = self.convert_data(source)
        self.source.rename('source','source')
        self.sink = self.convert_data(sink)
        self.sink.rename('sink','sink')

        mass_source = assemble(self.source*dx)
        mass_sink = assemble(self.sink*dx)

        if abs(mass_source-mass_sink) > tolerance :
            if force_balance:
                with self.sink.dat.vec as f:
                    f.scale(mass_source / mass_sink)
            else:
                raise ValueError("Source and sink terms are not balanced")

    def create_solution(self):
        return self.fems.create_solution()

    def save_solution(self, sol, file_name):
        self.fems.save_solution(sol,file_name)

    def save_inputs(self, file_name):
        out_file = File(file_name)
        out_file.write(self.source, self.sink, self.observed, self.confidence)
        
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

                                    
    def syncronize(self, sol, ctrl):
        """        
        Args:
         sol: Class with unkowns (tdens, pot in this case)
         ctrl:  Class with controls how we solve

        Returns:
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
        # PDE_pot = derivative(self.energy(problem, pot, tdens0), pot)
        test = self.fems.pot_test
        
        if (self.spaces == 'CR1DG0'):
            PDE_pot = tdens0 * inner(grad(pot),grad(test)) * dx - (self.source-self.sink) * test * dx
        elif (self.spaces == 'DG0DG0'):
            tdens_facet = avg(tdens0)
            PDE_pot = tdens_facet * jump(pot) * jump(test) / self.fems.delta_h * dS - (self.source-self.sink) * test * dx
        
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
        
    
    def iterate(self, sol, ctrl):
        """
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
            if (self.spaces == 'CR1DG0'):
                direction = assemble(-(
                    self.weights[0] * (
                        tdens_k ** (2 - gamma) * (
                            - dot(grad(pot_k), grad(pot_k))
                            + tdens_k**(gamma-1) )
                    ) * test_tdens * dx
                    + self.weights[1] * self.confidence * (tdens_k - self.observed) * test_tdens * dx
                    #+ self.weights[2] * inner(grad(tdens_k), grad(test_tdens)) * dx 
                ))
            elif (self.spaces == 'DG0DG0'):
                tdens_facet = avg(tdens_k**(2-gamma))
                test_facet = avg(test_tdens)
                direction = assemble(-(
                    - self.weights[0] * test_facet* tdens_facet * jump(pot_k) **2 / self.fems.delta_h * dS
                    + self.weights[0] * tdens_k * test_tdens * dx
                    + self.weights[1] * self.confidence * (tdens_k - self.observed) * test_tdens*dx
                    #+ self.weights[2] * inner(grad(tdens), grad(test_tdens)) * dx
                ))

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
            ierr = self.syncronize( sol, ctrl)
            
            self.nonlinear_iter = 1
            return ierr            
        else:
            print('value: self.ctrl.time_discretization_method not supported. Passed:',self.ctrl.time_discretization_method )
            ierr = 1
            #return tdpot, ierr, self;

    def solve(self, sol, ctrl):
        """
        Args:
        problem: Class with inputs  (rhs, q_exponent)
        sol  : Class with unkowns (pot,tdens, in this case)

        Returns:
         sol : solution updated from time t^k to t^{k+1} 

        """

        ierr = self.syncronize( sol, ctrl)
        iter = 0
        ierr_dmk = 0
        while ierr_dmk == 0:
            # get inputs

            # update with restarts
            sol_old = cp(sol)
            nrestart = 0 
            while nrestart < ctrl.max_restart:
                ierr = self.iterate(sol, ctrl)
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
