# import Solver 
#from rcis import Solver
from copy import deepcopy as cp
import gc
import sys

import numpy as np
import scipy as sp
import scipy.sparse.linalg as splinalg
from scipy.linalg import norm 
import time as cputiming
import os


import petsc4py
import time 
#from memory_profiler import profile


#from . import utilities
#from . import optimal_transport as ot

#from . import conductivity2image as conductivity2image
#from . import linear_algebra_utilities as linalg
from . conductivity2image import IdentityMap, HeatMap, PorousMediaMap
from . import utilities
from . import optimal_transport as ot

#from . import conductivity2image as conductivity2image
from . import linear_algebra_utilities as linalg


# function operations
from firedrake import *
from firedrake.__future__ import interpolate

#from firedrake_adjoint import ReducedFunctional, Control
#from pyadjoint.reduced_functional import  ReducedFunctional
#from pyadjoint.control import Control

#import firedrake.adjoint as fireadj # does not work ...
#from firedrake.adjoint import * # does give error when interpoalte is called
#import firedrake.adjoint.ReducedFunctional as ReducedFunctional # does not work ...
import firedrake.adjoint as fire_adj
fire_adj.continue_annotation()

from firedrake.tsfc_interface import TSFCKernel
from pyop2.global_kernel import GlobalKernel
from firedrake.petsc import PETSc
SNESReasons = utilities._make_reasons(PETSc.SNES.ConvergedReason())


# include all citations
utilities.include_citations(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), 
                     '../../citations/citations.bib')
    ))



def msg_bounds(vec,label):
    """
    Generate a message with the bounds of the vector
    """
    min = vec.min()[1]
    max = vec.max()[1]
    return ''.join([f'{min:2.1e}','<=',label,'<=',f'{max:2.1e}'])

def get_step_lenght(x,increment,x_lower_bound=0.0,step_lower_bound=1e-16):
    '''
    Get the step lenght to ensure that the new iterate is above as the lower bound
    '''
    np_x = x.array
    np_increment = increment.array
    negative = (np_increment<0).any()
    if negative:
        negative_indeces  = np.where(np_increment<0)[0]
        step = np.min((x_lower_bound-np_x[negative_indeces])/np_increment[negative_indeces])
        if step<0:
            raise ValueError('step lenght is negative')
        if (step<step_lower_bound):
            step = 0
        return step
    else:
        return 1





class SpaceDiscretization:
    '''
    Class containg fem discretization variables
    '''
    
    # include relevant citations
    Citations().register('FCP2021')
    def __init__(self, mesh, 
                 pot_space='CR', pot_deg=1, 
                 tdens_space='DG', tdens_deg=0, 
                 cell2face='harmonic_mean'):
        #tdens_fem='DG0',pot_fem='P1'):
        '''
        Initialize FEM spaces used to discretized the problem
        '''  
        self.pot_fem = FiniteElement(pot_space, mesh.ufl_cell(), pot_deg)
        self.pot_space = FunctionSpace(mesh, self.pot_fem)
        self.pot_trial = TrialFunction(self.pot_space)
        self.pot_test = TestFunction(self.pot_space)

        if (pot_space=='DG') and (pot_deg == 0):
            if mesh.geometric_dimension() == 2:
                x,y = mesh.coordinates
                x_func = assemble(interpolate(x, self.pot_space))
                y_func = assemble(interpolate(y, self.pot_space))
                self.delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)
            elif mesh.geometric_dimension() == 3:
                x,y,z = mesh.coordinates
                x_func = assemble(interpolate(x, self.pot_space))
                y_func = assemble(interpolate(y, self.pot_space))
                z_func = assemble(interpolate(z, self.pot_space))
                self.delta_h = sqrt(jump(x_func)**2 
                                    + jump(y_func)**2 
                                    + jump(z_func)**2)
            
            self.cell2face = cell2face


        # For Tdens unknow, create fem, function space, test and trial
        # space
        self.tdens_fem = FiniteElement(tdens_space, mesh.ufl_cell(), tdens_deg)
        self.tdens_space = FunctionSpace(mesh, self.tdens_fem)
        self.tdens_trial = TrialFunction(self.tdens_space)
        self.tdens_test = TestFunction(self.tdens_space)

        # quantities for DG0 laplacian
        alpha = Constant(4.0)
        h = CellSize(mesh)
        h_avg = (h('+') + h('-'))/2.0
        self.DG0_scaling = alpha/h_avg
        self.normal = FacetNormal(mesh)

        
        # create mass matrix $M_i,j=\int_{\xhi_l,\xhi_m}$ with
        # $\xhi_l$ funciton of Tdens
        mass_form = inner(self.tdens_trial,self.tdens_test)*dx
        self.tdens_mass_matrix = assemble( mass_form ,mat_type='aij').M.handle
        self.inv_tdens_mass_matrix = linalg.LinSolMatrix(self.tdens_mass_matrix, self.tdens_space,
                        solver_parameters={
                            'ksp_type':'cg',
                            'ksp_rtol': 1e-6,
                            'pc_type':'jacobi'})


        # create the mixed function space
        self.pot_tdens_space = FunctionSpace(mesh, self.pot_fem * self.tdens_fem)
        self.pot_tdens_trial = TrialFunction(self.pot_tdens_space)
        self.pot_tdens_test = TestFunction(self.pot_tdens_space)

        self.pot_is, self.tdens_is = self.pot_tdens_space.dof_dset.field_ises

    
    def cell2face_map(self, fun, approach=None):
        if approach is None:
            approach = self.cell2face
        if approach == 'arithmetic_mean':
            return (fun('+') + fun('-')) / 2
        elif approach == 'harmonic_mean':
            #return 2 * fun('+') * fun('-') / ( fun('+') + fun('-') )
            return conditional( gt(avg(fun), 0.0), fun('+') * fun('-') / avg(fun), 0.0)          
        else:
            raise ValueError('Wrong approach passed. Only arithmetic_mean or harmonic_mean are implemented')

    def Laplacian_form(self, space, weight=None, cell2face=None):
        """
        Return the Laplacian form for the given space
        with zero-Neumann boundary conditions
        """

        if cell2face is None:
            cell2face = self.cell2face

        if weight is None:
            weight = Constant(1.0)


        test = TestFunction(space)
        trial = TrialFunction(space)
        if space.ufl_element().degree() == 0:
            if space.mesh().ufl_cell().is_simplex():
                # if the mesh is simplicial, we use the DG0 laplacian taken from
                # https://www.firedrakeproject.org/demos/saddle_point_systems.py.html
                # Without scaling the scheme is not consistent.
                form = self.DG0_scaling * inner(jump(test, self.normal), jump(trial, self.normal)) * dS
            else:
                form = jump(test) * jump(trial) / self.delta_h * dS
        elif self.fems.tdens_space.ufl_element().degree() == 1:
            form = inner(grad(test), grad(trial)) * dx
        else:
            raise NotImplementedError('piecewise constant, or linear tdens is implemented')
        return form
    
    
    def Laplacian_Lagrangian(self, u, weight=None, cell2face=None):
        """
        Return the Lagrangian of a weighted-Laplacian equation for u.
        The weight is a function of the mesh.
        If the weight is not passed, it is assumed to be 1.0.
        """
        if cell2face is None:
            cell2face = self.cell2face

        if weight is None:
            weight = Constant(1.0)

        space = u.function_space()

        
        if space.ufl_element().degree() == 0:
            # the weight need to be "projected to the facets"
            facet_weight = self.cell2face_map(weight, cell2face)
            if space.mesh().ufl_cell().is_simplex():
                # if the mesh is simplicial, we use the DG0 laplacian taken from
                # https://www.firedrakeproject.org/demos/saddle_point_systems.py.html
                # Without scaling the scheme is not consistent.
                L = 0.5 * self.DG0_scaling * facet_weight * inner(jump(u, self.normal), jump(u, self.normal)) * dS
            else:
                L = 0.5 * facet_weight * jump(u)**2 / self.delta_h * dS
        elif self.fems.tdens_space.ufl_element().degree() == 1:
            L = 0.5 * weight * inner(grad(u), grad(u)) * dx
        else:
            raise NotImplementedError('piecewise constant, or linear tdens is implemented')
        return L



            
def set_step(increment,
             state, 
             deltat,
             type='adaptive', 
             lower_bound=1e-2, 
             upper_bound=0.5, 
             expansion=2,
             contraction=0.5):
    """
    Set the step lenght according to the control strategy
    and the increment
    """
    if (type == 'adaptive'):
        abs_inc = abs(increment)
        _,d_max = abs_inc.max()
        if (d_max < 0):
            step = upper_bound
        else:
            step = max(min(1.0 / d_max, upper_bound), lower_bound)
    elif (type == 'adaptive2'):
        order_down = -0.25
        order_up = 0.5
        r = increment / state
        r_np = r.array
        if np.min(r_np) < 0:
            negative = np.where(r_np < 0)
            hdown = (10**order_down-1) / r_np[negative]
            down = np.min(hdown)
        else:
            down = upper_bound
        if np.max(r_np) > 0:
            positive = np.where(r_np>0)
            hup = (10**order_up - 1) / r_np[positive]
            up = np.min(hup)
        else:
            up = upper_bound
        step = min(up,down)
        step = max(step,lower_bound)
        step = min(step,upper_bound)

    elif (type == 'expansive'):
        step = deltat * expansion
        step = min(step, upper_bound)
        step = max(step, lower_bound)
    elif (type == 'fixed'):
        step = deltat
    else:
        raise ValueError(f'{type=} not supported')
    return step

def delta_h(space):
    mesh = space.mesh()
    if mesh.geometric_dimension() == 2:
        x,y = mesh.coordinates
        x_func = assemble(interpolate(x, space))
        y_func = assemble(interpolate(y, space))
        delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)
    elif mesh.geometric_dimension() == 3:
        x,y,z = mesh.coordinates
        x_func = assemble(interpolate(x, pot_space))
        y_func = assemble(interpolate(y, pot_space))
        z_func = assemble(interpolate(z, pot_space))
        delta_h = sqrt(jump(x_func)**2 
                            + jump(y_func)**2 
                            + jump(z_func)**2)
    return delta_h

def cell2face_map(fun, approach):
    if approach == 'arithmetic_mean':
        return (fun('+') + fun('-')) / 2
    elif approach == 'harmonic_mean':
        avg_fun = avg(fun)
        return conditional( gt(avg_fun, 0.0), fun('+') * fun('-') / avg_fun, 0.0)          
    else:
        raise ValueError('Wrong approach passed. Only arithmetic_mean or harmonic_mean are implemented')



class NiotSolver:
    '''
    Solver for the network inpaiting problem-
    Args:
    spaces: strings denoting the discretization scheme used
    numpy_corruped: numpy 2d/3d array describing network to be reconstructed
    numpy_source: numpy 2d/3d array describing inlet flow
    numpy_sink: numpy 2d/3d array describing outlet flow
    force_balance: boolean to force balancing sink to ensure problem to be well posed
    '''

    # nested dictoctionary describing all 
    # controls of niot algorithm.
    # Use the ctrl_get/ctrl_set method to read/write the value
    global_ctrl = {
        # main controls
        'optimization_tol': 1e-2,
        'constraint_tol': 1e-6,
        'max_iter': 100,
        'max_restart': 2,
        'min_tdens' : 1e-8,
        # info 
        'verbose' : 0,
        'log_verbose': 2,
        'log_file': 'niot.log',
        #'inpainting' : {
        'discrepancy_weight': 1.0,
        'regularization_weight': 0.0,
        'penalization_weight': 1.0,
        'tdens2image' : {
            'type' : 'identity', # idendity, heat, pm
            'scaling': 1.0,
            'pm': {
                'sigma' : 1e-2,
                'exponent_m': 2,
                },
            'heat': {
                'sigma' : 1e-2
                },
        },
        'pot_solver':{
            'ksp': {
                'type' : 'minres',
                'max_iter' : 1000,
                },
            'pc': {
                'type' : 'hypre',
                },
        },
        'optimization_type' : 'dmk',
        'dmk': {
            'type' : 'tdens_mirror_descent',
            'tdens_mirror_descent_explicit' : {
                'gradient_scaling' : 'dmk',
                'deltat' : {
                    'type' : 'adaptive2',
                    'lower_bound' : 1e-2,
                    'upper_bound' : 0.5,
                    'expansion' : 2,
                    'contraction': 0.5,
                },                
            },
            'tdens_mirror_descent_semi_implicit' : {
                'gradient_scaling' : 'dmk',
                'deltat' : {
                    'type' : 'adaptive2',
                    'lower_bound' : 1e-2,
                    'upper_bound' : 0.5,
                    'expansion' : 2,
                    'contraction': 0.5,
                },                
            },
            'gfvar_gradient_descent_semi_implicit' : {
                'deltat' : {
                    'type' : 'adaptive2',
                    'lower_bound' : 1e-2,
                    'upper_bound' : 0.5,
                    'expansion' : 2,
                    'contraction': 0.5,
                },                
            },
            'gfvar_gradient_descent_explicit' : {
                'deltat' : {
                    'type' : 'adaptive2',
                    'lower_bound' : 1e-2,
                    'upper_bound' : 0.5,
                    'expansion' : 2,
                    'contraction': 0.5,
                },                
            },
            'tdens_logarithmic_barrier' : {
                'eps' : 1e-6,
                'deltat' : {
                    'type' : 'adaptive2',
                    'lower_bound' : 1e-6,
                    'upper_bound' : 0.5,
                    'expansion' : 2,
                    'contraction': 0.5,
                },                
            },
        }
    }

    def ctrl_set(self, key, value):
        '''
        Set the value of the key in the global__ctrl dictionary
        '''
        return utilities.nested_set(self.global_ctrl,key,value)
    
    def ctrl_get(self, key):
        '''
        Get the value of the key in the global_ctrl dictionary
        '''
        return utilities.nested_get(self.global_ctrl,key)   

    # register citations using Citations class in firedrake
    Citations().register('FCP2021')
    def __init__(self, btp, observed, 
                 confidence=1.0, 
                 spaces='DG0DG0',
                 cell2face='harmonic_mean',
                 setup=False):
        '''
        Initialize solver (spatial discretization)
        '''
        ###########################
        # SETUP FEM DISCRETIZATION
        ###########################
        
        self.mesh = btp.mesh
        self.comm  = self.mesh.comm
        self.spaces = spaces
        self.cell2face = cell2face
        
        if self.spaces == 'CR1DG0':
            self.fems = SpaceDiscretization(self.mesh,'CR', 1, 'DG', 0, cell2face)
        elif self.spaces == 'DG0DG0':
            if self.mesh.ufl_cell().is_simplex():
                raise ValueError('DG0DG0 only implemented for cartesian grids')
            self.fems = SpaceDiscretization(self.mesh,'DG',0,'DG',0, cell2face)
        else:
            raise ValueError('Wrong spaces only (pot,tdens) in (CR1,DG0) or (DG0,DG0) implemented')
        self.ConstansSpace = FunctionSpace(self.mesh, 'R', 0)



        # initialize the solution
        self.sol = self.create_solution()
        self.sol_old = self.sol.copy(deepcopy=True)

        # The class BranchedTransportProblem
        # that contains the physical information of the problem
        # and the branched transport exponent gamma 
        self.btp = btp
        
        # set img to be inpainted
        self.img_observed = observed
    
        # confidence 
        self.confidence = Function(self.fems.tdens_space)
        assemble(interpolate(confidence,self.fems.tdens_space), tensor=self.confidence)
        self.confidence.rename('confidence','confidence')

        # init infos
        self.iteration = 0
        self.restart = 0
        self.outer_iterations = 0
        self.nonlinear_iterations = 0
        self.nonlinear_res = 0.0
        self.gradients_computed = False

        # set to initial state
        self.deltat = 0.0


        self.gradient_D_P = Function(self.fems.tdens_space)
        self.rhs_ode = Function(self.fems.tdens_space)
        self.residuum = Function(self.fems.tdens_space)

        if setup:
            self.setup()


    def setup(self):
        """
        Initialize all controls-dependent variables:
        - pot_solver
        - tdens2image 
        - incremental solver
        It is called at the beginning of the solver
        """

        # map from tdens to image
        self.tdens_h = Function(self.fems.tdens_space) # used by tdens2image map
        self.tdens_h.rename('tdens_h')

        self.gfvar = Function(self.fems.tdens_space)
        self.gfvar.rename('gfvar')



        self.setup_tdensimage()


        # solver of poisson equation
        petsc_controls ={
            # krylov solver controls
            'ksp_type': 'cg',
            'ksp_atol': 1e-16,
            'ksp_rtol': self.ctrl_get('constraint_tol'),
            'ksp_divtol': 1e10,
            'ksp_max_it' : 1000,
            'ksp_initial_guess_nonzero': True, 
            'ksp_norm_type': 'unpreconditioned',
            #'ksp_monitor_true_residual' : None, 
            'pc_type': 'hypre'
        }
        if self.ctrl_get('verbose') >= 3:
            petsc_controls['ksp_monitor_true_residual'] = None
        
        
        self.setup_pot_solver(petsc_controls)

        log_verbose = self.ctrl_get('log_verbose')
        if log_verbose > 0:
            self.log_viewer = PETSc.Viewer().createASCII(self.ctrl_get('log_file'), 'w', comm=self.comm)
    
        # we need to initialize the increment solver
        self.shift_semi_implicit = Function(self.ConstansSpace)
        self.shift_semi_implicit.assign(0.1)
        test = TestFunction(self.fems.tdens_space)  
        trial = TrialFunction(self.fems.tdens_space)
        self.increment_form =  inner(test, trial)*dx
        self.increment_form += self.shift_semi_implicit * self.fems.Laplacian_form(self.fems.tdens_space)
        self.increment_h = Function(self.fems.tdens_space)
        self.increment_h.rename('increment_h')

        # self.rhs_semi_implicit = Function(self.fems.tdens_space)
        # self.increment_problem = LinearVariationalProblem(
        #     self.increment_form,
        #     self.rhs_semi_implicit,
        #     self.increment_h)
        
        # self.increment_solver = LinearVariationalSolver(
        #     self.increment_problem,
        #     solver_parameters={'ksp_type': 'cg',
        #                         'ksp_rtol': 1e-10,
        #                         'ksp_atol': 1e-13,
        #                         'pc_type': 'hypre'},
        #                         options_prefix='increment_solver_')


        self.print_info(f'Number of cells: {self.mesh.num_cells()}',priority=2,where=['stdout','log'])

    def setup_tdensimage(self):
        """
        Method initializing the map from image to tdens
        """
        self.image_h = Function(self.fems.tdens_space)
        self.image_h.rename('image_h') # used by tdens2image map

        self.tdens4transform = Function(self.fems.tdens_space)
        self.tdens4transform.rename('tdens4transform') # used by tdens2image map
        
        tdens2image = self.ctrl_get(['tdens2image', 'type'])
        scaling = self.ctrl_get(['tdens2image', 'scaling'])
        


        if tdens2image == 'identity':
            self.tdens2image_map = IdentityMap(self.fems.tdens_space, scaling=scaling)
            self.tdens2image = lambda x: self.tdens2image_map(x)

        elif tdens2image == 'heat':
            sigma = self.ctrl_get(['tdens2image', 'heat','sigma'])
            self.tdens2image_map = HeatMap(self.fems.tdens_space,
                                            scaling=scaling, 
                                            sigma=sigma)
            self.tdens2image = lambda x: self.tdens2image_map(x)

        elif tdens2image == 'pm':
            sigma = self.ctrl_get(['tdens2image', 'pm','sigma'])
            exponent_m = self.ctrl_get(['tdens2image', 'pm','exponent_m'])
            self.tdens2image_map = PorousMediaMap(
                self.fems.tdens_space,
                scaling=scaling, 
                sigma=sigma,
                exponent_m=exponent_m,
                nsteps=5)
            self.tdens2image = lambda x: self.tdens2image_map(x)

        else:
            raise ValueError(f"Map tdens2iamge not supported {tdens2image=}\n"
                             +"Only identity, heat, pm,  are implemented")

        
    def setup_pot_solver(self, petsc_controls):
        # chaced functions
        self.pot_h = Function(self.fems.pot_space) # used by pot_solver
        self.pot_h.rename('pot_h')
        self.rhs = (self.btp.source - self.btp.sink) * self.fems.pot_test * dx
        self.pot_PDE = derivative(self.joule(self.pot_h,self.tdens_h),self.pot_h)
        
        # the minus sign is to get -\div(\tdens \grad \pot)-f = 0
        self.weighted_Laplacian = derivative(-self.pot_PDE,self.pot_h)

        min_tdens = self.ctrl_get('min_tdens')
        self.pot_PDE_relaxed = derivative(self.joule(self.pot_h,self.tdens_h+10*min_tdens),self.pot_h)
        self.weighted_Laplacian_relaxed = self.weighted_Laplacian + 10*min_tdens * self.fems.Laplacian_form(self.fems.pot_space)

        
        #test = TestFunction(self.fems.pot_space)
        
        #pot_PDE = self.forcing * test * dx + tdens * inner(grad(pot_unknown), grad(test)) * dx 
        # Define the Nonlinear variational problem (it is linear in this case)
        #self.u_prob = NonlinearVariationalProblem(self.pot_PDE, self.pot_h)#, bcs=pot_bcs)
        self.u_prob = LinearVariationalProblem(self.weighted_Laplacian, self.rhs, self.pot_h, aP=self.weighted_Laplacian_relaxed)#, bcs=pot_bcs)

        context ={} # left to pass information to the solver
        if self.btp.Dirichlet is None:
            nullspace = VectorSpaceBasis(constant=True,comm=self.comm)
            
        
        #self.pot_solver = NonlPDEinearVariationalSolver(self.u_prob,
        self.pot_solver = LinearVariationalSolver(self.u_prob,
                                                solver_parameters=petsc_controls,
                                                nullspace=nullspace,
                                                appctx=context,
                                                options_prefix='pot_solver_')
        self.pot_solver.snes.ksp.setConvergenceHistory()

    def setup_increment_solver(self, shift=0.0):
        test = TestFunction(self.fems.tdens_space)  
        trial = TrialFunction(self.fems.tdens_space)
        form =  inner(test, trial)*dx
        form += shift * self.fems.Laplacian_form(self.fems.tdens_space)

        self.increment_h = Function(self.fems.tdens_space)
        self.increment_h.rename('increment_h')
        #PETSc.Sys.Print(f"niot_solver inside solve increment",comm=self.comm)
        self.IncrementMatrix = assemble(form).M.handle
        self.IncrementSolver = linalg.LinSolMatrix(self.IncrementMatrix,
                                                    self.fems.tdens_space, 
                                                    solver_parameters={
                                        #'ksp_monitor': None,                
                                        'ksp_type': 'cg',
                                        'ksp_rtol': 1e-10,
                                        'ksp_atol': 1e-13,
                                        'pc_type': 'hypre'},
                                        options_prefix='increment_solver_')
        #PETSc.Sys.Print(f"niot_solve setup done",comm=self.comm)


               
    def print_info(self, msg, priority=0, where=['stdout'], color='black'):
        '''
        Print messagge to stdout and to log 
        file according to priority passed
        '''
        for mode in where:
            if mode=='stdout':
                verbose = self.ctrl_get('verbose')
                if verbose >= priority: 
                    if color != 'black':
                        stdout_msg = utilities.color(color, msg)
                    else:
                        stdout_msg = msg
                    PETSc.Sys.Print('   '*(priority-1) + stdout_msg, comm=self.comm)

            if mode == 'log':
                log_verbose = self.ctrl_get('log_verbose')
                if log_verbose >0 and log_verbose >= priority:
                    self.log_viewer.pushASCIISynchronized()
                    self.log_viewer.printfASCIISynchronized('   '*(priority-1)+msg+'\n')

    

    def compute_residuum(self, sol):
        """
        Return the residual of the minimization problem
        w.r.t to tdens
        """
        pot, tdens = sol.subfunctions

        self.tdens_h.assign(tdens)
        self.pot_h.assign(pot)
        
        method = self.ctrl_get(['dmk','type'])
        if "tdens" in method:
            dw = self.ctrl_get('discrepancy_weight')
            pw = self.ctrl_get('penalization_weight')
            rw = self.ctrl_get('regularization_weight')
            if dw > 0:
                self.lagrangian_fun = assemble(dw*self.discrepancy(pot,self.tdens_h))   
                self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.lagrangian_fun, fire_adj.Control(self.tdens_h))
                self.gradient_discrepancy = self.lagrangian_fun_reduced.derivative()
                with self.gradient_discrepancy.dat.vec_ro as gD:
                    msg = utilities.msg_bounds(gD,'grad discrepancy   ')
                    self.print_info(
                        msg=msg,
                        priority=1, 
                        where=['stdout','log']
                        )
            else:
                self.gradient_discrepancy = 0.0

            if pw > 0:
                self.lagrangian_fun = assemble(pw*self.penalization(pot,self.tdens_h))
                self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.lagrangian_fun, fire_adj.Control(self.tdens_h))
                self.gradient_penalization = self.lagrangian_fun_reduced.derivative()
                with self.gradient_penalization.dat.vec_ro as gP:
                    msg = utilities.msg_bounds(gP,'grad penalty       ')
                    self.print_info(
                        msg=msg,
                        priority=1, 
                        where=['stdout','log']
                        )
            else:
                self.gradient_penalization = 0.0

            if rw > 0:
                self.lagrangian_fun = assemble(rw*self.regularization(pot,self.tdens_h))
                self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.lagrangian_fun, fire_adj.Control(self.tdens_h))
                self.gradient_regularization = self.lagrangian_fun_reduced.derivative()
            else:
                self.gradient_regularization = 0.0

            self.gradient_D_P.assign(self.gradient_discrepancy + self.gradient_penalization)
            self.rhs_ode.assign(self.gradient_discrepancy + self.gradient_penalization + self.gradient_regularization)
            
            #d = self.fems.tdens_mass_matrix.createVecLeft()
            self.residuum.assign(self.rhs_ode)
            with self.residuum.dat.vec as res, self.tdens_h.dat.vec as tdens_vec:
                res *= tdens_vec

            # marks the gradient as computed    
            self.gradients_computed = True


        elif "gfvar" in method:
            gfvar = self.gfvar 
            assemble(interpolate(self.gfvar_of_tdens(tdens),gfvar))
            dw = self.ctrl_get('discrepancy_weight')
            pw = self.ctrl_get('penalization_weight')
            rw = self.ctrl_get('regularization_weight')

            if dw > 0:            
                L = assemble(dw * self.discrepancy(pot,self.tdens_of_gfvar(gfvar)))
                lagrangian_fun_reduced = fire_adj.ReducedFunctional(L, fire_adj.Control(gfvar))
                self.gradient_discrepancy = lagrangian_fun_reduced.derivative()
                with self.gradient_discrepancy.dat.vec_ro as gD:
                    msg = utilities.msg_bounds(gD,'grad discrepancy   ')
                    self.print_info(
                        msg=msg,
                        priority=1, 
                        where=['stdout','log']
                        )
                self.gradient_discrepancy.rename('gradient_discrepancy')
            else:
                self.gradient_discrepancy = 0.0

            if pw > 0:
                L = assemble(pw * self.penalization(pot,self.tdens_of_gfvar(gfvar)))
                lagrangian_fun_reduced = fire_adj.ReducedFunctional(L, fire_adj.Control(gfvar))
                self.gradient_penalization = lagrangian_fun_reduced.derivative()
                with self.gradient_penalization.dat.vec_ro as gP:
                    msg = utilities.msg_bounds(gP,'grad penalty       ')
                    self.print_info(
                        msg=msg,
                        priority=1, 
                        where=['stdout','log']
                        )
                self.gradient_penalization.rename('gradient_penalization')
            else:
                self.gradient_penalization = 0.0

            if rw > 0:
                L = assemble(rw * self.regularization(pot, gfvar))
                self.regularization_fun_reduced = fire_adj.ReducedFunctional(L, fire_adj.Control(gfvar))
                self.gradient_regularization = self.regularization_fun_reduced.derivative()
                with self.gradient_regularization.dat.vec_ro as gR:
                    msg = utilities.msg_bounds(gR,'grad regularization')
                    self.print_info(
                        msg=msg,
                        priority=1, 
                        where=['stdout','log']
                        )
            else:
                self.gradient_regularization = 0.0

            self.gradient_D_P.assign(self.gradient_discrepancy + self.gradient_penalization)
            self.rhs_ode.assign(self.gradient_discrepancy + self.gradient_penalization + self.gradient_regularization)
            self.gradients_computed = True

            self.residuum.assign(self.rhs_ode)
            #with self.rhs_ode.dat.vec_ro as rhs_vec:
                # the derivative contains the mass matrix, 
                # we just sum to get the L1 norm of the residuum
            #    residuum = rhs_vec.norm(PETSc.NormType.NORM_1)
            #return residuum

        else:
            raise ValueError(f'Wrong optimization type {method=}')




    def create_solution(self):
        """
        Intialize solution
        """
        sol = Function(self.fems.pot_tdens_space,name=['pot','tdens'])
        sol.sub(0).vector()[:] = 0.0
        sol.sub(1).vector()[:] = 1.0

        sol.sub(0).rename('pot')
        sol.sub(1).rename('tdens')

        return sol
    
    def set_solution(self, pot=None, tdens=None):
        """
        Intialize solution
        """
        if pot is not None:
            self.sol.sub(0).assign(pot)
        if tdens is not None:
            self.sol.sub(1).assign(tdens)
    
    def get_otp_solution(self, sol):
        """
        Return the solution in the BranchedTransportProblem class
        """
        pot, tdens = sol.subfunctions
        
        pot_h = Function(self.fems.pot_space)
        pot_h.assign(pot)
        pot_h.rename('pot')
        
        tdens_h = Function(self.fems.tdens_space)
        tdens_h.assign(tdens)
        tdens_h.rename('tdens')

        DG0_vec = VectorFunctionSpace(self.mesh,'DG',0)
        vel = Function(DG0_vec)
        assemble(interpolate(- tdens * grad(pot), DG0_vec), tensor = vel)
        vel.rename('vel','Velocity')
        
        return pot, tdens, vel
    
    #@profile  
    def solve(self, callbacks=[]):
        '''
        Args:
        ctrl: Class with controls  (tol, 'max_i'ter, deltat, etc.)
        sol: Mixed function [pot,tdens]. It is changed in place.

        Returns:
         ierr : control flag. It is 0 if everthing worked.
        '''
        
        # Clear tape is required to avoid memory accumalation
        # It works but I don't know why
        # see also https://github.com/firedrakeproject/firedrake/issues/3133
        tape = fire_adj.get_working_tape()
        

        # Initialize the parameter-dependent solvers
        self.setup()
        # open log file
        #if self.ctrl_get('log_verbose') > 0:
        #    f_log = open(self.ctrl_get('log_file'), 'w')
        
        # solve initial 
        ierr = self.solve_pot_PDE(self.sol)
        if ierr != 0:
            self.print_info(
            msg=f'First solve_pot_PDE failed with {ierr}\n. Aborting', 
            priority=0, 
            where=['stdout','log'], 
            color='red')
            return ierr
        avg_outer = self.outer_iterations / max(self.nonlinear_iterations,1)
        self.print_info(
            msg=f'It: {0} avgouter: {avg_outer:.1f}', 
            priority=1, 
            where=['stdout','log'], 
            color='green')
        
        # udpack main controls and start main loop
        max_iter = self.ctrl_get('max_iter')
        
        self.iteration = 0
        ierr_dmk = 0
        while ierr_dmk == 0 and self.iteration < max_iter:
            # update with restarts
            
            msg = f"\nIt: {self.iteration+1} method {self.ctrl_get(['dmk','type'])}"
            self.print_info(msg, priority=2, where=['stdout','log'], color='green')
    
            ierr = self.iterate(self.sol)
            # clean memory every 10 iterations
            if self.iteration%5 == 0:
                # Clear tape is required to avoid memory accumalation
                # It works but I don't know why
                # see also https://github.com/firedrakeproject/firedrake/issues/3133
                tape.clear_tape()
                    
                # other clean up taken from 
                # https://github.com/LLNL/pyMMAopt/commit/e2f83bd932207a8adbd60ae793b3e5a3058daecf
                #TSFCKernel._cache.clear()
                #GlobalKernel._cache.clear()
                #gc.collect()
                #petsc4py.PETSc.garbage_cleanup(self.mesh._comm)
                #petsc4py.PETSc.garbage_cleanup(self.mesh.comm)

            if (ierr != 0):
                ierr_dmk = 1
                self.print_info(f'{ierr=}')
                break

            # study state of convergence
            self.iteration += 1
            if self.iteration == max_iter:
                ierr_dmk = 1
            
            # compute residuum
            self.compute_residuum(self.sol)
            with self.residuum.dat.vec_ro as res_vec:
                residual_opt = res_vec.norm(PETSc.NormType.NORM_1)
            
            avg_outer = self.outer_iterations / max(self.nonlinear_iterations,1)

            msg = (f'It: {self.iteration} '
                +f' dt: {self.deltat:.1e}'
                +f' var:{residual_opt:.1e}'
                +f' nsym:{self.nonlinear_iterations:1d}'
                +f' avgouter: {avg_outer:.1f}')
            self.print_info(
                msg, 
                priority=1,
                where=['stdout','log'],
                color='green'
            )
            with self.sol.dat.vec_ro as sol_vec:
                tdens_vec = sol_vec.getSubVector(self.fems.tdens_is)
                self.print_info(
                    msg=msg_bounds(tdens_vec,'tdens'),
                    priority=1, 
                    where=['main','log'],
                    color='black')
                    
            # call user-defined callbacks   
            for callback in callbacks:
                callback(self)

            # check convergence
            if (residual_opt < self.ctrl_get('optimization_tol')):
                ierr_dmk = 0
                break

        #if self.ctrl_get('log_verbose') > 0:
        #    f_log.close()

        tape.clear_tape()
                    
        # other clean up taken from 
        # https://github.com/LLNL/pyMMAopt/commit/e2f83bd932207a8adbd60ae793b3e5a3058daecf
        TSFCKernel._cache.clear()
        GlobalKernel._cache.clear()
        gc.collect()
        petsc4py.PETSc.garbage_cleanup(self.mesh._comm)
        petsc4py.PETSc.garbage_cleanup(self.mesh.comm)

        return ierr_dmk
              
    def discrepancy(self, pot, tdens ):
        '''
        Measure the discrepancy between I(tdens) and the observed data.
        '''
        # print min and max of tdens
        self.image_h = self.tdens2image_map(tdens)
        dis = self.confidence * 0.5 * (self.image_h - self.img_observed)**2 * dx
        return dis
    
    def joule(self, pot, tdens):
        '''
        Joule dissepated energy functional
        '''
        min_tdens = self.ctrl_get('min_tdens')
        weight = ( min_tdens + tdens) / self.btp.kappa**2
        joule_fun = ( self.btp.source - self.btp.sink) * pot * dx  - self.fems.Laplacian_Lagrangian(pot, weight)

        return joule_fun

    def weighted_mass(self, pot, tdens):
        '''
        Weighted tdens mass
        :math:`\int_{\Omega}\frac{1}{2\gamma} \mu^{\gamma} dx`
        '''
        return  0.5 * (tdens ** self.btp.gamma) /  self.btp.gamma  * dx

    def penalization(self, pot, tdens): 
        ''' 
        Definition of the penalization functional as the branched transport energy
        in FCP2021 (use Citations.print_all() to see the reference)
        The penalization is defined as
        :math:`\int_{\Omega} f u dx - \frac{\mu |\nabla u|^2}{2}+ \frac{1}{2\gamma} \mu^{\gamma} dx`
        '''
        otp_pen = self.joule(pot, tdens) + self.weighted_mass(pot,tdens)

        return otp_pen
    

    def regularization(self, pot, tdens):
        ''' 
        Definition of the regularization Form.
        Args:
            pot: potential function
            tdens: transport density function
        returns:
            reg: regularization functional
        '''
        reg = self.fems.Laplacian_Lagrangian(tdens)

        return reg
                
    def Lagrangian(self, pot, tdens):
        ''' 
        Definition of energy minimizated by the niot solver
        args:
            pot: potential function
            tdens: transport density function
        returns:
            Lag: Lagrangian functional = w_0*discrepancy + w_1*penalization + w_2*regularization
        '''
        wr = self.ctrl_get('regularization_weight')
        wp = self.ctrl_get('penalization_weight')
        wd = self.ctrl_get('discrepancy_weight')

        Lag = wp * self.penalization(pot,tdens)
        if abs(wd) > 1e-16:
            Lag += wd * self.discrepancy(pot,tdens)
        if abs(wr) > 1e-16:
            Lag += wr * self.regularization(pot,tdens)
        return Lag
    
    def solve_pot_PDE(self, sol, tol = None):
        '''
        The pot in sol=[pot,tdens] is updated so that it solves the PDE
        associated to the Lagrangian for a given tdens.
        
        argsolve(ses:
            ctrl: Class with controls how we solve
            sol: Mixed function [pot,tdens], changed in place.
         

        returns:
            ierr : control flag (=0 if everthing worked)
        '''
        # Define the PDE for pot varible only 
        # TODO: is there a better way to do define the PDE 
        # obtain taking the partial derivative of the Lagrangian?   
        
        pot, tdens = sol.subfunctions
        self.pot_h.assign(pot)
        self.tdens_h.assign(tdens)
        with self.tdens_h.dat.vec as td:
            self.print_info(
                msg=utilities.msg_bounds(td,'tdens'),
                priority=3, 
                where=['stdout','log'],
                color='black')
        with self.pot_h.dat.vec as p:
            self.print_info(
                msg=utilities.msg_bounds(p,'pot0'),
                priority=3, 
                where=['stdout','log'],
                color='black')

        
        # solve the problem
        try:
            # explicitly set the tolerance
            #if tol is not None:
            #    self.pot_solver.parameters['ksp_rtol'] = tol
            self.pot_solver.solve()
        except:
            pass
        ierr = self.pot_solver.snes.getConvergedReason()
        #self.pot_solver.snes.ksp.view()
        
        msg =  linalg.info_ksp(self.pot_solver.snes.ksp)
        self.print_info(
            msg, 
            priority=2, 
            where=['stdout','log'], 
            color='black')
           
            
        if (ierr < 0):
            self.print_info(msg)
        else:
            ierr = 0

        # get info of solver
        self.nonlinear_iterations = self.pot_solver.snes.getIterationNumber()
        self.outer_iterations = self.pot_solver.snes.getLinearSolveIterations()
        
        # move the pot solution in sol
        sol.sub(0).assign(self.pot_h)

        return ierr
    
    #@profile
    def tdens_mirror_descent_explicit(self, sol):
        # Tdens is udpdate along the direction of the gradient
        # of the energy w.r.t. tdens multiply by tdens**(2-gamma)
        # 
        #
        pot, tdens = sol.subfunctions
        self.pot_h.assign(pot) 
        self.tdens_h.assign(tdens)
        

        # We compute the gradient w.r.t to tdens of the Lagrangian
        if not self.gradients_computed:
            self.compute_residuum(sol)
        
        self.rhs_ode.assign(-(self.gradient_discrepancy 
                                + self.gradient_penalization 
                                  + self.gradient_regularization))
        with self.rhs_ode.dat.vec as rhs:
            self.print_info(
                msg=utilities.msg_bounds(rhs,'gradient'),
                priority=2, 
                where=['stdout','log'],
                color='black')


        # compute a scaling vector for the gradient
        scaling = Function(self.fems.tdens_space)
        self.increment_h = Function(self.fems.tdens_space)
        
        gradient_scaling = self.ctrl_get(['dmk','tdens_mirror_descent_explicit','gradient_scaling'])
        if gradient_scaling == 'dmk':            
            tdens_power = 2 - self.btp.gamma
        elif gradient_scaling == 'mirror_descent':
            tdens_power = 1.0
        elif gradient_scaling == 'no':
            tdens_power = 0.0     
        else:
            raise ValueError(f'Wrong scaling method {gradient_scaling=}')
        scaling = assemble(interpolate(self.tdens_h**tdens_power,self.fems.tdens_space))

        with self.rhs_ode.dat.vec as rhs, scaling.dat.vec_ro as scaling_vec, self.increment_h.dat.vec as d, self.tdens_h.dat.vec_ro as tdens_vec:
            #
            # estimate the step lenght
            # 
            self.fems.inv_tdens_mass_matrix.solve(rhs, d)           

            # scale the gradient w.r.t. tdens by tdens**tdens_power itself
            d *= scaling_vec

            ctrl_step = self.ctrl_get(['dmk','tdens_mirror_descent_explicit','deltat'])
            step = set_step(d,tdens_vec, 
                            self.deltat,
                            **ctrl_step)

            self.print_info(
                msg=utilities.msg_bounds(d,'increment tdens')+f' dt={step:.2e}',
                priority=2, 
                where=['stdout','log'],
                color='black')            
            self.deltat = step

            # update tdens
            tdens_vec.axpy(step, d)
            self.print_info(
                msg=utilities.msg_bounds(tdens_vec,'tdens')+f' dt={step:.2e}',
                priority=2,
                where=['stdout','log'],
                color='blue')
            

        # threshold from below tdens
        utilities.threshold_from_below(self.tdens_h, 0)

        # assign new tdens to solution
        sol.sub(1).assign(self.tdens_h)
        
        # compute pot associated to new tdens
        tol = self.ctrl_get('constraint_tol')
        ierr = self.solve_pot_PDE(sol, tol=tol) 

        self.print_info(
                msg='UPDATE DONE',
                priority=2, 
                where=['stdout','log'],
                color='black')        
        
        return ierr            
    
    def tdens_mirror_descent_semi_implicit(self, sol):
        # Tdens is udpdate along the direction of the gradient
        # of the energy w.r.t. tdens multiply by tdens**(2-gamma)
        # 
        #
        pot, tdens = sol.subfunctions
        self.pot_h.assign(pot) 
        self.tdens_h.assign(tdens)
       
        # We compute the gradient w.r.t to tdens of the Lagrangian
        # Since the Lagrangian contains the map tdens2image, 
        # we need to use the adjoint method, where the Jacobian-vector product
        # of tdens2image is computed automatically.
        #     
        # We follow the example in 
        # see also https://www.dolfin-adjoint.org/en/latest/documentation/custom_functions.html
        # Note how we need to write 
        #   L=assemble(functional)
        #   reduced_functional = ReducedFunctional(functional, Control(tdens))
        #   compute gradient
        # instead of
        #   L=functional
        #   PDE = derivative(L,tdens)
        #   gradient = assemble(PDE)
        
        wd = self.ctrl_get('discrepancy_weight')
        wp = self.ctrl_get('penalization_weight')
        wr = self.ctrl_get('regularization_weight')

        self.lagrangian_fun = assemble(
            wd * self.discrepancy(pot, tdens)
            + wp * self.penalization(pot, tdens)
            )
        var = fire_adj.Control(tdens)
        self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.lagrangian_fun, var )
        self.rhs_ode = self.lagrangian_fun_reduced.derivative()
        self.rhs_ode *= -1 # minus gradient


        # compute a scaling vector for the gradient
        scaling = Function(self.fems.tdens_space)
        self.increment_h = Function(self.fems.tdens_space)

        gradient_scaling = self.ctrl_get(['dmk','tdens_mirror_descent_semi_implicit','gradient_scaling'])
        if gradient_scaling == 'dmk':            
            tdens_power = 2 - self.btp.gamma
        elif gradient_scaling == 'mirror_descent':
            tdens_power = 1.0
        elif gradient_scaling == 'no':
            tdens_power = 0.0     
        else:
            raise ValueError(f'Wrong scaling method {gradient_scaling=}')
        scaling = assemble(interpolate(self.tdens_h**tdens_power,self.fems.tdens_space))

        with self.rhs_ode.dat.vec as rhs, scaling.dat.vec_ro as scaling_vec, self.increment_h.dat.vec as d, self.tdens_h.dat.vec_ro as tdens_vec:
            #
            # estimate the step lenght
            # 
            self.fems.inv_tdens_mass_matrix.solve(rhs, d)           

            # scale the gradient w.r.t. tdens by tdens**tdens_power itself
            d *= scaling_vec

            ctrl_step = self.ctrl_get(['dmk','tdens_mirror_descent_semi_implicit','deltat'])
            step = set_step(d,tdens_vec, 
                            self.deltat,
                            **ctrl_step)

            self.print_info(
                msg=utilities.msg_bounds(d,'increment tdens')+f' dt={step:.2e}',
                priority=2, 
                where=['stdout','log'],
                color='black')            
            self.deltat = step

            
            self.print_info(
                utilities.msg_bounds(tdens_vec,'tdens'),
                priority=2, 
                color='blue')
            wr = self.ctrl_get('regularization_weight')
            shift = step * wr
            #self.shift_semi_implicit.assign(shift) # this change the form
            self.setup_increment_solver(shift=shift)
                
            test = TestFunction(self.fems.tdens_space)
            tdens_integrated = assemble(self.tdens_h * test * dx)
            with tdens_integrated.dat.vec_ro as tdens0_vec:
                rhs *= scaling_vec
                rhs.scale(step)
                rhs.axpy(1.0, tdens0_vec)
            
            self.IncrementSolver.solve(rhs, tdens_vec) 
            self.print_info(
            msg=self.IncrementSolver.info(),
            priority=2,
            where=['stdout','log'])



            # update
            tdens_vec.axpy(step, d)
            self.print_info(
                msg=utilities.msg_bounds(tdens_vec,'tdens')+f' dt={step:.2e}',
                priority=2,
                where=['stdout','log'],
                color='blue')
            

        # threshold from below tdens
        utilities.threshold_from_below(self.tdens_h, 0)

        # assign new tdens to solution
        sol.sub(1).assign(self.tdens_h)
        
        # compute pot associated to new tdens
        tol = self.ctrl_get('constraint_tol')
        ierr = self.solve_pot_PDE(sol, tol=tol)        
        
        return ierr  
    
    def tdens_logarithmic_barrier(self, sol):
        # Tdens is udpdate along the direction of the gradient
        # of the energy w.r.t. tdens multiply by tdens**(2-gamma)
        # 
        #
        pot, tdens = sol.subfunctions
        self.pot_h.assign(pot) 
        self.tdens_h.assign(tdens)
       
        # We compute the gradient w.r.t to tdens of the Lagrangian
        # Since the Lagrangian contains the map tdens2image, 
        # we need to use the adjoint method, where the Jacobian-vector product
        # of tdens2image is computed automatically.
        #     
        # We follow the example in 
        # see also https://www.dolfin-adjoint.org/en/latest/documentation/custom_functions.html
        # Note how we need to write 
        #   L=assemble(functional)
        #   reduced_functional = ReducedFunctional(functional, Control(tdens))
        #   compute gradient
        # instead of
        #   L=functional
        #   PDE = derivative(L,tdens)
        #   gradient = assemble(PDE)
        
        wd = self.ctrl_get('discrepancy_weight')
        wp = self.ctrl_get('penalization_weight')
        wr = self.ctrl_get('regularization_weight')
        eps = self.ctrl_get(['dmk','tdens_logarithmic_barrier','eps'])

        #eps = eps0 * (0.99)**self.iteration
        with self.tdens_h.dat.vec_ro as tdens_vec:
            eps = max(1e-4, tdens_vec.min()[1])
        

        self.lagrangian_fun = assemble(
            wd * self.discrepancy(pot, tdens)
            + wp * self.penalization(pot, tdens)
            - eps * ln(tdens)*dx
            )
        var = fire_adj.Control(tdens)
        self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.lagrangian_fun, var )
        self.rhs_ode = self.lagrangian_fun_reduced.derivative()
        self.rhs_ode *= -1 # minus gradient


        # compute a scaling vector for the gradient
        self.increment_h = Function(self.fems.tdens_space)

        with self.rhs_ode.dat.vec as rhs, self.increment_h.dat.vec as d, self.tdens_h.dat.vec_ro as tdens_vec:
            #
            # estimate the step lenght
            # 
            self.fems.inv_tdens_mass_matrix.solve(rhs, d)           

            # scale the gradient w.r.t. tdens by tdens**tdens_power itself
            ctrl_step = self.ctrl_get(['dmk','tdens_logarithmic_barrier','deltat'])
            step = set_step(d,tdens_vec, 
                            self.deltat,
                            **ctrl_step)

            self.print_info(
                msg=utilities.msg_bounds(d,'increment tdens')+f' dt={step:.2e}',
                priority=2, 
                where=['stdout','log'],
                color='black')            
            self.deltat = step

            
            self.print_info(
                utilities.msg_bounds(tdens_vec,'tdens'),
                priority=2, 
                color='blue')
            wr = self.ctrl_get('regularization_weight')
            shift = step * wr
            self.shift_semi_implicit.assign(shift) # this change the form
            self.setup_increment_solver(shift=shift)
                
            test = TestFunction(self.fems.tdens_space)
            tdens_integrated = assemble(self.tdens_h * test * dx)
            with tdens_integrated.dat.vec_ro as tdens0_vec:
                rhs.scale(step)
                rhs.axpy(1.0, tdens0_vec)
            
            self.IncrementSolver.solve(rhs, tdens_vec) 
            self.print_info(
            msg=self.IncrementSolver.info(),
            priority=2,
            where=['stdout','log'])

            #self.rhs_ode.assign(self.increment_h)

            # update
            tdens_vec.axpy(step, d)
            self.print_info(
                msg=utilities.msg_bounds(tdens_vec,'tdens')+f' dt={step:.2e}',
                priority=2,
                where=['stdout','log'],
                color='blue')
            

        # threshold from below tdens
        utilities.threshold_from_below(self.tdens_h, 0)

        # assign new tdens to solution
        sol.sub(1).assign(self.tdens_h)
        
        # compute pot associated to new tdens
        tol = self.ctrl_get('constraint_tol')
        ierr = self.solve_pot_PDE(sol, tol=tol)        
        
        return ierr  
              
       
    def tdens_of_gfvar(self,gfvar):
        #return sqrt(gfvar)
        return (gfvar)**(2/self.btp.gamma)
        
        
    
    def gfvar_of_tdens(self,tdens):
        #return (tdens)**2
        return (tdens)**(self.btp.gamma/2)
        
    def gfvar_gradient_descent_explicit(self, sol):
        '''
        Update of using transformation tdens_of_gfvar and gradient descent
        args:
            sol : Class with unkowns (pot,tdens, in this case)
        returns:
            ierr : control flag. It is 0 if everthing worked.
        Update of gfvar using gradient descent direction
        '''
        method_ctrl = self.ctrl_get(['dmk','gfvar_gradient_descent_explicit'])


        # convert tdens to gfvar
        pot , tdens = sol.subfunctions
        gfvar = self.gfvar
        assemble(interpolate(self.gfvar_of_tdens(tdens),tensor= gfvar))

        # compute gradient of energy w.r.t. gfvar
        # see tdens_mirror_descent for more details on the implementation
        dw = self.ctrl_get('discrepancy_weight')
        pw = self.ctrl_get('penalization_weight')
        rw = self.ctrl_get('regularization_weight')
        L = assemble(
            dw * self.discrepancy(pot,self.tdens_of_gfvar(gfvar))
            + pw * self.penalization(pot,self.tdens_of_gfvar(gfvar))
            + rw * self.regularization(pot, gfvar)
            )
        
        #with fire_adj.stop_annotating():
        control_var = fire_adj.Control(gfvar)
        reduced_functional = fire_adj.ReducedFunctional(L, control_var)
        self.rhs_ode = reduced_functional.derivative()
        self.rhs_ode *= -1


        update = Function(self.fems.tdens_space)
        with self.rhs_ode.dat.vec as rhs, gfvar.dat.vec_ro as gfvar_vec, update.dat.vec as d:
            # scale by the inverse mass matrix
            self.fems.inv_tdens_mass_matrix.solve(rhs, d)
            
            # update

            ctrl_step = method_ctrl['deltat']
            step = set_step(d, gfvar_vec, 
                            self.deltat,
                            **ctrl_step)
            self.deltat = step
            self.print_info(utilities.msg_bounds(d,'gfvar increment')+f' dt={step:.2e}',priority=3, color='blue')
            
            # update
            gfvar_vec.axpy(step, d)
            
            self.print_info(utilities.msg_bounds(gfvar_vec,'gfvar'),priority=3, color='blue')
        
        # convert gfvar to tdens
        utilities.threshold_from_below(gfvar, 0)
        assemble(interpolate(self.tdens_of_gfvar(gfvar),tensor=self.tdens_h))
        sol.sub(1).assign(self.tdens_h)
        with self.tdens_h.dat.vec_ro as tdens_vec:
            self.print_info(utilities.msg_bounds(tdens_vec,'tdens'), priority=2,color='blue')   

        # compute pot associated to new tdens
        tol = self.ctrl_get('constraint_tol')
        ierr = self.solve_pot_PDE(sol, tol=tol)
        
        return ierr
    
    def gfvar_gradient_descent_semi_implicit(self, sol):
        '''
        Update of using transformation tdens_of_gfvar and gradient descent
        args:
            sol : Class with unkowns (pot,tdens, in this case)
        returns:
            ierr : control flag. It is 0 if everthing worked.
        Update of gfvar using gradient descent direction
        '''
        self.sol_old.assign(sol)

        self.restart = 0
        max_restart = self.ctrl_get('max_restart')
        ierr = -1
        while self.restart < max_restart and ierr != 0:
            # convert tdens to gfvar
            pot , tdens = sol.subfunctions
            self.tdens_h.assign(tdens)
            gfvar = self.gfvar
            assemble(interpolate(self.gfvar_of_tdens(tdens),tensor=gfvar))


            # compute gradient of energy w.r.t. gfvar
            # see tdens_mirror_descent for more details on the implementation
            #PETSc.Sys.Print(f"niot_solver starting gradient",comm=self.comm)
            if self.gradients_computed:
                # this is done to avoid recomputing the gradient 
                self.rhs_ode = self.gradient_D_P
                self.rhs_ode *= -1.0
            else:
                dw = self.ctrl_get('discrepancy_weight')
                pw = self.ctrl_get('penalization_weight')
                L = assemble(
                    dw * self.discrepancy(pot,self.tdens_of_gfvar(gfvar))
                    + pw * self.penalization(pot,self.tdens_of_gfvar(gfvar))
                    )
                
                #with fire_adj.stop_annotating():
                var = fire_adj.Control(gfvar)
                reduced_functional = fire_adj.ReducedFunctional(L, var)
                self.rhs_ode = reduced_functional.derivative()
                self.rhs_ode *= -1.0
            
            #PETSc.Sys.Print(f"niot_solver starting update",comm=self.comm)
            update = Function(self.fems.tdens_space)
            with self.rhs_ode.dat.vec as rhs, gfvar.dat.vec_ro as gfvar_vec, update.dat.vec as d:
                # scale by the inverse mass matrix
                self.fems.inv_tdens_mass_matrix.solve(rhs, d)
                #PETSc.Sys.Print(f"niot_solver increment",comm=self.comm)
                # estimate the step lenght
                ctrl_step = self.ctrl_get(['dmk','gfvar_gradient_descent_semi_implicit','deltat'])
                if self.restart == 0:
                    step = set_step(d,gfvar_vec, 
                                self.deltat,
                                **ctrl_step)
                    self.deltat = step
                else:
                    self.deltat *= ctrl_step['contraction']
                #PETSc.Sys.Print(f"niot_solver deltat",comm=self.comm)
                self.print_info(utilities.msg_bounds(d,'gfvar increment')+f' dt={self.deltat:.2e}', priority=2, where=['stdout','log'], color='blue')
                
                
                # 
                # M(gf-gf_0)/step + grad P+D(gf  ) + wr (-L) gf= 0
                # ~ semi_implicit 
                # M(gf-gf_0)/step + grad P+D(gf_0) + wr (-L) gf = 0
                #
                # (M+step*wr*L) gf = M gf_0 + step*rhs
                #
                self.print_info(utilities.msg_bounds(gfvar_vec,'gfvar'), priority=3, where=['stdout','log'], color='blue')
                wr = self.ctrl_get('regularization_weight')
                shift=step*wr
                self.shift_semi_implicit.assign(shift) # this change the form
                self.setup_increment_solver(shift=step*wr)
                #PETSc.Sys.Print(f"niot_solver setup solve increment",comm=self.comm)
                
                test = TestFunction(self.fems.tdens_space)
                gf0 = assemble(gfvar*test*dx)
                with gf0.dat.vec_ro as g0_vec:
                    rhs.scale(step)
                    rhs.axpy(1.0, g0_vec)
                
                self.IncrementSolver.solve(rhs, gfvar_vec) 
                #PETSc.Sys.Print(f"niot_solver solve increment",comm=self.comm)
                self.print_info(
                msg=self.IncrementSolver.info(),
                priority=2,
                where=['stdout','log'])

                self.print_info(utilities.msg_bounds(gfvar_vec,'gfvar'), priority=2, where=['stdout','log'], color='blue')
            
            # convert gfvar to tdens
            utilities.threshold_from_below(gfvar, 0)
            assemble(interpolate(self.tdens_of_gfvar(gfvar),tensor = self.tdens_h))
            #PETSc.Sys.Print(f"niot_solver tdens",comm=self.comm)
            sol.sub(1).assign(self.tdens_h)
            with self.tdens_h.dat.vec_ro as tdens_vec:
                self.print_info(utilities.msg_bounds(tdens_vec,'tdens'), priority=2, where=['stdout','log'], color='blue')   
            


            #PETSc.Sys.Print(f"niot_solver solve pot",comm=self.comm)
            # compute pot associated to new tdens
            tol = self.ctrl_get('constraint_tol')
            ierr = self.solve_pot_PDE(sol, tol=tol)

            if ierr != 0:
                sol.assign(self.sol_old)
                fire_adj.stop_annotating()
                self.restart += 1
                self.print_info(f'Restart {self.restart} failed with {ierr}.', priority=0, where=['stdout','log'], color='red')
            
        


        return ierr
    
   
    #@profile  
    def iterate(self, sol):
        '''
        Args:
         ctrl : Class with controls  (tol, max_iter, deltat, etc.)
         solution updated from time t^k to t^{k+1} solution updated from time t^k to t^{k+1} sol : Class with unkowns (pot,tdens, in this case)

        Returns:
         ierr : control flag. It is 0 if everthing worked.

        '''
        method = self.ctrl_get(['dmk','type'])
        if method == 'tdens_mirror_descent_explicit':
            ierr = self.tdens_mirror_descent_explicit(sol)
            return ierr
        
        if method == 'tdens_mirror_descent_semi_implicit':
            self.sol_old.assign(sol)
            ierr = -1
            while self.restart < self.ctrl_get('max_restart') and ierr != 0:
                ierr = self.tdens_mirror_descent_semi_implicit(sol)
                if ierr != 0:
                    sol.assign(self.sol_old)
                    self.restart += 1                    
            return ierr
        
        elif method == 'gfvar_gradient_descent_explicit':
            ierr = self.gfvar_gradient_descent_explicit(sol)
            return ierr
        
        elif method == 'gfvar_gradient_descent_semi_implicit':
            ierr = self.gfvar_gradient_descent_semi_implicit(sol)
            return ierr
        
        elif method == 'tdens_logarithmic_barrier':
            ierr = self.tdens_logarithmic_barrier(sol)
            return ierr
        
        else:
            raise ValueError('value: dmk_type not supported.\n',
                              f'Passed:{method}')   
    
    def save_solution(self, sol, filename):            
        '''
        Save into a file the solution [pot,tdens] and velocity
        '''
        pot, tdens = sol.subfunctions
        utilities.save2pvd([pot,tdens],filename)

    def save_checkpoint(self, sol, filename):
        '''
        Write solution to a checkpoint file
        '''
        # check extension
        if (filename[-3:] != '.h5'):
            raise ValueError('The filename must have extension .h5')

        with CheckpointFile(filename, 'w') as afile:
            afile.save_mesh(self.mesh)  
            sol.rename('sol','Solution')
            afile.save_function(sol)
        

    def load_checkpoint(self, filename):
        '''
        Load solution from a checkpoint file
        '''
         # check extension
        if (filename[-3:] != '.h5'):
            raise ValueError('The filename must have extension .h5')

        with CheckpointFile(filename, 'r') as afile:
            mesh = afile.load_mesh('mesh')
            sol = afile.load_function(mesh, 'sol')
        return sol 



def callback_record_algorithm(self, save_solution, save_directory, save_solution_every):
    """
    Record data along algorithm exceution
    """
    
    # unpack related controls
    current_iteration = self.current_iteration
    sol = self.sol

    if save_solution == 'no':
        pass
    elif save_solution == 'all':
        filename = os.path.join(save_directory,f'sol{current_iteration:06d}.pvd')
        self.save_solution(sol,filename)
    elif (save_solution == 'some') and (current_iteration % save_solution_every == 0):
        pot, tdens = sol.subfunctions          
        assemble(interpolate(self.tdens2image(tdens),tensor= self.image_h))

        filename = os.path.join(save_directory,f'sol{current_iteration:06d}.pvd')
        utilities.save2pvd([pot, tdens, self.image_h],filename)
    
