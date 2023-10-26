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

from petsc4py import PETSc
import time 
#from memory_profiler import profile


from . import utilities
from . import optimal_transport as ot

from . import conductivity2image as conductivity2image
from . import linear_algebra_utilities as linalg
SNESReasons = utilities._make_reasons(PETSc.SNES.ConvergedReason())

# function operations
from firedrake import *
#from firedrake_adjoint import ReducedFunctional, Control
#from pyadjoint.reduced_functional import  ReducedFunctional
#from pyadjoint.control import Control

#import firedrake.adjoint as fireadj # does not work ...
#from firedrake.adjoint import * # does give error when interpoalte is called
#import firedrake.adjoint.ReducedFunctional as ReducedFunctional # does not work ...
import firedrake.adjoint as fire_adj
fire_adj.continue_annotation()

from petsc4py import PETSc as p4pyPETSc

# include all citations
utilities.include_citations(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), 
                     '../../citations/citations.bib')
    ))



def msg_bounds(vec,label):
    min = vec.min()[1]
    max = vec.max()[1]
    return ''.join([f'{min:2.1e}','<=',label,'<=',f'{max:2.1e}'])

def get_step_lenght(x,increment,x_lower_bound=0.0,step_lower_bound=1e-16):
    '''
    Get the step lenght to ensure that the new iterate is above as the lower bound
    '''
    np_x = x.array
    np_increment = increment.array
    #print(utilities.msg_bounds(x,'x'))
    #print(utilities.msg_bounds(increment,'increment'))
    negative = (np_increment<0).any()
    #print('negative',negative)
    if negative:
        negative_indeces  = np.where(np_increment<0)[0]
        #print('negative',len(negative_indeces),'size',len(np_increment))
        #print(np.min(np_increment[negative_indeces]))
        #print(np.min(np_x[negative_indeces]))
        step = np.min((x_lower_bound-np_x[negative_indeces])/np_increment[negative_indeces])
        #print('step lenght', step)
        if step<0:
            #print('x',np_x)
            #print('increment',np_increment)
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
    def __init__(self, mesh, pot_space='CR', pot_deg=1, tdens_space='DG', tdens_deg=0):
        #tdens_fem='DG0',pot_fem='P1'):
        '''
        Initialize FEM spaces used to discretized the problem
        '''  
        # For Pot unknow, create fem, function space, trial and test functions
        #meshes = MeshHierarchy(mesh, 1)
        #print('meshes',meshes)
        #print('meshes',len(meshes))

        self.pot_fem = FiniteElement(pot_space, mesh.ufl_cell(), pot_deg)
        self.pot_space = FunctionSpace(mesh, self.pot_fem)
        self.pot_trial = TrialFunction(self.pot_space)
        self.pot_test = TestFunction(self.pot_space)

        if (pot_space=='DG') and (pot_deg == 0):
            if mesh.geometric_dimension() == 2:
                x,y = mesh.coordinates
                x_func = interpolate(x, self.pot_space)
                y_func = interpolate(y, self.pot_space)
                self.delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)
            elif mesh.geometric_dimension() == 3:
                x,y,z = mesh.coordinates
                x_func = interpolate(x, self.pot_space)
                y_func = interpolate(y, self.pot_space)
                z_func = interpolate(z, self.pot_space)
                self.delta_h = sqrt(jump(x_func)**2 
                                    + jump(y_func)**2 
                                    + jump(z_func)**2)
        
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

    
    def save_solution(self, sol, file_name):
        '''
        Save solution to file
        '''
        pot, tdens = sol.subfunctions
        pot.rename('pot','Potential')
        tdens.rename('tdens','Optimal Transport Tdens')

        DG0_vec = VectorFunctionSpace(self.pot_space.mesh(),'DG',0)
        vel = Function(DG0_vec)
        vel.interpolate(-tdens*grad(pot))
        vel.rename('vel','Velocity')

        utilities.save2pvd([pot, tdens, vel],file_name)

    def Laplacian_form(self, which_variable):
        """
        Return the Laplacian form for the given space
        with zero-Neumann boundary conditions
        """
        if which_variable == 'pot':
            space = self.pot_space
        elif which_variable == 'tdens':
            space = self.tdens_space
        else:
            raise ValueError('Wrong variable passed. Only pot or tdens are implemented')
        
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
    
    def cell2face(self, fun, approach='arithmetic_mean'):
        if approach == 'arithmetic_mean':
            return (fun('+') + fun('-')) / 2
        elif approach == 'harmonic_mean':
            return 2 * fun('+') * fun('-') / ( fun('+') + fun('-') )
            #return conditional( gt(avg(fun), 0.0), fun('+') * fun('-') / avg(fun), 0.0)
            
        else:
            raise ValueError('Wrong approach passed. Only arithmetic_mean or harmonic_mean are implemented')

    def Laplacian_Lagrangian(self, u, weight=None, cell2face='arithmetic_mean'):
        """
        Return the Lagrangian of a weighted-Laplacian equation for u.
        The weight is a function of the mesh.
        If the weight is not passed, it is assumed to be 1.0.
        """
        space = u.function_space()
        if weight is None:
            weight = Constant(1.0)
        
        if space.ufl_element().degree() == 0:
            # the weight need to be "projected to the facets"
            facet_weight = self.cell2face(weight, cell2face)
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


class Controls:
    '''
    Class with Dmk Solver 
    '''
    global_ctrl={'tol':1e-2,
                 'max_iter':100,
                 'max_restart':2,
                 'optimization_method':'dmk',
                 }
    
    discretization_ctrl={'spaces':'CR1DG0',
                         'cell2face':'harmonic_mean',
                         }
    
    tdens2image_ctrl={'tdens2image':'identity',
                        'scaling':1.0,
                        'sigma_smooth':1.0}
    
    inpainting_ctrl={'weight_discrepancy':1.0,
                    'weight_regularization':0.0,
                    'gamma':0.5}

    dmk_ctrl={'time_discretization_method':'mirror_descent',
            'gradient_scaling':'dmk',
            'deltat':0.5,
            'nonlinear_tol':1e-6,
            'linear_tol':1e-6,
            'nonlinear_max_iter':30,
            'linear_max_iter':1000,
            }   

    def __init__(self,
                tol=1e-2,
                max_iter=100,
                max_restart=2,
                # conductivity to image parameters
                tdens2image='identity',
                scaling=1.0,
                sigma_smoothing=1e-4,
                # inpainting parameters
                weight_discrepancy=1.0,
                weight_regularization=0.0,
                gamma=0.5,
                # discretization parameters
                spaces='CR1DG0',
                cell2face='harmonic_mean',
                # optimization parameters
                time_discretization_method='mirror_descent',
                gradient_scaling='dmk',
                deltat=0.5,
                nonlinear_tol=1e-6,
                linear_tol=1e-6,
                nonlinear_max_iter=30,
                linear_max_iter=1000,
                ):
        '''
        Set the controls of the Dmk algorithm
        '''
        #: float: stop tolerance
        self.tol = tol

        #: float: branched transport exponent
        self.gamma = gamma

        #############################
        # tdens to image parameters 
        #############################

        #: str: conductivity to image approach
        # 'identity': identity map
        # 'laplacian': guassian filter-like based on heat equation
        self.tdens2image = tdens2image

        # real: multiplicative scaling factor in image = scaling*tdens2image(tdens)
        self.scaling = scaling

        #: real: smoothing parameter for conductivity to image map
        self.sigma_smoothing = sigma_smoothing

        #############################
        # inpainting parameters
        #############################
        self.weight_discrepancy = weight_discrepancy
        self.weight_regularization = weight_regularization


        #############################
        # discretization parameters
        #############################
        self.spaces = spaces
        self.cell2face = cell2face

        #############################
        # optimization parameters
        #############################


        #: character: time discretization approach
        self.time_discretization_method = time_discretization_method
        
        #: int: max number of time steps
        self.max_iter = max_iter

        #: int: max number of update restarts
        self.max_restart = max_restart
        
        #: real: time step size
        self.deltat = deltat

        # variables for set and reset procedure
        self.deltat_control = 'adaptive'
        self.deltat_min = 1e-2
        self.deltat_max = 0.5
        self.deltat_expansion = 2
        #: real : time step size for contraction in case of failure 
        self.deltat_contraction = 0.5

        #: real : lower bound for tdens
        self.min_tdens = 1e-8

        #: str: gradient scaling
        # 'dmk': scale by tdens**(2-gamma)
        # 'mirror_descent': scale by tdens
        self.gradient_scaling = gradient_scaling

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
        self.verbose_log = 2
        self.file_log = 'niot.log'

        #: info save solution
        self.save_solution = 'not'
        #: int: frequency of solution saving
        self.save_frequency = 10
        self.save_directory = './'

    def set_step(self, increment, state=None):
        """
        Set the step lenght according to the control strategy
        and the increment
        """
        if (self.deltat_control == 'adaptive'):
            abs_inc = abs(increment)
            _,d_max = abs_inc.max()
            if (d_max < 0):
                step = self.deltat_max
            else:
                print(f'{d_max=:.2e}')
                step = max(min(1.0 / d_max, self.deltat_max),self.deltat_min)
        elif (self.deltat_control == 'adaptive2'):
            order_down = -1
            order_up = 1
            r = increment / state
            r_np = r.array
            if np.min(r_np) < 0:
                negative = np.where(r_np<0)
                hdown = (10**order_down-1) / r[negative]
                deltat_down = np.min(hdown)
            else:
                deltat_down = self.deltat_max
            if np.max(r_np) > 0:
                positive = np.where(r_np>0)
                hup = (10**order_up-1) / r[positive]
                deltat_up = np.min(hup)
            else:
                deltat_up = self.deltat_max
            step = min(deltat_up,deltat_down)
            step = max(step,self.deltat_min)
            step = min(step,self.deltat_max)

        elif (self.deltat_control == 'expansive'):
            step = max(min(self.deltat * self.deltat_expansion, self.deltat_max),self.deltat_min)
        elif (self.deltat_control == 'fixed'):
            step = self.deltat
        else:
            raise ValueError(f'{self.deltat_control=} not supported')
        return step

    def print_info(self, msg, priority=0, color='black', modes=['main']):
        '''
        Print messagge to stdout and to log 
        file according to priority passed
        '''
        for mode in modes:
            if mode=='main':
                if (self.verbose >= priority): 
                    if color != 'black':
                        msg = utilities.color(color, msg)
                    print('   '*(priority-1)+msg)

            elif mode == 'log':
                if (self.save_log > 0 and self.verbose_log >= priority):
                    print(msg, file=self.log_file)           
            
 
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

    # register citations using Citations class in firedrake
    Citations().register('FCP2021')

    #def __init__(self, corrupted, transport_inputs, ctrl=Controls, confidence=1.0)
    #def __init__(self, corrupted, confidence, source, sinkotp_inputs, ctrl)
    # def __init__(self, spaces, numpy_corrupted, DG0_cell2face='harmonic_mean'):
    #     '''
    #     Initialize solver (spatial discretization) from numpy_image (2d or 3d data)
    #     '''
    #     self.spaces = spaces
    #     if spaces == 'CR1DG0':
    #         print('building mesh')
    #         self.mesh = self.build_mesh_from_numpy(numpy_corrupted, mesh_type='simplicial')
    #         # create FEM spaces
    #         print('building fems')
    #         self.fems = SpaceDiscretization(self.mesh,'CR',1, 'DG',0)
    #     elif spaces == 'DG0DG0':
    #         self.mesh = self.build_mesh_from_numpy(numpy_corrupted, mesh_type='cartesian')
    #         # create FEM spaces
    #         self.fems = SpaceDiscretization(self.mesh,'DG',0,'DG',0)
    #     else:
    #         raise ValueError('Wrong spaces only (pot,tdens) in (CR1,DG0) or (DG0,DG0) implemented')
    #     self.mesh.name = 'mesh'
    #     print(f'Number of cells: {self.mesh.num_cells()}')

    #     self.DG0 = FunctionSpace(self.mesh, 'DG', 0)

    #     # set function 
    #     self.img_observed = self.numpy2function(numpy_corrupted,name='obs')
    
    #     # set niot parameter to default
    #     self.set_inpainting_parameters(
    #         weights=[1.0,1.0,0.0],
    #         confidence=1.0,
    #         tdens2image='identity')
        
        
    #     # init infos
    #     self.outer_iterations = 0
    #     self.nonlinear_iterations = 0
    #     self.nonlinear_res = 0.0

    #     # cached functions
    #     self.rhs_ode = Function(self.fems.tdens_space)
    #     self.tdens_h = Function(self.fems.tdens_space)
    #     self.tdens_h.rename('tdens_h')
    #     self.pot_h = Function(self.fems.pot_space)

    #     alpha = Constant(4.0)
    #     h = CellSize(self.mesh)
    #     h_avg = (h('+') + h('-'))/2.0
    #     self.DG0_scaling = alpha/h_avg
    #     self.normal = FacetNormal(self.mesh)
        
    #     if ((DG0_cell2face == 'arithmetic_mean') or
    #         (DG0_cell2face == 'harmonic_mean')):
    #         self.DG0_cell2face = DG0_cell2face
    #     else:   
    #         raise ValueError('Wrong DG0_cell2face method. Passed '+DG0_cell2face+
    #                         '\n Only arithmetic_mean and harmonic_mean are implemented')
        

    def __init__(self, btp, observed, confidence, ctrl=Controls()):
        '''
        Initialize solver (spatial discretization) from numpy_image (2d or 3d data)
        '''
        self.spaces = ctrl.spaces
        self.mesh = btp.mesh
        if self.spaces == 'CR1DG0':
            self.fems = SpaceDiscretization(self.mesh,'CR',1, 'DG',0)
        elif self.spaces == 'DG0DG0':
            self.fems = SpaceDiscretization(self.mesh,'DG',0,'DG',0)
        else:
            raise ValueError('Wrong spaces only (pot,tdens) in (CR1,DG0) or (DG0,DG0) implemented')
        self.mesh.name = 'mesh'
        print(f'Number of cells: {self.mesh.num_cells()}')
        self.DG0 = FunctionSpace(self.mesh, 'DG', 0)

        # cached functions
        self.rhs_ode = Function(self.fems.tdens_space)
        self.tdens_h = Function(self.fems.tdens_space)
        self.tdens_h.rename('tdens_h')
        self.pot_h = Function(self.fems.pot_space)
        self.pot_h.rename('pot_h')
        self.image_h = Function(self.fems.tdens_space)
        self.image_h.rename('image_h')
        self.image_h.assign(1.0)


        # The class BranchedTransportProblem
        # that contains the physical information of the problem
        # and the branched transport exponent gamma 
        self.btp = btp
        self.min_tdens = ctrl.min_tdens
        self.kappa = Function(self.fems.tdens_space)
        if isinstance(self.btp.kappa, float):
            self.kappa.assign(self.btp.kappa)
        elif isinstance(self.btp.kappa, Function):
            self.kappa.project(self.btp.kappa)
        else:
            raise ValueError('kappa must be a float or a function')
        self.source = self.btp.source
        self.sink = self.btp.sink
        self.Dirichlet = self.btp.Dirichlet
    

        # set img to be inpainted
        self.img_observed = observed
    
        
        if ((ctrl.cell2face == 'arithmetic_mean') or
            (ctrl.cell2face == 'harmonic_mean')):
            self.DG0_cell2face = ctrl.cell2face
        else:   
            raise ValueError('Wrong DG0_cell2face method. Passed '+DG0_cell2face+
                            '\n Only arithmetic_mean and harmonic_mean are implemented')
        

        #######################
        # set niot parameters
        #######################
        # confidence metrix, appaers in the discrepancy term
        self.confidence = confidence
        #self.confidence.rename('confidence','confidence')
        
        # weights of the discrepancy, penalization and transport energy
        self.weights = np.array([ctrl.weight_discrepancy,1.0,ctrl.weight_regularization])

        # map from tdens to image
        print(f'{ctrl.tdens2image=}',f'{ctrl.scaling=}',f'{ctrl.sigma_smoothing=}')
        self.scaling = ctrl.scaling
        if ctrl.tdens2image == 'identity':
            self.tdens2image = lambda x: self.scaling*x
        elif ctrl.tdens2image == 'heat':
            test = TestFunction(self.fems.tdens_space)  
            trial = TrialFunction(self.fems.tdens_space)
            form =  inner(test, trial)*dx
            form += ctrl.sigma_smoothing * self.fems.Laplacian_form('tdens')
                        
            self.HeatMatrix = assemble(form).M.handle
            self.HeatSmoother = linalg.LinSolMatrix(self.HeatMatrix,
                                                    self.fems.tdens_space, 
                                                    solver_parameters={
                                        'ksp_monitor': None,                
                                        'ksp_type': 'cg',
                                        'ksp_rtol': 1e-10,
                                        'pc_type': 'hypre'},
                                        options_prefix='heat_smoother_')
            
            test = TestFunction(self.fems.tdens_space)
            self.rhs_heat = self.tdens_h * test * dx
            self.heat_problem = LinearVariationalProblem(form,
                                                         self.rhs_heat,
                                                         self.tdens_h,
                                                         constant_jacobian=True)
            # self.heat_solver = LinearSolver(self.HeatMatrix,
            #                                  solver_parameters={
            #                                     'ksp_type': 'cg',
            #                                     'ksp_rtol': 1e-10,
            #                                     'pc_type': 'hypre',},
            #                                     options_prefix='heat_solver_')
                                            
            print('defining tdens2image')
            self.tdens2image = lambda fun: self.scaling*conductivity2image.LaplacianSmoothing(fun,self.HeatSmoother)
            #self.tdens2image = lambda fun: self.scaling * self.heat_solver.solve(fun,)
        elif ctrl.tdens2image == 'pm':
            test = TestFunction(self.fems.tdens_space)
            self.image_h.assign(self.img_observed)
            facet_image = self.fems.cell2face(ctrl.min_tdens + self.image_h, self.DG0_cell2face)
            pm_PDE = (
                (self.image_h - self.tdens_h) * test * dx 
                + ctrl.sigma_smoothing * facet_image * jump(self.image_h) * jump(test) / self.fems.delta_h * dS
            )
            self.pm_problem = NonlinearVariationalProblem(
                pm_PDE, self.image_h)

            self.pm_solver = NonlinearVariationalSolver(
                self.pm_problem,
                solver_parameters={
                    'snes_type': 'newtonls',
                    'snes_rtol': 1e-12,
                    'snes_atol': 1e-16,
                    'snes_linesearch_type':'bt',
                    'snes_monitor': None,
                    'ksp_type': 'gmres',
                    'ksp_rtol': 1e-8,
                    'pc_type': 'hypre'},
                options_prefix='porous_solver_')

            def tdens2image_map(fun):
                self.tdens_h.interpolate(fun)
                with self.tdens_h.dat.vec as tdens_vec:
                    print(utilities.msg_bounds(tdens_vec,'tdens_vec'))
                with self.image_h.dat.vec as image_vec:
                    print(utilities.msg_bounds(image_vec,'image_vec'))
                    
                self.pm_solver.solve()
                self.image_h *= self.scaling
                with self.image_h.dat.vec as image_vec:
                    print(utilities.msg_bounds(image_vec,'image_vec'))
                
                return self.image_h
            
            self.tdens2image = tdens2image_map

        
            

        else:
            raise ValueError(f'Map tdens2iamge not supported {ctrl.tdens2image}\n'
                             +'Only identity, heat are implemented')

        
        # init infos
        self.outer_iterations = 0
        self.nonlinear_iterations = 0
        self.nonlinear_res = 0.0

        
        
        self.ctrl = ctrl

    def print(self, ctrl, msg):
        if ctrl.print_stdout:
            print(msg)
        if ctrl.print_log:
            print(msg,file=self.log_file)

        
    def setup_pot_solver(self,ctrl):
        self.pot_PDE = derivative(self.Lagrangian(self.pot_h,self.tdens_h),self.pot_h)
        self.weighted_Laplacian = derivative(-self.pot_PDE,self.pot_h)
        self.rhs = (self.source - self.sink) * self.fems.pot_test * dx

        #test = TestFunction(self.fems.pot_space)
        #pot_PDE = self.forcing * test * dx + tdens * inner(grad(pot_unknown), grad(test)) * dx 
        # Define the Nonlinear variational problem (it is linear in this case)
        #self.u_prob = NonlinearVariationalProblem(self.pot_PDE, self.pot_h)#, bcs=pot_bcs)
        self.u_prob = LinearVariationalProblem(self.weighted_Laplacian, self.rhs, self.pot_h)#, bcs=pot_bcs)

        # set the solver parameters
        snes_ctrl={
            #'snes_rtol': 1e-16,
            #'snes_atol': ctrl.nonlinear_tol,
            #'snes_stol': 1e-16,
            #'snes_type': 'ksponly',
            #'snes_max_it': ctrl.nonlinear_max_iter,
            # krylov solver controls
            'ksp_type': 'cg',
            'ksp_atol': 1e-12,
            'ksp_rtol': ctrl.nonlinear_tol,
            'ksp_divtol': 1e4,
            'ksp_max_it' : 100,
            'ksp_initial_guess_nonzero': True,
            'ksp_norm_type': 'unpreconditioned',
            # preconditioner controls
            'pc_type': 'hypre',
            #'ksp_monitor_true_residual': None,
        }
        print(ctrl.nonlinear_tol)
        if ctrl.verbose >= 1:
            pass#snes_ctrl['snes_monitor'] = None
        if  ctrl.verbose >= 1:
            pass#snes_ctrl['ksp_monitor'] = None
        
        context ={} # left to pass information to the solver
        if self.Dirichlet is None:
            nullspace = VectorSpaceBasis(constant=True,comm=COMM_WORLD)
        
        #self.snes_solver = NonlPDEinearVariationalSolver(self.u_prob,
        self.snes_solver = LinearVariationalSolver(self.u_prob,
                                                solver_parameters=snes_ctrl,
                                                nullspace=nullspace,
                                                appctx=context,
                                                options_prefix='pot_solver_')
        self.snes_solver.snes.ksp.setConvergenceHistory(length=100)
    
    def record_algorithm(self, sol, ctrl, current_iteration):
        """
        Record data along algorithm exceution
        """
        if ctrl.save_solution == 'no':
            pass
        elif ctrl.save_solution == 'all':
            filename = os.path.join(ctrl.save_directory,f'sol{current_iteration:06d}.pvd')
            self.save_solution(sol,filename)
        elif (ctrl.save_solution == 'some') and (current_iteration % ctrl.save_solution_every == 0):
            pot, tdens = sol.subfunctions          
            self.image_h.interpolate(self.tdens2image(tdens))

            filename = os.path.join(ctrl.save_directory,f'sol{current_iteration:06d}.pvd')
            utilities.save2pvd([pot, tdens, self.image_h],filename)
               

    def residual(self, sol):
        """
        Return the residual of the minimization problem
        w.r.t to tdens
        """
        pot, tdens = sol.subfunctions

        self.tdens_h.assign(tdens)
        self.pot_h.assign(pot)
        self.lagrangian_fun = assemble(self.Lagrangian(self.pot_h,self.tdens_h))
        # The stop_annotating is required to avoid memory accumalation
        # It works but I don't know why.   
        with fire_adj.stop_annotating():
            self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.lagrangian_fun, fire_adj.Control(self.tdens_h))
            self.rhs_ode = self.lagrangian_fun_reduced.derivative()
        d = self.fems.tdens_mass_matrix.createVecLeft()
        
        with self.rhs_ode.dat.vec_ro as f_vec, self.tdens_h.dat.vec as tdens_vec:
            self.fems.tdens_inv_mass_matrix.solve(f_vec,d)
            d *= tdens_vec
            residuum = d.norm()#PETSc.NormType.NORM_INFINITY)
        return residuum

    def create_solution(self):
        """
        Intialize solution
        """
        sol = Function(self.fems.pot_tdens_space,name=['pot','tdens'])
        sol.sub(0).vector()[:] = 0.0
        sol.sub(1).vector()[:] = 1.0

        return sol
    
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
        vel.interpolate(- tdens * grad(pot))
        vel.rename('vel','Velocity')
        
        return pot, tdens, vel
    
    #@profile  
    def solve(self, ctrl, sol):
        '''
        Args:
        ctrl: Class with controls  (tol, max_iter, deltat, etc.)
        sol: Mixed function [pot,tdens]. It is changed in place.

        Returns:
         ierr : control flag. It is 0 if everthing worked.
        '''
        # Clear tape is required to avoid memory accumalation
        # It works but I don't know why
        # see also https://github.com/firedrakeproject/firedrake/issues/3133
        tape = fire_adj.get_working_tape()
        tape.clear_tape()

        self.setup_pot_solver(ctrl)
        ierr = self.solve_pot_PDE(ctrl, sol)
        avg_outer = self.outer_iterations / max(self.nonlinear_iterations,1)
        ctrl.print_info(f'It: {0} avgouter: {avg_outer:.1f}', 
                priority=1, color='green', modes=['main','log'])

        iter = 0
        ierr_dmk = 0
        sol_old = cp(sol)           
        tape.clear_tape()
        while ierr_dmk == 0 and iter < ctrl.max_iter:
            # update with restarts
            sol_old.assign(sol)
            nrestart = 0 
            
            while nrestart < ctrl.max_restart:
                msg = f'\nIt: {iter} method {ctrl.time_discretization_method}'
                if nrestart > 0:
                    msg += f'! restart {nrestart} '
                ctrl.print_info(msg, priority=2, color='green', modes=['main','log'])
    
                ierr = self.iterate(ctrl, sol)
                # clean memory every 10 iterations
                if iter%10 == 0:
                    # Clear tape is required to avoid memory accumalation
                    # It works but I don't know why
                    # see also https://github.com/firedrakeproject/firedrake/issues/3133
                    tape.clear_tape()

                if ierr == 0:
                    break
                else:
                    nrestart +=1
                    # reset controls after failure
                    ctrl.deltat = max(ctrl.deltat_min,ctrl.deltat_contraction * ctrl.deltat)
                    msg =(f'{ierr=}. Failure in due to {SNESReasons[ierr]}')
                    ctrl.print_info(msg, priority=1, color='red', modes=['main','log'])

                    # restore old solution
                    sol.assign(sol_old)

            if (ierr != 0):
                print('ierr',ierr)
                break

            # study state of convergence
            iter += 1
            
            var_tdens = self.residual(sol)
            avg_outer = self.outer_iterations / max(self.nonlinear_iterations,1)

            msg = (f'It: {iter} '
                +f' dt: {ctrl.deltat:.1e}'
                +f' var:{var_tdens:.1e}'
                +f' nsym:{self.nonlinear_iterations:1d}'
                +f' avgouter: {avg_outer:.1f}')
            ctrl.print_info(msg, priority=1, color='green', modes=['main','log'])
            with sol.dat.vec_ro as sol_vec:
                tdens_vec = sol_vec.getSubVector(self.fems.tdens_is)
                ctrl.print_info(msg_bounds(tdens_vec,'tdens'),
                                 priority=1, color='black', modes=['main','log'])
                

            # save data    
            self.record_algorithm(sol, ctrl, iter)

            # check convergence
            if (var_tdens < ctrl.tol ):
                ierr_dmk = 0
                break

            # break if max iter is reached
            if (iter == ctrl.max_iter):
                ierr_dmk = 1
        
        return ierr_dmk
              
    def discrepancy(self, pot, tdens ):
        '''
        Measure the discrepancy between I(tdens) and the observed data.
        '''
        dis = self.confidence * ( self.img_observed - self.tdens2image(tdens))**2 * dx
        return dis
    
    def joule(self, pot, tdens):
        '''
        Joule dissepated energy functional
        '''
        if self.spaces == 'CR1DG0':        
            joule_fun = ( ( self.source - self.sink) * pot * dx 
                        - 0.5 * (tdens) / self.kappa**2 * dot(grad(pot), grad(pot)) * dx
            )
            #if self.Neumann is not None:
            #    n = FacetNormal(self.mesh)
            #    opt_pen += dot(self.Neumann,n) * pot * ds
                      
        elif self.spaces == 'DG0DG0':
            if self.DG0_cell2face == 'arithmetic_mean':
                facet_tdens = avg((tdens+self.min_tdens)/self.kappa**2)
            elif self.DG0_cell2face == 'harmonic_mean':
                left = (tdens('+')+self.min_tdens)/self.kappa('+')**2
                right =(tdens('-')+self.min_tdens)/self.kappa('-')**2
                facet_tdens = 2*left*right/(left+right)
            else:
                raise ValueError('Wrong cell2facet method. Passed '+self.DG0_cell2facet+
                                 '\n Only arithmetic_mean and harmonic_mean are implemented')
            
            joule_fun = ( (self.source - self.sink) * pot * dx
                        - 0.5 * facet_tdens * jump(pot)**2 / self.fems.delta_h * dS)
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
        
        # If tdens is piecewise constant,
        # we use finite difference or DG0 laplacian
        if self.fems.tdens_space.ufl_element().degree() == 0:
            if self.mesh.ufl_cell().is_simplex():
                # DG0 laplacian taken from 
                # https://www.firedrakeproject.org/demos/saddle_point_systems.py.html
                reg = 0.5 * self.fems.DG0_scaling * inner(jump(tdens, self.fems.normal), jump(tdens, self.fems.normal)) * dS
            else:
                reg = 0.5 * jump(tdens)**2 / self.fems.delta_h * dS
        else:
            raise NotImplementedError('Only piecewise constant tdens is implemented')
        reg = self.fems.Laplacian_Lagrangian(tdens, cell2face = self.DG0_cell2face)

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
        Lag = self.weights[1] * self.penalization(pot,tdens)
        if abs(self.weights[0]) > 1e-16:
            Lag += self.weights[0] * self.discrepancy(pot,tdens)
        if abs(self.weights[2]) > 1e-16:
            Lag += self.weights[2] * self.regularization(pot,tdens)
        return Lag
    
    def solve_pot_PDE(self, ctrl, sol):
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

        # solve the problem
        try:
            self.snes_solver.solve()
        except:
            pass
        ierr = self.snes_solver.snes.getConvergedReason()
        
        msg =  linalg.info_ksp(self.snes_solver.snes.ksp)
        ctrl.print_info(msg, priority=2, color='black', modes=['main','log'])
           
            
        if (ierr < 0):
            print(f' Failure in due to {SNESReasons[ierr]}')
        else:
            ierr = 0

        # get info of solver
        self.nonlinear_iterations = self.snes_solver.snes.getIterationNumber()
        self.outer_iterations = self.snes_solver.snes.getLinearSolveIterations()
        
        # move the pot solution in sol
        sol.sub(0).assign(self.pot_h)

        return ierr
    
    #@profile
    def tdens_mirror_descent(self, ctrl, sol):
        # Tdens is udpdate along the direction of the gradient
        # of the energy w.r.t. tdens multiply by tdens**(2-gamma)
        # 
        #
        pot, tdens = sol.subfunctions
        self.pot_h.assign(pot) 
        self.tdens_h.assign(tdens)
       
        time_start = time.time()
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
        self.lagrangian_fun = assemble(self.Lagrangian(self.pot_h,self.tdens_h))
        # The stop_annotating is required to avoid memory accumalation
        # It works but I don't know why.   
        with fire_adj.stop_annotating():
            self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.lagrangian_fun, fire_adj.Control(self.tdens_h))
            self.rhs_ode = self.lagrangian_fun_reduced.derivative()
        time_stop = time.time()


        # compute a scaling vector for the gradient
        update = Function(self.fems.tdens_space)
        scaling = Function(self.fems.tdens_space)
        
        if ctrl.gradient_scaling == 'dmk':            
            tdens_power = 2 - self.btp.gamma
        elif ctrl.gradient_scaling == 'mirror_descent':
            tdens_power = 1.0
        else:
            raise ValueError(f'Wrong scaling method {ctrl.scaling}')
        scaling.interpolate(self.tdens_h**tdens_power)

        with self.rhs_ode.dat.vec as rhs, scaling.dat.vec_ro as scaling_vec, update.dat.vec as d, self.tdens_h.dat.vec_ro as tdens_vec:
    
            #rhs *= scaling_vec
            rhs.scale(-1)
            self.fems.tdens_inv_mass_matrix.solve(rhs, d)           


            # scale the gradient w.r.t. tdens by tdens**tdens_power itself
            d *= scaling_vec
            step = ctrl.set_step(d,tdens_vec)
            ctrl.print_info(utilities.msg_bounds(d,'increment tdens')+f' dt={step:.2e}',
                            priority=2, color='black',modes=['main','log'])
            
            ctrl.deltat = step
            
            # update
            tdens_vec.axpy(step, d)
            ctrl.print_info(utilities.msg_bounds(tdens_vec,'tdens')+f' dt={step:.2e}',
                            priority=2, color='blue',modes=['main','log'])
            

        # threshold from below tdens
        utilities.threshold_from_below(self.tdens_h, 0)

        # assign new tdens to solution
        sol.sub(1).assign(self.tdens_h)
        
        # compute pot associated to new tdens
        ierr = self.solve_pot_PDE(ctrl, sol)        
        
        return ierr            
       
    def gfvar2tdens(self,gfvar):
        return gfvar**(2/self.btp.gamma)
        #return 2*atan(gfvar)
    def tdens2gfvar(self,tdens):
        return tdens**(self.btp.gamma/2)
        #return tan(tdens/2)
    
    def gfvar_gradient_descent(self, ctrl, sol):
        '''
        Update of using transformation tdens=gfvar**2 and gradient descent
        args:
            ctrl : Class with controls  (tol, max_iter, deltat, etc.)
            sol : Class with unkowns (pot,tdens, in this case)
        returns:
            ierr : control flag. It is 0 if everthing worked.
        Update of gfvar using gradient descent direction
        '''
        
        # convert tdens to gfvar
        pot, tdens = sol.subfunctions
        self.tdens_h.assign(tdens)
        gfvar = Function(self.fems.tdens_space)
        gfvar.interpolate(sqrt(tdens))

        # compute gradient of energy w.r.t. gfvar
        # see tdens_mirror_descent for more details on the implementation
        L = assemble(self.Lagrangian(pot, self.tdens_h))
        with fire_adj.stop_annotating():
            reduced_functional = fire_adj.ReducedFunctional(L, fire_adj.Control(self.tdens_h))
            self.rhs_ode = reduced_functional.derivative()
        
        # compute derivative of the transformation gfvar2tdens
        # TODO: this is really a bad workaround working
        # only for piecewise constant tdens
        if self.fem.tdens_space.ufl_element().degree() != 0:
            raise NotImplementedError('Only piecewise constant tdens is implemented')
        else:
            f = self.gfvar2tdens(gfvar)*dx
            scaling_cofun = assemble(derivative(f,gfvar))
            scaling_fun = Function(self.fems.tdens_space)
            with scaling_cofun.dat.vec_ro as scaling_cofun_vec, scaling_fun.dat.vec as scaling_fun_vec:
                self.fems.tdens_mass_matrix.solve(scaling_cofun_vec, scaling_fun_vec)
            # chain rule
            with self.rhs_ode.dat.vec as rhs, scaling_fun.dat.vec_ro as scaling_fun_vec, update.dat.vec as d:
                self.rhs_ode *= scaling_fun

        update = Function(self.fems.tdens_space)
        with self.rhs_ode.dat.vec as rhs, gfvar.dat.vec_ro as gfvar_vec, update.dat.vec as d:
            # scale by the inverse mass matrix
            self.fems.tdens_inv_mass_matrix.solve(rhs, d)
            
            # update
            step = ctrl.set_step(d)
            ctrl.deltat = step
            print(utilities.msg_bounds(d,'gfvar increment')+f' dt={step:.2e}')
            gfvar_vec.axpy(-step, d)

        # convert gfvar to tdens
        utilities.threshold_from_below(gfvar, 0)
        sol.sub(1).assign(self.gfvar2tdens(gfvar))

        # compute pot associated to new tdens
        ierr = self.solve_pot_PDE(ctrl, sol)
        print(ierr)
        return ierr
    
   
        
    def iterate(self, ctrl, sol):
        '''
        Args:
         ctrl : Class with controls  (tol, max_iter, deltat, etc.)
         solution updated from time t^k to t^{k+1} solution updated from time t^k to t^{k+1} sol : Class with unkowns (pot,tdens, in this case)

        Returns:
         ierr : control flag. It is 0 if everthing worked.

        '''
        if (ctrl.time_discretization_method == 'tdens_mirror_descent'):
            ierr = self.tdens_mirror_descent(ctrl, sol)
            return ierr
        
        elif (ctrl.time_discretization_method == 'gfvar_gradient_descent'):
            ierr = self.gfvar_gradient_descent(ctrl, sol)
            return ierr
        else:
            raise ValueError('value: self.ctrl.time_discretization_method not supported.\n',
                              f'Passed:{ctrl.time_discretization_method}')   
    
    def save_solution(self, sol, filename):            
        '''
        Save into a file the solution [pot,tdens] and velocity
        '''
        self.fems.save_solution(sol,filename)

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



