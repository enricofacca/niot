# import Solver 
#from rcis import Solver
from copy import deepcopy as cp
import gc
import sys


import warnings
warnings.filterwarnings("ignore")

import numpy as np
import scipy as sp
import scipy.sparse.linalg as splinalg
from scipy.linalg import norm 
import time as cputiming
import os
import petsctools

import petsc4py
import time 
#from memory_profiler import profile


from . conductivity2image import IdentityMap, HeatMap, PorousMediaMap, Barenblatt
from . import utilities
from . import optimal_transport as ot
from . import linear_algebra_utilities as linalg
from . import image2dat as i2d


# function operations
from firedrake import *
from firedrake.functionspace import DualSpace
from firedrake.__future__ import interpolate
import firedrake.adjoint as fire_adj


from progress.bar import FillingSquaresBar
#fire_adj.get_working_tape().progress_bar = FillingSquaresBar



from firedrake.tsfc_interface import TSFCKernel
from pyop2.global_kernel import GlobalKernel
from firedrake.petsc import PETSc
from firedrake.petsc import flatten_parameters

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

def Laplacian_facet_weight(mesh, mode = "center_distance"):
    '''
    Return a facet-based quantity that scales as h, the mesh typical length
    '''
    if mode == "center_distance":
        DG0 = FunctionSpace(mesh, 'DG', 0)
        if mesh.geometric_dimension() == 2:
            x,y = mesh.coordinates
            x_func = assemble(interpolate(x, DG0))
            y_func = assemble(interpolate(y, DG0))
            delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)
        
        elif mesh.geometric_dimension() == 3:
            x,y,z = mesh.coordinates
            x_func = assemble(interpolate(x, DG0))
            y_func = assemble(interpolate(y, DG0))
            z_func = assemble(interpolate(z, DG0))
            delta_h = sqrt(jump(x_func)**2 
                           + jump(y_func)**2 
                           + jump(z_func)**2)
    elif mode == "cell_over_face":
        delta_h = 1.0 / avg(FacetArea(mesh) / CellVolume(mesh))
    else:
        raise ValueError('mode must be - center_distance or face_over_cell')
    return delta_h


def h_size(mesh, mode = "CellDiameter"):
    if mode == "CellDiameter":
        return CellDiameter(mesh)
    elif mode == "cell_over_facet":
        return CellVolume(mesh) / FacetArea(mesh)
    else:
        raise ValueError('mode must be - CellDiameter or cell_over_facet')


def d_face_interior(mesh):
    if mesh.extruded:
        d_interior = dS_v + dS_h
    else:
        d_interior = dS
    return d_interior

def d_face_exterior(mesh):
    if mesh.extruded:
        d_exterior = ds_b + ds_t + ds_v
    else:
        d_exterior = ds
    return d_exterior

def simplex_DG0_scaling(mesh):
    # quantities for DG0 laplacian
    alpha = 4.0
    h = h_size(mesh, mode = "cell_over_facet")
    h_avg = (h('+') + h('-'))/2.0
    DG0_scaling = alpha(4.0)/h_avg

    return DG0_scaling
            


class SpaceDiscretization:
    '''
    Class containg fem discretization variables
    '''
    
    # include relevant citations
    petsctools.cite('FCP2021')
    def __init__(self, mesh, 
                 pot_space='CR', 
                 pot_deg=1, 
                 tdens_space='DG', 
                 tdens_deg=0, 
                 cell2face='harmonic_mean',
                 h_mode="CellDiameter",
                 ):
        #tdens_fem='DG0',pot_fem='P1'):
        '''
        Initialize FEM spaces used to discretized the problem
        '''  
        self.pot_fem = FiniteElement(pot_space, mesh.ufl_cell(), pot_deg)
        self.pot_space = FunctionSpace(mesh, self.pot_fem)
        self.pot_trial = TrialFunction(self.pot_space)
        self.pot_test = TestFunction(self.pot_space)

        #if ((pot_space=='DG') or (pot_space =="DQ") ) and (pot_deg == 0):
        self.delta_h = Laplacian_facet_weight(mesh, mode = "center_distance")#"face_over_cell")
        self.cell2face = cell2face
        # quantities for DG0 laplacian
        alpha = Constant(4.0)
        self.h = h_size(mesh, mode = h_mode)
        # self.h = h_size(mesh, mode = "face_over_cell")
        h_avg = (self.h('+') + self.h('-'))/2.0
        self.DG0_scaling = alpha/h_avg
        self.normal = FacetNormal(mesh)

        # For Tdens unknow, create fem, function space, test and trial
        # space
        self.tdens_fem = FiniteElement(tdens_space, mesh.ufl_cell(), tdens_deg)
        self.tdens_space = FunctionSpace(mesh, self.tdens_fem)
        self.tdens_trial = TrialFunction(self.tdens_space)
        self.tdens_test = TestFunction(self.tdens_space)


        # # velocity field
        # if pot_deg >= 1:
        #     self.velocity_space = VectorFunctionSpace(mesh, 'DG', 0)
        # else:
        #     # deg=0
        #     if space.mesh().ufl_cell().is_simplex():
        #         raise NotImplementedError('Only piecewise constant is implemented for simplicial meshes')
            
        #     if mesh.extruded:
        #         raiseself.velocity_space = FunctionSpace(mesh, 'DG', 0)
        #     else:
        #         if (mesh.cell_type().cellname() == 'quadrilateral'
        #                 or mesh.cell_type().cellname() == 'hexahedron'):
                    
        #         self.velocity_space = VectorFunctionSpace(mesh, 'RTCF', 0)

        

        
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
        # detect degree of function space
        degree = space.ufl_element().degree()
        
        # detect an extruded mesh is used
        if isinstance(degree, tuple):
            if all(d == 0 for d in degree):
                degree = 0
            else:
                raise ValueError('Only piecewise constant is implemented for extruded meshes')
        else:
            degree = degree

        d_internal_faces = d_face_interior(space.mesh())

        test = TestFunction(space)
        trial = TrialFunction(space)
        if degree == 0:
            # the weight need to be "projected to the facets"
            if weight is not None:
                facet_weight = self.cell2face_map(weight, cell2face)
            else:
                facet_weight = 1.0
            
            if space.mesh().ufl_cell().is_simplex():
                # if the mesh is simplicial, we use the DG0 laplacian taken from
                # https://www.firedrakeproject.org/demos/saddle_point_systems.py.html
                # Without scaling the scheme is not consistent.
                form = self.DG0_scaling * facet_weight * inner(jump(test, self.normal), jump(trial, self.normal)) * d_internal_faces
            else:
                form =  facet_weight * jump(test) * jump(trial) / self.delta_h * d_internal_faces
        elif degree == 1:
            if weight is None:
                weight = Constant(1.0)
            form = weight * inner(grad(test), grad(trial)) * dx
        else:
            raise NotImplementedError('piecewise constant, or linear tdens is implemented')
        return form
    
    
            


    def Laplacian_Lagrangian(self, 
                             u, 
                             weight=None, 
                             cell2face=None):
        """
        Return the Lagrangian of a weighted-Laplacian equation for u.
        The weight is a function of the mesh.
        If the weight is not passed, it is assumed to be 1.0.
        """
        
        if weight is None:
            weight = Constant(1.0)

        space = u.function_space()

        # detect degree of function space
        degree = space.ufl_element().degree()
        # an extruded mesh is used
        if isinstance(degree, tuple):
            if all(d == 0 for d in degree):
                degree = 0
            else:
                raise ValueError('Only piecewise constant is implemented for extruded meshes')
        else:
            degree = degree
            

        d_internal_faces = d_face_interior(space.mesh())

        if degree == 0:
            # the weight need to be "projected to the facets"
            if weight is not None:
                facet_weight = self.cell2face_map(weight, cell2face)
            else:
                facet_weight = 1.0
            
            if space.mesh().ufl_cell().is_simplex():
                # if the mesh is simplicial, we use the DG0 laplacian taken from
                # https://www.firedrakeproject.org/demos/saddle_point_systems.py.html
                # Without scaling the scheme is not consistent.
                L = 0.5 * self.DG0_scaling * facet_weight * inner(jump(u, self.normal), jump(u, self.normal)) * d_internal_faces
            else:
                L = 0.5 * facet_weight * jump(u)**2 / self.delta_h * d_internal_faces
        elif degree == 1:
            L = 0.5 * weight * inner(grad(u), grad(u)) * dx
        else:
            raise NotImplementedError('piecewise constant, or linear tdens is implemented')
        return L
    
    def apply_weak_Dirichlet_lhs(self, 
                             weak_Dirichlet, 
                             Laplacian_form,
                             penalty=1e2):
        """
        Apply weak Dirichlet boundary conditions to the form L.
        Args:
        weak_Dirichlet: list of tuples (marker_function, values_function, boundary_measure)
                        marker_function: A DG function that is 1.0 on the boundary
                        values_function: A function that contains the values of the Dirichlet boundary
                        boundary_measure: the measure of the boundary
        A_form: the bilinear form to which the boundary conditions are applied
        """
        # Dirichlet boundary conditions are imposed weakly, using a penalty method.
        # See:
        # https://bitbucket.org/fenics-project/dolfin/src/master/python/demo/undocumented/dg-poisson/demo_dg-poisson.py
        # https://fenicsproject.org/pub/course/lectures/2017-nordic-phdcourse/lecture_10_discontinuous_galerkin.pdf
        
        function_space = Laplacian_form.arguments()[0].function_space()
        test = TestFunction(function_space)
        trial = TrialFunction(function_space)

        for values_function, d_boundary_measure, marker in weak_Dirichlet:
            Laplacian_form += penalty / self.h * test * trial * marker * d_boundary_measure
        return Laplacian_form
            
    def apply_weak_Dirichlet_rhs(self,
                                 weak_Dirichlet,
                                 rhs_form,
                                 penalty=1e2):
            
        function_space = rhs_form.arguments()[0].function_space()
        test = TestFunction(function_space)
        for values_function, d_boundary_measure, marker in weak_Dirichlet:
            rhs_form += penalty / self.h * values_function * test * marker * d_boundary_measure
        return rhs_form

    


            
def set_step(increment,
             state, 
             deltat,
             type='adaptive', 
             lower_bound=1e-2, 
             upper_bound=0.5, 
             expansion=2,
             contraction=0.5,
             order_down=-0.8,
                order_up=0.8,
                ):
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
        
        r = increment / state
        _, r_min = r.min()
        _, r_max = r.max()

        #r_np = r.array
        if r_min < 0:
            #negative = np.where(r_np < 0)
            #hdown = (10**order_down-1) / r_np[negative]
            #down = np.min(hdown)
            down = (10**order_down-1) / r_min
        else:
            down = upper_bound
        
        if r_max > 0:
            #positive = np.where(r_np>0)
            #hup = (10**order_up - 1) / r_np[positive]
            #up = np.min(hup)
            up = (10**order_up - 1) / r_max
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
        'restart': False,
        # info 
        'verbose' : 0,
        'log_verbose': 2,
        'log_file': 'niot.log',
        'adjoint_verbose': 0,
        #'inpainting' : {
        'discrepancy_weight': 1.0,
        'discrepancy_norm': "l2",
        'discrepancy_dual_h1_sigma' : 1.0,
        'regularization_weight': 0.0,
        'penalization_weight': 1.0,
        # tdens to image mapping
        "use_adjoint" : False,
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
            'type' : 'tdens_mirror_descent_explicit',
            'tdens_mirror_descent_explicit' : {
                'gradient_scaling' : 'dmk',
                'deltat' : {
                    'type' : 'adaptive2',
                    'lower_bound' : 1e-2,
                    'upper_bound' : 0.5,
                    'expansion' : 2,
                    'contraction': 0.5,
                    'order_down': -0.7,
                    'order_up': 0.7,
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
    petsctools.cite('FCP2021')
    #@profile
    def __init__(self, btp, observed, 
                 confidence=1.0, 
                 spaces='DG0DG0',
                 cell2face='harmonic_mean',
                 setup=False,
                 ensemble_comm=None,
                 ):
        '''
        Initialize solver (spatial discretization)
        '''
        ###########################
        # SETUP FEM DISCRETIZATION
        ###########################
                          
        self.mesh = btp.mesh
        self.comm  = self.mesh.comm
        self.ensemble_comm = None
        if ensemble_comm is not None:
            self.ensemble_comm = ensemble_comm
        self.spaces = spaces
        self.cell2face = cell2face
        
        #mode = "CellDiameter"
        mode = "cell_over_facet"
        self.Dirichlet_penalty = 1e2
        if self.spaces == 'CR1DG0':
            self.fems = SpaceDiscretization(self.mesh,'CR', 1, 'DG', 0, cell2face, h_mode=mode)
        if self.spaces == 'CG1DG0':
            self.fems = SpaceDiscretization(self.mesh,'CG', 1, 'DG', 0, cell2face, h_mode=mode)
        elif self.spaces == 'DG0DG0':
            if self.mesh.ufl_cell().is_simplex():
                raise ValueError('DG0DG0 only implemented for cartesian grids')
            self.fems = SpaceDiscretization(self.mesh,'DG',0,'DG',0, cell2face, h_mode=mode)
        else:
            raise ValueError('Wrong spaces only (pot,tdens) in (CR1,DG0) or (DG0,DG0) implemented')
        self.ConstansSpace = FunctionSpace(self.mesh, 'R', 0)

        

        # initialize the solution
        self.sol = self.create_solution()
        self.sol_old = self.sol.copy(deepcopy=True)

        # The class BranchedTransportProblem
        # that contains the physical information of the problem
        # and the branched transot exponent gamma 
        self.btp = btp
        
        # set img to be inpainted
        self.img_observed = observed
    
        # confidence 
        self.confidence = Function(self.fems.tdens_space)
        assemble(interpolate(confidence,self.fems.tdens_space), tensor=self.confidence)
        self.confidence.rename('confidence','confidence')

        # weights

        self.discrepancy_weight =  Function(self.ConstansSpace, name="discrepancy_weight")
        self.penalization_weight =  Function(self.ConstansSpace, name="penalization_weight")
        self.regularization_weight = Function(self.ConstansSpace, name="regularization_weight")
        


        # init infos
        self.iteration = 0
        self.restart = 0
        self.outer_iterations = 0
        self.nonlinear_iterations = 0
        self.nonlinear_res = 0.0
        self.gradients_computed = False

        # set to initial state
        self.deltat = 0.0

        element = self.fems.tdens_space.ufl_element()
        self.gradient_D_P = Cofunction(DualSpace(self.mesh,element))
        self.rhs_ode = Cofunction(DualSpace(self.mesh,element))
        self.residuum = Cofunction(DualSpace(self.mesh,element))
        self.gradient_regularization = Cofunction(DualSpace(self.mesh,element))
        self.gradient_discrepancy = Cofunction(DualSpace(self.mesh,element))        
        self.gradient_penalization = Cofunction(DualSpace(self.mesh,element))
        self.gradient_regularization = Cofunction(DualSpace(self.mesh,element))
        self.gradient_lagrangian = Cofunction(DualSpace(self.mesh,element))

        if setup:
            self.setup()

    #@profile
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
            "snes_monitor": None,
            # krylov solver controls
            'ksp_type': 'minres',
            'pc_type': 'hypre',
            'ksp_atol': self.ctrl_get('constraint_tol'),#1e-16,
            'ksp_rtol': 1e-16,#self.ctrl_get('constraint_tol'),
            'ksp_dtol': 1e5,
            'ksp_max_it' : 1000,
            #'ksp_initial_guess_nonzero': True, 
            #'ksp_norm_type': 'unpreconditioned',
            #'ksp_monitor_true_residual' : None, 
        }
        if self.mesh.geometric_dimension() == 3:
            hypre_ctrl_3d = {
                            # tuning parameters for the multigrid
                            # https://mooseframework.inl.gov/releases/moose/2021-09-15/application_development/hypre.html
                            "pc_hypre_type": "boomeramg",
                            "pc_hypre_boomeramg_strong_threshold": 0.75,
                            "pc_hypre_boomeramg_max_iter": 1,
                            "pc_hypre_boomeramg_agg_nl": 3,
                            "pc_hypre_boomeramg_interp_type": "ext+i",  # "classic" or "ext+i"
                        }
            petsc_controls.update(hypre_ctrl_3d)
            
            

        if self.ctrl_get('verbose') >= 3:
            petsc_controls['ksp_monitor_true_residual'] = None
        self.setup_pot_solver(petsc_controls)

        log_verbose = self.ctrl_get('log_verbose')
        if log_verbose > 0:
            log_file = self.ctrl_get('log_file')
            try:
                os.remove(log_file)
            except OSError:
                pass
            if hasattr(self, 'log_viewer'):
                self.log_viewer.destroy()
            self.log_viewer = PETSc.Viewer().createASCII(log_file, 'w', comm=self.comm)
    
        # we need to initialize the increment solver
        self.shift_semi_implicit = Function(self.ConstansSpace)
        self.shift_semi_implicit.assign(0.1)
        
        test = TestFunction(self.fems.tdens_space)  
        trial = TrialFunction(self.fems.tdens_space)
        self.increment_form =  inner(test, trial)*dx
        self.increment_form += self.shift_semi_implicit * self.fems.Laplacian_form(self.fems.tdens_space)
        self.increment_h = Function(self.fems.tdens_space)
        self.increment_h.rename('increment_h')

        # # function for h^{-1} norm
        test = TestFunction(self.fems.pot_space)  
        trial = TrialFunction(self.fems.pot_space)
        self.pot_dual_h1 = Function(self.fems.pot_space)
        self.pot_dual_h1.rename('pot_dual_h1')

        self.dual_h1_sigma =  Function(self.ConstansSpace, name="discrepancy_dual_h1_sigma")
        self.dual_h1_sigma.assign(self.ctrl_get('discrepancy_dual_h1_sigma'))
        PETSc.Sys.Print("sigma = scaling of l^2 term", self.ctrl_get('discrepancy_dual_h1_sigma'))
        
        self.dual_h1_form = self.dual_h1_sigma * self.fems.Laplacian_form(self.fems.pot_space, self.confidence, cell2face="arithmetic_mean")
        test = TestFunction(self.fems.pot_space)
        trial = TrialFunction(self.fems.pot_space)
        self.dual_h1_form += test * trial * self.confidence * dx 
        self.dual_h1_rhs = ( self.image_h - self.img_observed ) * test * dx
        

        #t = assemble(self.dual_h1_Lagrangian)
        
        # Define the problem
        #self.h1_dual_PDE = derivative(self.dual_h1_Lagrangian, self.pot_dual_h1)
        self.dual_h1_problem = LinearVariationalProblem(self.dual_h1_form, self.dual_h1_rhs, self.pot_dual_h1)
        #self.h1_dual_pde_problem = NonlinearVariationalProblem(self.h1_dual_PDE, self.pot_dual_h1)        
        
        
        # Define solver
        solver_parameters={
                #'snes_type': 'ksponly',
                'ksp_type': 'minres',
                'ksp_rtol': 1e-10,
                'ksp_atol': 1e-8,
                'ksp_max_it': 500,
                'pc_type': 'hypre',
                #'snes_monitor': None,
                #'snes_linesearch_monitor': None,
                #'ksp_monitor': None,
                }
        if self.ctrl_get('adjoint_verbose') >= 2:
            solver_parameters.update({"snes_monitor": None})
        if self.ctrl_get('adjoint_verbose') >= 3:
            solver_parameters.update({"ksp_monitor": None})
        
        if self.mesh.geometric_dimension() == 3:
            if self.mesh.ufl_cell().is_simplex():
                hypre_ctrl_3d = {
                        # tuning parameters for the multigrid
                        # https://mooseframework.inl.gov/releases/moose/2021-09-15/application_development/hypre.html
                        "pc_hypre_type": "boomeramg",
                        "pc_hypre_boomeramg_strong_threshold": 0.65,
                        "pc_hypre_boomeramg_max_iter": 1,
                        "pc_hypre_boomeramg_agg_nl": 0,
                        "pc_hypre_boomeramg_interp_type": "ext+i",  # "classic" or "ext+i"
                    }
            else:   
                hypre_ctrl_3d = {
                        # tuning parameters for the multigrid
                        # https://mooseframework.inl.gov/releases/moose/2021-09-15/application_development/hypre.html
                        "pc_hypre_type": "boomeramg",
                        "pc_hypre_boomeramg_strong_threshold": 0.75,
                        "pc_hypre_boomeramg_max_iter": 1,
                        "pc_hypre_boomeramg_agg_nl": 3,
                        "pc_hypre_boomeramg_interp_type": "ext+i",  # "classic" or "ext+i"
                    }
            solver_parameters.update(hypre_ctrl_3d)
        self.dual_h1_solver = LinearVariationalSolver(self.dual_h1_problem,
                                                        solver_parameters=solver_parameters,
                                                        options_prefix='dual_h1_solver_')
        

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

        num_cells = 0
        num_vertices = 0
        num_facets = 0
        nproc = PETSc.COMM_WORLD.getSize()
        for i in range(nproc):
            if i == PETSc.COMM_WORLD.getRank():
                if isinstance(self.mesh, ExtrudedMeshTopology):
                    num_cells = self.mesh.num_cells() * (self.mesh.layers-1)
                    num_vertices = self.mesh.num_vertices() * self.mesh.layers
                    num_facets = ( self.mesh.num_cells() * (self.mesh.layers) # horizontal facets
                                + self.mesh.num_facets() * (self.mesh.layers-1) ) # vertical facets
                else:
                    num_cells = self.mesh.num_cells()
                    num_vertices = self.mesh.num_vertices()
                    num_facets = self.mesh.num_facets()
            PETSc.COMM_WORLD.barrier()


        self.print_info(f'Cells: {num_cells}'
                        + f' Nodes: {num_vertices}'
                        + f' Facets: {num_facets}',
                        priority=2, where=['stdout','log'])
        
        # open log file
        tdens2image = self.ctrl_get('tdens2image')
        
        max_iter = self.ctrl_get('max_iter')
        wd = self.ctrl_get('discrepancy_weight')

        # set up the optimization algorithm
        use_adjoint = self.ctrl_get('use_adjoint')   


        # set up gradient computation
        dw = self.ctrl_get('discrepancy_weight')
        self.discrepancy_weight.assign(dw)
        

        # Discrepancy 
        # self.discrepancy_form = self.discrepancy_weight * self.discrepancy(self.pot_h,self.tdens_h)
        # if dw > 0:
            
        #     if use_adjoint:
        #         # The following is required to keep track of the 
        #         # adjoint computation, like when the map from tdens to image is 
        #         # defined as the solution of a PDE (for example the poruous media map).
        #         self.adj_discrepancy_fun = assemble(self.discrepancy_form)
        #         self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.adj_discrepancy_fun, fire_adj.Control(self.tdens_h))
        #     else:
        #         # Simple derivative computation
        #         # It uses less memory, but it requires the functional
        #         # as combination of operations manegable by automatic differiantion.
        #         #self.gradient_discrepancy = assemble(derivative(self.lagrangian_fun, self.tdens_h))
        #         self.gradient_discrepancy_form = derivative(self.discrepancy_form, 
        #                                                     self.tdens_h,
        #                                                     coefficient_derivatives=self.tdens2image_map.cd)
                
        # # Penalization
        # pw = self.ctrl_get('penalization_weight')
        # self.penalization_weight.assign(pw)
        # self.penalization_form = self.penalization_weight * self.penalization(self.pot_h,self.tdens_h)
        # if abs(pw) > 1e-16:
        #     if use_adjoint:
        #         self.adj_penalization_fun = assemble(self.penalization_form)
        #         self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.adj_penalization_fun, fire_adj.Control(self.tdens_h))
        #     else:
        #         self.gradient_penalization_form = derivative(self.penalization_form, self.tdens_h)


        

    def setup_tdensimage(self):
        """
        Method initializing the map from image to tdens
        """
        self.image_h = Function(self.fems.tdens_space)
        self.image_h.rename('image_h') # used by tdens2image map

        #self.reconstruction = Function(self.fems.tdens_space)
        #self.reconstruction.rename('reconstruction') 


        self.tdens4transform = Function(self.fems.tdens_space)
        self.tdens4transform.rename('tdens4transform') # used by tdens2image map
        
        tdens2image = self.ctrl_get(['tdens2image', 'type'])
        scaling = self.ctrl_get(['tdens2image', 'scaling'])
        
        
        
        #import firedrake.adjoint as fire_adj
        #fire_adj.continue_annotation()


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
            PETSc.Sys.Print(f'Using PM map for tdens2image mapping')
            try:
                sigma = self.ctrl_get(['tdens2image', 'pm','sigma'])
                exponent_m = self.ctrl_get(['tdens2image', 'pm','exponent_m'])
                scaling = self.ctrl_get(['tdens2image', 'scaling'])
                label_pm = f'pm_{exponent_m:.1f}_{sigma:.2e}'
            except:
                cond_zero = self.ctrl_get(['tdens2image', 'pm','cond_zero'])
                PETSc.Sys.Print(f'Using cond_zero {cond_zero} in pm tdens2image map')
                exponent_p = self.ctrl_get(['tdens2image', 'pm','exponent_p'])
                PETSc.Sys.Print(f'Using exponent_p {exponent_p} in pm tdens2image map')
                scaling = self.ctrl_get(['tdens2image', 'pm','scaling'])
                try:
                    correction = self.ctrl_get(['tdens2image', 'pm','correction'])
                except:
                    correction = 1.0
                PETSc.Sys.Print(f'Using correction factor {correction} in pm tdens2image map')
                dim = self.mesh.geometric_dimension() 

                dim = self.mesh.geometric_dimension()
                if exponent_p < dim-1:
                    raise ValueError('p<d')
                exponent_m = (2 + exponent_p - (dim - 1) ) / (exponent_p - (dim - 1 ) )

                Bar = Barenblatt(exponent_m,dim-1)
            
                # find the time to get 
                # M = M_0 * (r(\sigma))**p 
                # sigma = (cond_zero**(-1/exponent_p) * K_md ** (-1/2) * B **(1/2))**(1/beta)
                sigma = Bar.sigma(cond_zero,exponent_p) * correction
            
                label_pm = f'pm_{exponent_p:.1f}_{cond_zero:.2e}_{scaling:.2e}'


            if self.mesh.ufl_cell().is_simplex():
                mode = "exponential"
                solver_parameters={
                    "snes_type": 'newtonls',
                    'snes_linesearch_type':'bt',
                    #"snes_linesearch_type": "basic",
                    #"snes_linesearch_maxstep": 0.1,
                    #"snes_linesearch_damping": 1.0,
                    #"snes_linesearch_monitor": None,
                    #"snes_linesearch_maxlambda" : 0.1,
                    #"snes_linesearch_max_it": 4,
                    'snes_rtol': 1e-8,
                    'snes_atol': 1e-16,
                    'snes_stol': 1e-10,
                    'snes_max_it': 100,
                    # inexact Newton with Eisenstat-Walker
                    'snes_ksp_ew': None,
                    'snes_ksp_ew_rtol0': 1e-2,
                    'snes_ksp_ew_rtolmax': 1e-6,
                    'ksp_type': 'gmres',
                    #'ksp_rtol': 1e-6,
                    #'ksp_atol': 1e-12,
                    'ksp_max_it': 500,
                    'pc_type': 'hypre',
                    'snes_monitor': None,
                    'snes_linesearch_monitor': None,
                    'ksp_monitor': None,
                    }
                if self.mesh.geometric_dimension() == 3:
                    hypre_ctrl_3d = {
                                # tuning parameters for the multigrid
                                # https://mooseframework.inl.gov/releases/moose/2021-09-15/application_development/hypre.html
                                "pc_hypre_type": "boomeramg",
                                "pc_hypre_boomeramg_strong_threshold": 0.75,
                                "pc_hypre_boomeramg_max_iter": 1,
                                "pc_hypre_boomeramg_agg_nl": 3,
                                "pc_hypre_boomeramg_interp_type": "ext+i",  # "classic" or "ext+i"
                            }
                    solver_parameters.update(hypre_ctrl_3d)


            else:
                mode = "direct"
                solver_parameters={
                    'snes_type': 'newtonls',
                    'snes_rtol': 1e-6,
                    'snes_atol': 1e-6,
                    'snes_stol': 1e-6,
                    'snes_max_it': 100,
                    'snes_linesearch_type':'bt',
                    # inexact Newton with Eisenstat-Walker
                    #'snes_ksp_ew': True,
                    #'snes_ksp_ew_rtol0': 1e-2,
                    #'snes_ksp_ew_rtolmax': 1e-6,
                    'ksp_type': 'gmres',
                    'ksp_rtol': 1e-6,
                    'ksp_atol': 1e-6,
                    'ksp_max_it': 500,
                    'pc_type': 'hypre',
                    'snes_monitor': None,
                    'snes_linesearch_monitor': None,
                    'ksp_monitor': None,
                    }

                if self.mesh.geometric_dimension() == 3:
                    hypre_ctrl_3d = {
                                # tuning parameters for the multigrid
                                # https://mooseframework.inl.gov/releases/moose/2021-09-15/application_development/hypre.html
                                "pc_hypre_type": "boomeramg",
                                "pc_hypre_boomeramg_strong_threshold": 0.75,
                                "pc_hypre_boomeramg_max_iter": 1,
                                "pc_hypre_boomeramg_agg_nl": 3,
                                "pc_hypre_boomeramg_interp_type": "ext+i",  # "classic" or "ext+i"
                            }
                    solver_parameters.update(hypre_ctrl_3d)

            self.tdens2image_map = PorousMediaMap(
                self.fems.pot_space,
                scaling=self.ctrl_get(['tdens2image', 'pm','scaling']),
                sigma=sigma,
                exponent_m=exponent_m,
                nsteps=self.ctrl_get(['tdens2image', 'pm','nsteps']),
                dt0=self.ctrl_get(['tdens2image', 'pm','dt0']),
                solver_parameters=solver_parameters,
                name=label_pm,
                mode=mode)
            self.tdens2image = lambda x: self.tdens2image_map(x)

            try:
                reuse_images = self.ctrl_get(['tdens2image', 'pm','store_images']) == 1
                self.tdens2image_map.use_stored_images_as_initial_guess = reuse_images
                self.tdens2image_map.store_images = reuse_images
            except:
                self.tdens2image_map.use_stored_images_as_initial_guess = False
                self.tdens2image_map.store_images = False
            """ Store the images at each time step"""

            
            """ Boolean for activation of using store intermediate images as initial guess"""
        


        else:
            raise ValueError(f"Map tdens2iamge not supported {tdens2image=}\n"
                             +"Only identity, heat, pm,  are implemented")

        
    def setup_pot_solver(self, petsc_controls):
        # chaced functions
        self.pot_h = Function(self.fems.pot_space) # used by pot_solver
        self.pot_h.rename('pot_h')
        self.pot_h.assign(0.0)


        self.rhs_norm = assemble((self.btp.source - self.btp.sink)**2 * dx)

        # the minus sign is to get -\div(\tdens \grad \pot)-f = 0
        self.pot_PDE = derivative(self.joule(self.pot_h,self.tdens_h/self.rhs_norm),self.pot_h)
        self.weighted_Laplacian = derivative(-self.pot_PDE,self.pot_h)

        # the forcing term
        self.rhs = (self.btp.source - self.btp.sink) /self.rhs_norm * self.fems.pot_test * dx
        
        
        # Set Weighted Laplacian
        #self.weighted_Laplacian = self.fems.Laplacian_form(self.fems.pot_space, weight=self.tdens_h, cell2face=self.fems.cell2face)
        
        
        # 
        # Boundary conditions
        # 
        penalty = self.Dirichlet_penalty
        if self.btp.weak_Dirichlet is not None:
            self.weighted_Laplacian = self.fems.apply_weak_Dirichlet_lhs(
                self.btp.weak_Dirichlet, self.weighted_Laplacian, penalty=penalty)
            self.rhs = self.fems.apply_weak_Dirichlet_rhs(
                self.btp.weak_Dirichlet, self.rhs, penalty=penalty)
        
        
        if self.btp.Dirichlet is not None:
            pot_bcs = []
            for bc in self.btp.Dirichlet:
                pot_bcs.append(DirichletBC(self.fems.pot_space, bc[1], bc[0]))
        else:
            pot_bcs = None

        # setup the nullspace
        if self.btp.Dirichlet is None and self.btp.weak_Dirichlet is None:
            nullspace = VectorSpaceBasis(constant=True,comm=self.comm)
        else:
            nullspace = None


        #
        # Setup Linear Variational Problem
        #
        min_tdens = self.ctrl_get('min_tdens')
        relax_preconditioner = False
        if relax_preconditioner:
            self.weighted_Laplacian_relaxed = self.weighted_Laplacian + 10*min_tdens * self.fems.Laplacian_form(self.fems.pot_space)
        else:
            self.weighted_Laplacian_relaxed = None


        # setup the linear variational problem
        self.u_prob = LinearVariationalProblem(self.weighted_Laplacian, # bilinear form
                                               self.rhs, # linear form
                                               self.pot_h, # solution
                                               aP = self.weighted_Laplacian_relaxed, # preconditioner form
                                               bcs = pot_bcs # boundary conditions
                                               ) 

        
            
        
        context = {} # use this to pass information to the solver
        self.pot_solver = LinearVariationalSolver(self.u_prob,
                                                solver_parameters = petsc_controls,
                                                nullspace = nullspace,
                                                appctx = context,
                                                options_prefix = 'pot_solver_')
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


               
    def print_info(self, msg, priority=0, where=['stdout'], color='black', verbose=None):
        '''
        Print messagge to stdout and to log 
        file according to priority passed
        '''
        for mode in where:
            if mode=='stdout':
                if verbose is None:
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
                    self.log_viewer.printfASCII('   '*(priority-1)+msg+'\n')

    

    def compute_residuum(self, sol):
        """
        Return the residual of the minimization problem w.r.t to tdens of gfvar.
        We slip the computation of each component since each component may give us usefull information.
        """
        pot, tdens = sol.subfunctions

        self.tdens_h.assign(tdens)
        self.pot_h.assign(pot)
        
        
        method = self.ctrl_get(['dmk','type'])

        use_adjoint = self.ctrl_get("use_adjoint")

        


        if "tdens" in method:
            dw = self.ctrl_get('discrepancy_weight')
            pw = self.ctrl_get('penalization_weight')
            rw = self.ctrl_get('regularization_weight')

            
            self.discrepancy_weight.assign(dw)
            self.penalization_weight.assign(pw)
            self.regularization_weight.assign(rw)

            # Discrepancy 
            # self.discrepancy_form = self.discrepancy_weight * self.discrepancy(self.pot_h,self.tdens_h)
            # if dw > 0:
                
            #     if use_adjoint:
            #         # The following is required to keep track of the 
            #         # adjoint computation, like when the map from tdens to image is 
            #         # defined as the solution of a PDE (for example the poruous media map).
            #         fire_adj.continue_annotation()
            #         self.adj_discrepancy_fun = assemble(self.discrepancy_form)
            #         self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.adj_discrepancy_fun, fire_adj.Control(self.tdens_h))
            #         fire_adj.stop_annotation()
            #     else:
            #         # Simple derivative computation
            #         # It uses less memory, but it requires the functional
            #         # as combination of operations manegable by automatic differiantion.
            #         #self.gradient_discrepancy = assemble(derivative(self.lagrangian_fun, self.tdens_h))
            #         self.gradient_discrepancy_form = derivative(self.discrepancy_form, 
            #                                                     self.tdens_h,
            #                                                     coefficient_derivatives=self.tdens2image_map.cd)
                    
            # # Penalization
            # pw = self.ctrl_get('penalization_weight')
            # self.penalization_weight.assign(pw)
            # self.penalization_form = self.penalization_weight * self.penalization(self.pot_h,self.tdens_h)
            # if abs(pw) > 1e-16:
            #     if use_adjoint:
            #         self.adj_penalization_fun = assemble(self.penalization_form)
            #         self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.adj_penalization_fun, fire_adj.Control(self.tdens_h))
            #     else:
            #         self.gradient_penalization_form = derivative(self.penalization_form, self.tdens_h)

                



            # Discrepancy 
            if dw > 0:
                
                if use_adjoint :
                    adjoint_verbose = self.ctrl_get('adjoint_verbose')
                    if not hasattr(self, "adj_discrepancy_fun_reduced"):
                        # The following is required to keep track of the 
                        # adjoint computation, like when the map from tdens to image is 
                        # defined as the solution of a PDE (for example the poruous media map).
                        fire_adj.continue_annotation()
                        self.print_info(
                            msg="Start annotation",
                            priority=1, 
                            where=['stdout','log'],
                            verbose=adjoint_verbose
                            )
                        self.discrepancy_form = self.discrepancy_weight * self.discrepancy(self.pot_h,self.tdens_h)
                        self.adj_discrepancy_fun = assemble(self.discrepancy_form)
                        
                        self.print_info(
                            msg=f"computed discrepancy form {self.adj_discrepancy_fun:.2e}",
                            priority=1, 
                            where=['stdout','log'],
                            verbose=adjoint_verbose
                            )
                        self.print_info(
                            msg="Compute reduced",
                            priority=1, 
                            where=['stdout','log'],
                            verbose=adjoint_verbose
                            )
                        self.adj_discrepancy_fun_reduced = fire_adj.ReducedFunctional(self.adj_discrepancy_fun, fire_adj.Control(self.tdens_h))
                        self.print_info(
                            msg="DISCR",
                            priority=1, 
                            where=['stdout','log'],
                            verbose=adjoint_verbose
                            )
                        self.print_info(
                            msg="End annotation. Reduced functional is defined",
                            priority=1, 
                            where=['stdout','log'],
                            verbose=adjoint_verbose
                            )
                        fire_adj.pause_annotation()

                        #tape = fire_adj.get_working_tape()
                        #tape.visualise("tape_discrepancy.pdf")
                    
                    else:
                        self.adj_discrepancy_fun = self.adj_discrepancy_fun_reduced(self.tdens_h)
                        self.print_info(
                            msg="Compute reduced",
                            priority=1, 
                            where=['stdout','log'],
                            verbose=adjoint_verbose
                            )
                        
                        
                    # the following is required since the ouptut of the adjoint is stored as function
                    # while is a co-function (is integrated over the mesh)
                    gradient_fun = self.adj_discrepancy_fun_reduced.derivative()

                    with gradient_fun.dat.vec_ro as gD, self.gradient_discrepancy.dat.vec as conf_vec:
                        #gD.copy(conf_vec)
                        self.fems.tdens_mass_matrix.mult(gD, conf_vec)
                    
                    self.print_info(
                        msg="computed gradient",
                        priority=1, 
                        where=['stdout','log'],
                        verbose=adjoint_verbose
                        )
                    
                    
                    #tape = fire_adj.get_working_tape()
                    #tape.clear_tape()
                else:
                    map_type = self.ctrl_get(["tdens2image", "type"])
                    if map_type == 'identity':
                        # set the discrepancy term and assembly
                        self.discrepancy_form = self.discrepancy_weight * self.discrepancy(self.pot_h,self.tdens_h)
                        
                        
                        
                        self.adj_discrepancy_fun = assemble(self.discrepancy_form)
                        
                        # accordinf to the norm use we can simplfy the computaion 
                        # of the gradient
                        discrepancy_norm = self.ctrl_get('discrepancy_norm')
                        if discrepancy_norm == 'l2':
                            self.gradient_discrepancy_form = derivative(self.discrepancy_form, 
                                                                 self.tdens_h,
                                                                 coefficient_derivatives=self.tdens2image_map.cd)
                        elif discrepancy_norm == 'dual_h1':
                            
                            # discrepancy form is 
                            # 
                            # int (I(tdens)-I_obs) * dual_pot(tdens)
                            # 
                            # But the subdifferential is just dual_pot(tdens) so we can compute
                            #           
                            self.gradient_discrepancy_form = ( 2.0 # correction 
                                                              * self.discrepancy_weight # scaling factor (confidence goes in the solver)
                                                              * self.pot_dual_h1 
                                                              * self.tdens2image_map.scaling # include the derivate of map
                                                              * self.fems.tdens_test * dx) #defining the form
                        else:
                            raise ValueError(f"Only l2 and dual_h1 implemented")
                    else:
                        raise ValueError(f"Not adjoint works only for identity map")
                

                    # Simple derivative computation
                    # It uses less memory, but it requires the functional
                    # as combination of operations manegable by automatic differiantion.
                    self.gradient_discrepancy = assemble(self.gradient_discrepancy_form)



                # print the gradient discrepancy
                with self.gradient_discrepancy.dat.vec_ro as gD:
                    msg = utilities.msg_bounds(gD,'grad discrepancy   ')
                    self.print_info(
                        msg=msg,
                        priority=1, 
                        where=['stdout','log']
                        )
            else:
                self.gradient_discrepancy.assign(0.0)

            #with self.reconstruction.dat.vec as rec_vec, self.image_h.dat.vec as img_vec:
            #    # print bounds
            #    PETSc.Sys.Print(utilities.msg_bounds(img_vec,'IMG'))
            #    PETSc.Sys.Print(utilities.msg_bounds(rec_vec,'REC'))

            # Penalization
            if pw > 0:
                self.penalization_form = self.penalization_weight * self.penalization(self.pot_h,self.tdens_h)
                # no need to use adjoint here, since the penalization is expressed as pure firedrake functions
                self.gradient_penalization_form = derivative(self.penalization_form, self.tdens_h)
                self.gradient_penalization = assemble(self.gradient_penalization_form)
                
                with self.gradient_penalization.dat.vec_ro as gP:
                    msg = utilities.msg_bounds(gP,f'grad penalty')
                    self.print_info(
                        msg=msg,
                        priority=1, 
                        where=['stdout','log']
                        )
            else:
                self.gradient_penalization.assign(0.0)

            if rw > 0:
                self.regularization_form = self.regularization_weight * self.regularization(pot,self.tdens_h)
                
                if use_adjoint: 
                    self.lagrangian_fun = assemble(self.regularization_form)
                    self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.lagrangian_fun, fire_adj.Control(self.tdens_h))
                    self.gradient_regularization = self.lagrangian_fun_reduced.derivative()
                else:
                    self.gradient_regularization = assemble(derivative(self.regularization_form,self.tdens_h))
            else:
                self.gradient_regularization.assign(0.0)

            self.gradient_D_P.assign(self.gradient_discrepancy + self.gradient_penalization)
            self.gradient_lagrangian.assign(self.gradient_discrepancy + self.gradient_penalization + self.gradient_regularization)
            
            #d = self.fems.tdens_mass_matrix.createVecLeft()
            self.residuum.assign(self.gradient_lagrangian)
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
                self.discrepancy_form = dw * self.discrepancy(pot,self.tdens_of_gfvar(gfvar))
                if use_adjoint:
                    lagrangian_fun = assemble(self.discrepancy_form)
                    lagrangian_fun_reduced = fire_adj.ReducedFunctional(lagrangian_fun, fire_adj.Control(gfvar))
                    self.gradient_discrepancy = lagrangian_fun_reduced.derivative()
                else:
                    self.gradient_discrepancy = assemble(derivative(self.discrepancy_form, gfvar))
                
                
                
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
                self.penalization_form = pw * self.penalization(pot,self.tdens_of_gfvar(gfvar))
                if use_adjoint:
                    lagrangian_fun = assemble(self.penalization_form)
                    lagrangian_fun_reduced = fire_adj.ReducedFunctional(lagrangina_fun, fire_adj.Control(gfvar))
                    self.gradient_penalization = lagrangian_fun_reduced.derivative()
                else:
                    self.gradient_penalization = assemble(derivative(self.penalization_form, gfvar))

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
                self.regularization_form  = rw * self.regularization(pot, gfvar)
                if use_adjoint:
                    lagrangian_fun = assemble(self.regularization_form )
                    lagrangian_fun_reduced = fire_adj.ReducedFunctional(lagrangian_fun, fire_adj.Control(gfvar))
                    self.gradient_regularization = lagrangian_fun_reduced.derivative()
                else:
                    self.gradient_regularization = assemble(derivative(self.regularization_form, gfvar))
                
                
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
            self.gradient_lagrangian.assign(self.gradient_discrepancy + self.gradient_penalization + self.gradient_regularization)
            self.gradients_computed = True

            self.residuum.assign(self.self.gradient_lagrangian)

        else:
            raise ValueError(f'Wrong optimization type {method=}')




    def create_solution(self):
        """
        Intialize solution
        """
        sol = Function(self.fems.pot_tdens_space,name=['pot','tdens'])
        sol.sub(0).assign(0.0)
        sol.sub(1).assign(1.0)

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
        if self.fems.pot_space.ufl_element().degree() == 1:
            assemble(interpolate(- tdens * grad(pot), DG0_vec), tensor = vel)
        else:
            if self.mesh.extruded:
                #RT1 =  FunctionSpace(self.mesh, "RTC",1)
                # RT1 element on a prism
                W0_h = FiniteElement("RTCF", quadrilateral, 1)
                W0_v = FiniteElement("DG", interval, 0)
                W0 = HDivElement(TensorProductElement(W0_h, W0_v))
                W1_h = FiniteElement("DG", quadrilateral, 0)
                W1_v = FiniteElement("CG", interval, 1)
                W1 = HDivElement(TensorProductElement(W1_h, W1_v))
                W_elt = W0 + W1
                RT1 = FunctionSpace(self.mesh, W_elt)
            else:
                RT1 = FunctionSpace(self.mesh, "Raviart-Thomas", 1)
            cond = self.fems.cell2face_map(tdens_h)
            gradpot = jump(pot) / self.fems.delta_h
            vel_RT1 = Function(RT1,name="flux")
            test = TestFunction(RT1)
            d_internal_faces = d_face_interior(self.mesh)
            rhs_form = cond * gradpot * div(test) * d_internal_faces
            mass_form = test * trial * dx 
            # setup the linear variational problem
            inter_prob = LinearVariationalProblem(, # bilinear form
                                               rhs_form, # linear form
                                               vel_RT1, # solution
                                               ) 
            petsc_controls = {
                "ksp_type": "minres",
                "pc_type": "hypre"}
            inter_solver = LinearVariationalSolver(inter_prob,
                                                solver_parameters = petsc_controls,
                                                options_prefix = 'inter_solver_')
            inter_solver.solve()
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
        use_adjoint = self.ctrl_get("use_adjoint")
        #if use_adjoint:
        #tape = fire_adj.get_working_tape()
    
        #self.compute_residuum(self.sol)
        if not self.ctrl_get('restart'):
            # Initialize the parameter-dependent solvers
            #self.setup()

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
            self.iteration = 0


        map_type = self.ctrl_get(["tdens2image", "type"])
        if map_type == 'pm':
            # if we want to use the intermidate images as initial guess we first compute the
            # images so we can record their assignment during the setup of the reduced functional
            if self.tdens2image_map.use_stored_images_as_initial_guess and not self.tdens2image_map.stored_images:
                self.print_info(
                    msg=f'Computing initial images for porous media map', 
                    priority=1, 
                    where=['stdout','log'], 
                    color='green')
                self.tdens2image_map.use_stored_images_as_initial_guess = False
                self.image_h = self.tdens2image_map(self.tdens_h)
                self.tdens2image_map.use_stored_images_as_initial_guess = True

            
        # udpack main controls and start main loop
        max_iter = self.ctrl_get('max_iter')
        
        


        ierr_dmk = 0
        self.local_iteration = 0
        while ierr_dmk == 0 and self.local_iteration < max_iter:
            # update with restarts
            tic = time.time()
            msg = f"\nIt: {self.iteration+1} method {self.ctrl_get(['dmk','type'])}"
            self.print_info(msg, priority=2, where=['stdout','log'], color='green')
    
            ierr = self.iterate(self.sol)
            # clean memory every 10 iterations
            update_time = time.time() -tic

            if self.local_iteration%2 == 0:
            #     self.print_info("cleaning tape", priority=2, where=['stdout','log'], color='green')
            #     #if use_adjoint :
            #         # Clear tape is required to avoid memory accumalation
            #         # It works but I don't know why
            #         # see also https://github.com/firedrakeproject/firedrake/issues/3133
            #     tape.clear_tape()
                        
            #     # other clean up taken from 
            #     # https://github.com/LLNL/pyMMAopt/commit/e2f83bd932207a8adbd60ae793b3e5a3058daecf
            #     #TSFCKernel._cache.clear()
            #     #GlobalKernel._cache.clear()
                gc.collect()
                petsc4py.PETSc.garbage_cleanup(self.mesh._comm)
                petsc4py.PETSc.garbage_cleanup(self.mesh.comm)

            if (ierr != 0):
                ierr_dmk = ierr
                self.print_info(f'{ierr=}')
                break

            # study state of convergence
            self.local_iteration += 1
            self.iteration += 1

            if self.local_iteration == max_iter:
                ierr_dmk = 1
            
            # compute residuum
            self.compute_residuum(self.sol)
            with self.residuum.dat.vec_ro as res_vec:
                residual_opt = res_vec.norm(PETSc.NormType.NORM_1)
            
            avg_outer = self.outer_iterations / max(self.nonlinear_iterations,1)

            msg = (f'It: {self.iteration} '
                +f' dt: {self.deltat:.1e}'
                +f' var:{residual_opt:.1e}'
                +f' cpu: {update_time:.1f}'
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
        # if use_adjoint:
        #     tape.clear_tape()
                        
        #     # other clean up taken from 
        #     # https://github.com/LLNL/pyMMAopt/commit/e2f83bd932207a8adbd60ae793b3e5a3058daecf
        #     #TSFCKernel._cache.clear()
        #     #GlobalKernel._cache.clear()
        # gc.collect()
        # petsc4py.PETSc.garbage_cleanup(self.mesh._comm)
        # petsc4py.PETSc.garbage_cleanup(self.mesh.comm)

        return ierr_dmk
              
    def discrepancy(self, pot, tdens ):
        '''
        Measure the discrepancy between I(tdens) and the observed data.
        '''
        # store image
        self.image_h.interpolate(self.tdens2image_map(tdens))

        #with self.image_h.dat.vec as img_vec, self.reconstruction.dat.vec as img_rec_vec:
        #    img_vec.copy(img_rec_vec)
        #    #PETSc.Sys.Print(utilities.msg_bounds(img_rec_vec,'IMG recosntruction'))

        discrepancy_norm = self.ctrl_get('discrepancy_norm')
        if discrepancy_norm == "l2":
            dis = self.confidence * 0.5 * (self.image_h - self.img_observed)**2 * dx
        elif discrepancy_norm == "dual_h1":
            # this should be stored by adjoint
            #assemble(interpolate(self.image_h - self.img_observed,self.fems.tdens_space), tensor=self.difference_dual_h1)
            #self.difference_dual_h1.interpolate(self.image_h - self.img_observed)
            
            # dual norm is 
            # $ \int u (I(\mu) - Img_obs) dx
            # with
            # this -\Delta u = I(\mu) - Img_obs  should be
            self.dual_h1_solver.solve()
            dis = self.pot_dual_h1 * (self.image_h - self.img_observed) * dx
        else:
            raise ValueError(f'Wrong discrepancy norm {discrepancy_norm=}')

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
        :math:`int_{\Omega}\frac{1}{2\gamma} \mu^{\gamma} dx`
        '''
        return  0.5 * (tdens ** self.btp.gamma) /  self.btp.gamma  * dx

    def penalization(self, pot, tdens): 
        ''' 
        Definition of the penalization functional as the branched transport energy
        in FCP2021 (use Citations.print_all() to see the reference)
        The penalization is defined as
        :math:`int_{\Omega} f u dx - \frac{\mu |\nabla u|^2}{2}+ \frac{1}{2\gamma} \mu^{\gamma} dx`
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
        rw = self.ctrl_get('regularization_weight')
        pw = self.ctrl_get('penalization_weight')
        dw = self.ctrl_get('discrepancy_weight')

        self.discrepancy_weight.assign(dw)
        self.penalization_weight.assign(pw)
        self.regularization_weight.assign(rw)

        Lag = self.penalization_weight * self.penalization(pot,tdens)
        if abs(dw) > 1e-16:
            Lag += self.discrepancy_weight * self.discrepancy(pot,tdens)
        if abs(rw) > 1e-16:
            Lag += self.regularization_weight * self.regularization(pot,tdens)
        return Lag
    
    #@profile
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
            self.pot_solver.solve()
        except:
            pass
        ierr = self.pot_solver.snes.getConvergedReason()
        
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
        pot.assign(self.pot_h,annotate=False)

        return ierr
    
    #@profile
    def tdens_mirror_descent_explicit(self, sol):
        # Tdens is udpdate along the direction of the gradient
        # of the energy w.r.t. tdens multiply by tdens**(2-gamma)
        # 
        #
        pot, tdens = sol.subfunctions
        self.pot_h.assign(pot,annotate=False) 
        self.tdens_h.assign(tdens)
        

        # We compute the gradient w.r.t to tdens of the Lagrangian
        if not self.gradients_computed:
            self.compute_residuum(sol)
        
        self.rhs_ode.assign(-1.0*self.gradient_lagrangian)


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

        self.discrepancy_weight.assign(wd)
        self.penalization_weight.assign(wp)
        self.regularization_weight.assign(wr)


        self.lagrangian_fun = assemble(
            self.discrepancy_weight * self.discrepancy(pot, tdens)
            + self.penalization_weight * self.penalization(pot, tdens)
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
    
    
    def save_function(self, func, filename):
        '''
        Save into a file the function func
        '''
        
        # get extension
        ext = filename.split('.')[-1]
        if ext == 'gz':
            cartesian_fun = assemble(interpolate(func,self.DG0_cartesian))
            
        elif ext == 'h5':
            with HDF5File(self.mesh.comm, filename, 'w') as h5f:
                h5f.write(func, 'function')
        else:
            raise ValueError(f'Extension {ext} not supported.')



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
    
