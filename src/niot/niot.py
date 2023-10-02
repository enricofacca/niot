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
import utilities
import linear_algebra_utilities as linalg
from petsc4py import PETSc
import time 
from memory_profiler import profile
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
utilities.include_citations('../citations/citations.bib')



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

        

    def create_solution(self):
        '''
        Create a mixed function sol=(pot,tdens) 
        defined on the mixed function space.
        '''
        sol = Function(self.pot_tdens_space,name=('pot','tdens'))
        sol.sub(0).vector()[:] = 0.0
        sol.sub(1).vector()[:] = 1.0

        return sol

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
                    
class Controls:
    '''
    Class with Dmk Solver 
    '''
    def __init__(self,
                tol=1e-2,
                time_discretization_method='mirror_descent',
                deltat=0.5,
                max_iter=100,
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

        #: real : lower bound for tdens
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

        #: info save solution
        self.save_solution = 'not'
        #: int: frequency of solution saving
        self.save_frequency = 10
        self.save_directory = './'

    def set_step(self, increment):
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
        elif (self.deltat_control == 'expansive'):
            step = max(min(self.deltat * self.deltat_expansion, self.deltat_max),self.deltat_min)
        elif (self.deltat_control == 'fixed'):
            step = self.deltat
        return step

    def print_info(self, msg, priority=0, color='black'):
        '''
        Print messagge to stdout and to log 
        file according to priority passed
        '''        
        if (self.verbose > priority):
            if color != 'black':
                msg = utilities.color_msg(color, msg)
            print(msg)   



 
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

    def __init__(self, spaces, numpy_corrupted, DG0_cell2face='harmonic_mean'):
        '''
        Initialize solver (spatial discretization) from numpy_image (2d or 3d data)
        '''
        self.spaces = spaces
        if spaces == 'CR1DG0':
            print('building mesh')
            self.mesh = self.build_mesh_from_numpy(numpy_corrupted, mesh_type='simplicial')
            self.mesh_type = 'simplicial'
            # create FEM spaces
            print('building fems')
            self.fems = SpaceDiscretization(self.mesh,'CR',1, 'DG',0)
        elif spaces == 'DG0DG0':
            self.mesh = self.build_mesh_from_numpy(numpy_corrupted, mesh_type='cartesian')
            self.mesh_type = 'cartesian'
            # create FEM spaces
            self.fems = SpaceDiscretization(self.mesh,'DG',0,'DG',0)
        else:
            raise ValueError('Wrong spaces only (pot,tdens) in (CR1,DG0) or (DG0,DG0) implemented')
        self.mesh.name = 'mesh'
        print(f'Number of cells: {self.mesh.num_cells()}')

        self.DG0 = FunctionSpace(self.mesh, 'DG', 0)

        # set function 
        self.img_observed = self.numpy2function(numpy_corrupted,name='obs')
    
        # set niot parameter to default
        self.set_inpainting_parameters(
            weights=[1.0,1.0,0.0],
            confidence=1.0,
            tdens2image='identity')
        
        
        # init infos
        self.outer_iterations = 0
        self.nonlinear_iterations = 0
        self.nonlinear_res = 0.0

        # cached functions
        self.rhs_ode = Function(self.fems.tdens_space)
        self.tdens_h = Function(self.fems.tdens_space)
        self.tdens_h.rename('tdens_h')
        self.pot_h = Function(self.fems.pot_space)

        alpha = Constant(4.0)
        h = CellSize(self.mesh)
        h_avg = (h('+') + h('-'))/2.0
        self.DG0_scaling = alpha/h_avg
        self.normal = FacetNormal(self.mesh)
        
        if ((DG0_cell2face == 'arithmetic_mean') or
            (DG0_cell2face == 'harmonic_mean')):
            self.DG0_cell2face = DG0_cell2face
        else:   
            raise ValueError('Wrong DG0_cell2face method. Passed '+DG0_cell2face+
                            '\n Only arithmetic_mean and harmonic_mean are implemented')
        
        
    def setup_pot_solver(self,ctrl):
        self.pot_PDE = derivative(self.Lagrangian(self.pot_h,self.tdens_h),self.pot_h)
        self.weighted_Laplacian = derivative(-self.pot_PDE,self.pot_h)
        self.rhs = self.forcing * self.fems.pot_test * dx

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
            'ksp_monitor_true_residual': None,
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
                                                appctx=context)
        self.snes_solver.snes.setConvergenceHistory()        
        

    def build_mesh_from_numpy(self, np_image, mesh_type='simplicial'): 
        '''
        Create a mesh (first axis size=1) from a numpy array
        '''
        
        if (np_image.ndim == 2):
            if (mesh_type == 'simplicial'):
                quadrilateral = False
            elif (mesh_type == 'cartesian'):
                quadrilateral = True
            print(f'Mesh type {mesh_type} quadrilateral {quadrilateral}')
            
            height, width  = np_image.shape
            print(f'{width=} {height=}')
            self.nx = height
            self.ny = width
            self.nz = 0

            mesh = RectangleMesh(
                width,
                height,
                1,
                height/width, 
                quadrilateral = quadrilateral,
                reorder=False,
                diagonal="right",
                )
                
        elif (np_image.ndim == 3):
            height, width, depth = np_image.shape
            self.nx = width
            self.ny = height
            self.nz = depth

            if (mesh_type == 'simplicial'):
                hexahedral = False
            elif (mesh_type == 'cartesian'):
                hexahedral = True
            print(f'{mesh_type=} {hexahedral=}')
            mesh = BoxMesh(nx=height,
                        ny=width, 
                        nz=depth,  
                        Lx=1, 
                        Ly=height/width,
                        Lz=height/depth,
                        hexahedral=hexahedral,
                        reorder=False,
                        #diagonal="default"
                        )
        else:
            raise ValueError('Only 2D and 3D images are supported')
        return mesh

    def set_optimal_transport_parameters(self, source, sink, 
                                         gamma=0.8,
                                         kappa=1.0,
                                         Neumann = [[0,'on_boundary']],
                                         Dirichlet=None,
                                         tolerance_unbalance=1e-12,
                                         force_balance=False,
                                         min_tdens=1e-8):
        '''
        Set optimal transport parameters
        Args:
        source: firedrake function with source term
        sink: firedrake function with sink term
        gamma: penalization parameter
        kappa: local resistivity parameter
        Neumann: list of Neumann boundary conditions
        Dirichlet: list of Dirichlet boundary conditions
        tolerance_unbalance: tolerance to check if source and sink are balanced
        force_balance: boolean to force balancing sink to ensure problem to be well posed
        '''

        # Check data consistency
        mass_source = assemble(source*dx)
        mass_sink = assemble(sink*dx)
        mass_Neumann = 0#assemble(Neumann*dS)

        mass_balance = mass_source - mass_sink + mass_Neumann

        print('mass_source',mass_source)
        print('mass_sink',mass_sink)
        print('mass_Neumann',mass_Neumann)
        print('mass_balance',mass_balance)

        # create hard copy because we may scale
        # the source and sink termsnp_
        self.source = cp(source)
        self.sink = cp(sink)
        if abs(mass_balance) > tolerance_unbalance :
            if force_balance:
                with self.sink.dat.vec as f:
                    f.scale(mass_source / mass_sink)
            else:
                raise ValueError('Source and sink terms are not balanced')

        
        self.forcing = self.source - self.sink
        rhs = assemble(self.forcing * self.fems.pot_test * dx)
        with rhs.dat.vec as f:
            self.f_norm  = f.norm()
        self.gamma = gamma
        
        self.kappa = Function(self.fems.tdens_space)
        if isinstance(kappa, float):
            self.kappa.assign(kappa)
        elif isinstance(kappa, Function):
            self.kappa.project(kappa)
        else:
            raise ValueError('kappa must be a float or a function')
        self.Neumann = Neumann
        self.Dirichlet = Dirichlet
        self.min_tdens = min_tdens
       
    def set_inpainting_parameters(self,
                 weights=np.array([1.0,1.0,0.0]),
                 confidence=1.0,
                 tdens2image='identity',
                 scaling=1.0, # used only if tdens2image is 'scaling'
                 sigma_smoothing=0.1 # used only if tdens2image is 'laplacian_smoothing'
                 ):
        '''
        Set inpainting specific parameters 
        Args:
        weights: array with weights of the discrepancy, penalization and transport energy
        confidence: confidence of the observed data
        tdens2image: function that maps tdens to image

        Returns:
        None
        '''
        if weights is not None:
            if len(weights)!=3:
                raise ValueError(f'3 weights are required, Transport Energy, Discrepancy, Penalization')
            self.weights = weights

        if confidence is not None:
            self.confidence = self.convert_data(confidence)
            #self.confidence.rename('confidence','confidence')
        

        print(f'{tdens2image=}')
        if tdens2image == 'identity':
            self.tdens2image = lambda x: x
        elif tdens2image == 'scaling':
            self.scaling = scaling
            self.tdens2image = lambda x: scaling*x
        elif tdens2image == 'laplacian_smoothing':
            self.sigma_smoothing = sigma_smoothing
            test = TestFunction(self.fems.tdens_space)  
            trial = TrialFunction(self.fems.tdens_space)
            if  self.mesh.ufl_cell().is_simplex():
                form = ( inner(test, trial)*dx 
                        + self.sigma_smoothing * inner(jump(test), jump(trial))  / self.fems.delta_h * dS )
            else:
                form = ( inner(test, trial)*dx 
                        + self.sigma_smoothing * inner(grad(test), grad(trial))*dx)
            
            self.LaplacianMatrix = assemble(form).M.handle
            self.LaplacianSmoother = linalg.LinSolMatrix(self.LaplacianMatrix,
                                                        self.fems.tdens_space, 
                                                        solver_parameters={
                                            'ksp_type': 'cg',
                                            'ksp_rtol': 1e-10,
                                            'pc_type': 'hypre'})
            
            self.tdens2image = lambda x: utils.LaplacianSmoothing(x,self.LaplacianSmoother)
        else:
            raise ValueError(f'Map tdens2iamge not supported {tdens2image}\n'
                             +'Only identity, scaling and laplacian_smoothing are implemented')
            

    def numpy2function(self, value, name=None):
        '''
        Convert np array (2d o 3d) into a function compatible with the mesh solver.
        Args:
        
        value: numpy array (2d or 3d) with images values

        returns: piecewise constant firedake function 
        ''' 
        
        # we flip vertically because images are read from left, right, top to bottom
        

        if self.mesh.ufl_cell().is_simplex():
            # Each pixel is splitted in two triangles.
            if self.mesh.geometric_dimension() == 2:
                value = np.flip(value,0)

                # double the value to copy the pixel value to the triangles
                double_value = np.zeros([2,value.shape[0],value.shape[1]])
                double_value[0,:,:] = value[:,:]
                double_value[1,:,:] = value[:,:]
                triangles_image = double_value.swapaxes(0,2).flatten()
                DG0 = FunctionSpace(self.mesh,'DG',0)
                
                img_function = Function(DG0)#, value=triangles_image, name=name)
                with img_function.dat.vec as d:
                    d.array = triangles_image
                if name is not None:
                    img_function.rename(name,name)
            elif self.mesh.geometric_dimension() == 3:
                # copy the value to copy the voxel value
                # flatten the numpy matrix (EF: I don't know why swapaxes is needed)
                print('reshaping')
                flat_value = value.swapaxes(0,2).flatten()
                print('reshaping done')
                ncopies = self.mesh.num_cells() // (value.shape[0]*value.shape[1]*value.shape[2])
                copies = np.zeros([ncopies,len(flat_value)])
                for i in range(ncopies):
                    print(f'{i=}')
                    copies[i,:] = flat_value[:]
                print('reshaping')
                cells_values = copies.swapaxes(0,1).flatten()
                print('reshaping done')
                # define function
                DG0 = FunctionSpace(self.mesh,'DG',0)
                img_function = Function(DG0, val=cells_values, name=name)
                
            return img_function
        else:
            if (self.mesh.ufl_cell().cellname() != 'quadrilateral'):
                raise ValueError('Only simplicial and quadrilateral meshes are supported')
            DG0 = FunctionSpace(self.mesh,'DG',0)
            img_function = Function(DG0)
            value_flipped = np.flip(value,0)
            with img_function.dat.vec as d:
                d.array = value_flipped.flatten('F')
            if (name is not None):
                img_function.rename(name,name)
            return img_function
        
    def function2numpy(self, function):
        """
        Convert function to numpy array (2d or 3d).
        If the mesh is simplicial, the function is averaged neighbouring cells.
        If the mesh is cartesian, the results is reshaped to the original image shape.
        """
        if self.mesh.ufl_cell().is_simplex():
            # Each pixel is splitted in two triangles.
            if self.mesh.geometric_dimension() == 2:
                # get the values of the function
                with function.dat.vec_ro as f:
                    value = f.array
                    print(f'{value.shape=}')
                    # reshape the values in a matrix of size (2,nx*ny)
                    value = value.reshape([-1,2])
                    # average the values along the first dimension
                    value = np.mean(value,1)
                    print(f'{value.shape=}')
                    # reshape the values in a matrix of size (nx,ny)
                    value = value.reshape([self.nx,self.ny],order='F')

                    return np.flip(value,0)
                
            elif self.mesh.geometric_dimension() == 3:
                raise NotImplementedError('3D mesh not implemented yet')
        else:
            if (self.mesh.ufl_cell().cellname() != 'quadrilateral'):
                raise ValueError('Only simplicial and quadrilateral meshes are supported')
            # get the values of the function
            with function.dat.vec_ro as f:
                value = f.array
                # reshape the values in a matrix of size (nx,ny)
                new_shape = [self.nx,self.ny]
                if self.mesh.geometric_dimension() == 3:
                    new_shape.append(self.nz)
                value = value.reshape(new_shape,order='F')
                return np.flip(value,0)


    def convert_data(self, data):
        print(type(data))
        if isinstance(data, np.ndarray):
            return self.numpy2function(data)
        elif isinstance(data, functionspaceimpl.FunctionSpace) or isinstance(data, function.Function):
            # function must be defined on the same mesh
            if data.function_space().mesh() == self.mesh:
                return data
            else:
                raise ValueError('Data not defined on the same mesh of the solver')
        elif isinstance(data, float):
            return Constant(data)

        elif isinstance(data, constant.Constant):
            return data 
        else:
            raise ValueError('Type '+str(type(data))+' not supported')

    def record_algorithm(self, sol, ctrl, current_iteration):
        """
        Record data along algorithm exceution
        """
        if ctrl.save_solution == 'all':
            filename = os.path.join(ctrl.save_directory,f'sol{current_iteration:06d}.pvd')
            self.save_solution(sol,filename)
        elif (ctrl.save_solution == 'some') and (current_iteration % ctrl.save_solution_every == 0):
            filename = os.path.join(ctrl.save_directory,f'sol{current_iteration:06d}.pvd')
            self.save_solution(sol,filename)
        

    def residual(self, sol):
        """
        Return the residual of the minimization problem
        w.r.t to tdens
        """
        pot, tdens = sol.subfunctions

        tdens_h = Function(self.fems.tdens_space)
        tdens_h.assign(tdens)
        tdens_PDE = derivative(self.Lagrangian(pot,tdens_h),tdens_h)
        f = assemble(tdens_PDE)
        d = self.fems.tdens_mass_matrix.createVecLeft()
        with f.dat.vec_ro as f_vec, tdens_h.dat.vec as tdens_vec:
            self.fems.tdens_inv_mass_matrix.solve(f_vec,d)
            d *= tdens_vec
            residuum = d.norm()#PETSc.NormType.NORM_INFINITY)
        return residuum

    def create_solution(self):
        return self.fems.create_solution()
    
    @profile
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

        print('solving pot_0')
        self.setup_pot_solver(ctrl)
        ierr = self.solve_pot_PDE(ctrl, sol)
        avg_outer = self.outer_iterations / max(self.nonlinear_iterations,1)
        print(utilities.color('green',
                            f'It: {0} '+
                f' avgouter: {avg_outer:.1f}'))

        iter = 0
        ierr_dmk = 0
        sol_old = cp(sol)           
        tape.clear_tape()
        while ierr_dmk == 0:
            # update with restarts
            sol_old.assign(sol)
            nrestart = 0 
            while nrestart < ctrl.max_restart:
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
                    print(ierr)
                    # reset controls after failure
                    ctrl.deltat = max(ctrl.deltat_min,ctrl.deltat_contraction * ctrl.deltat)
                    print(f' Failure in due to {SNESReasons[ierr]}. Restarting with deltat = {ctrl.deltat:.2e}')

                    # restore old solution
                    sol.assign(sol_old)

            if (ierr != 0):
                print('ierr',ierr)
                break

            # study state of convergence
            iter += 1
            
            var_tdens = self.residual(sol)
            avg_outer = self.outer_iterations / max(self.nonlinear_iterations,1)
            print(utilities.color('green',
                            f'It: {iter} '+
                    f' dt: {ctrl.deltat:.1e}'+
                    f' var:{var_tdens:.1e}'
                    f' nsym:{self.nonlinear_iterations:1d}'+
                    f' avgouter: {avg_outer:.1f}'))
            with sol.dat.vec_ro as sol_vec:
                tdens_vec = sol_vec.getSubVector(self.fems.tdens_is)
                print(msg_bounds(tdens_vec,'tdens'))

            # save data    
            self.record_algorithm(sol, ctrl, iter)

            # check convergence
            if (var_tdens < ctrl.tol ):
                ierr_dmk = 0
                break

            # break if max iter is reached
            if (iter == ctrl.max_iter):
                ierr_dmk = 1
        
        #fire_adj.get_working_tape().clear_tape()

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
        return  0.5 * (tdens ** self.gamma) /  self.gamma  * dx

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
                reg = 0.5 * self.DG0_scaling * inner(jump(tdens, self.normal), jump(tdens, self.normal)) * dS
            else:
                reg = 0.5 * jump(tdens)**2 / self.fems.delta_h * dS
        else:
            raise NotImplementedError('Only piecewise constant tdens is implemented')
            
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
        
        if False:
            pot, tdens = sol.subfunctions
            pot_unknown = self.pot_h
            pot_unknown.assign(pot)

            # we scale by the norm of the forcing term
            # to ensure that we get the relative residual
            pot_PDE = 1/self.f_norm * derivative(self.Lagrangian(pot_unknown,tdens),pot_unknown)
            pot_bcs = self.Dirichlet
            #test = TestFunction(self.fems.pot_space)
            #pot_PDE = self.forcing * test * dx + tdens * inner(grad(pot_unknown), grad(test)) * dx 
            # Define the Nonlinear variational problem (it is linear in this case)
            self.u_prob = NonlinearVariationalProblem(pot_PDE, pot_unknown)#, bcs=pot_bcs)

            # set the solver parameters
            snes_ctrl={
                'snes_rtol': 1e-16,
                'snes_atol': ctrl.nonlinear_tol,
                'snes_stol': 1e-16,
                'snes_type': 'ksponly',
                'snes_max_it': ctrl.nonlinear_max_iter,
                # krylov solver controls
                'ksp_type': 'cg',
                'ksp_atol': 1e-30,
                'ksp_rtol': ctrl.nonlinear_tol,
                'ksp_divtol': 1e4,
                'ksp_max_it' : 100,
                # preconditioner controls
                'pc_type': 'hypre',
            }
            if ctrl.verbose >= 1:
                snes_ctrl['snes_monitor'] = None
            if  ctrl.verbose >= 2:
                snes_ctrl['ksp_monitor'] = None
            
            context ={} # left to pass information to the solver
            if self.Dirichlet is None:
                nullspace = VectorSpaceBasis(constant=True,comm=COMM_WORLD)
            
            self.snes_solver = NonlinearVariationalSolver(self.u_prob,
                                                    solver_parameters=snes_ctrl,
                                                    nullspace=nullspace,
                                                    appctx=context)
            self.snes_solver.snes.setConvergenceHistory()        
        
        pot, tdens = sol.subfunctions
        self.pot_h.assign(pot)
        self.tdens_h.assign(tdens)

        # solve the problem
        try:
            self.snes_solver.solve()
        except:
            pass
        ierr = self.snes_solver.snes.getConvergedReason()
        
        print('ierr',ierr)
            
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
    
    def tdens_mirror_descent(self, ctrl, sol):
        '''
        Procedure for the update of tdens using mirror descent.
        args:
            ctrl : Class with controls  (tol, max_iter, deltat, etc.)
            sol : Class with mixed function [pot,tdens]. It is changed in place.
        returns:
            ierr : control flag. It is 0 if everthing worked.
        '''

        # compute gradient of energy w.r.t. tdens
        pot, tdens = sol.subfunctions
        tdens_h = Function(self.fems.tdens_space)
        tdens_h.assign(tdens)
        PDE_tdens = derivative(self.Lagrangian(pot,tdens_h),tdens_h)
        bcs_tdens = None
        
        
        # compute update vectors using mass matrix
        rhs_ode = assemble(PDE_tdens, bcs=bcs_tdens)
        update = Function(self.fems.tdens_space)
        scaling = Function(self.fems.tdens_space)
        scaling.interpolate(tdens_h)#**(2-self.gamma))
        with rhs_ode.dat.vec as rhs, scaling.dat.vec_ro as scaling_vec, update.dat.vec as d, tdens_h.dat.vec_ro as tdens_vec:
            # scale the gradient w.r.t. tdens by tdens itself
            rhs *= scaling_vec
            # scale by the inverse mass matrix
            self.fems.tdens_inv_mass_matrix.solve(rhs, d)
            # update
            # update
            step = ctrl.set_step(d)
            ctrl.deltat = step
            tdens_vec.axpy(-ctrl.deltat, d)

        utilities.threshold_from_below(tdens_h, 0)

        sol.sub(1).assign(tdens_h)
        
        
        # compute pot associated to new tdens
        ierr = self.solve_pot_PDE(ctrl, sol)
        
        return ierr
    
    def gfvar2tdens(self,gfvar):
        return gfvar**(2/self.gamma)
        #return 2*atan(gfvar)
    def tdens2gfvar(self,tdens):
        return tdens**(self.gamma/2)
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
        gfvar = Function(self.fems.tdens_space)
        gfvar.interpolate(self.tdens2gfvar(tdens))

        # compute gradient of energy w.r.t. gfvar
        PDE_discr = derivative(self.discrepancy(pot,self.gfvar2tdens(gfvar)),gfvar)
        PDE_otp = derivative(self.penalization(pot,self.gfvar2tdens(gfvar)),gfvar)
        PDE_reg = derivative(self.regularization(pot,gfvar),gfvar)

        PDE = (self.weights[0] * PDE_discr 
               + self.weights[1] * PDE_otp 
               + self.weights[2] * PDE_reg)
        bcs_gfvar = None
        
        # matrix for the mass matrix
        # test = self.fems.tdens_test
        # trial = self.fems.tdens_trial
        # form = test*trial*dx + derivative(self.weights[2]*PDE_reg, gfvar)
        # matrix = assemble(form).M.handle
        # inv_matrix = linalg.LinSolMatrix(matrix, self.fems.tdens_space,
        #                 solver_parameters={
        #                     'ksp_type':'cg',
        #                     'ksp_rtol': 1e-6,
        #                     'pc_type':'hypre'})






        # compute update vectors using mass matrix
        rhs_ode = assemble(PDE, bcs=bcs_gfvar)
        update = Function(self.fems.tdens_space)
        with rhs_ode.dat.vec as rhs, gfvar.dat.vec_ro as gfvar_vec, update.dat.vec as d:
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
    
    #@profile
    def mirror_descent(self, ctrl, sol):
        # Tdens is udpdate along the direction of the gradient
        # of the energy w.r.t. tdens multiply by tdens**(2-gamma)
        # 
        #
        pot, tdens = sol.subfunctions
        self.pot_h.assign(pot) 
        self.tdens_h.assign(tdens)
       
        new = True
        if new:
            self.lagrangian_fun = assemble(self.Lagrangian(self.pot_h,self.tdens_h))
            with fire_adj.stop_annotating():
                self.lagrangian_fun_reduced = fire_adj.ReducedFunctional(self.lagrangian_fun, fire_adj.Control(self.tdens_h))
                self.rhs_ode = self.lagrangian_fun_reduced.derivative()
            #self.rhs_ode = fire_adj.compute_gradient(self.lagrangian_fun, fire_adj.Control(self.tdens_h)) #self.lagrangian_fun_reduced.derivative()
            gc.collect()
        else:
            PDE_tdens_discr = derivative(self.weights[0]*self.discrepancy(pot,self.tdens_h),self.tdens_h)
            PDE_tdens_opt = derivative(self.weights[1]*self.penalization(pot,self.tdens_h),self.tdens_h)
            
            PDE_tdens_reg = derivative(self.weights[2]*self.regularization(pot,self.tdens_h),self.tdens_h)
            if (self.spaces=='CR1DG0' and self.weights[2]>1e-16):
                test = self.fems.tdens_test
                PDE_tdens_reg = self.weights[2] * self.DG0_scaling * inner(jump(self.tdens_h, self.normal), jump(test, self.normal)) * dS
            bcs_tdens = None

            self.PDE_tdens = (PDE_tdens_discr + PDE_tdens_opt + PDE_tdens_reg)
        
            # compute update vectors using mass matrix
            time_start = time.time()
            self.rhs_ode = assemble(self.PDE_tdens, bcs=bcs_tdens)
            time_stop = time.time()


        #rhs_reg = assemble(PDE_tdens_reg, bcs=bcs_tdens)
        update = Function(self.fems.tdens_space)
        scaling = Function(self.fems.tdens_space)
        scaling.interpolate(self.tdens_h**(2-self.gamma))
        with self.rhs_ode.dat.vec as rhs, scaling.dat.vec_ro as scaling_vec, update.dat.vec as d, self.tdens_h.dat.vec_ro as tdens_vec:
            # scale the gradient w.r.t. tdens by tdens itself
            rhs *= scaling_vec
            print(utilities.msg_bounds(rhs,'rhs'))
            

            #if abs(self.weights[2])>1e-16:
            #    print('regularization')
            #    with rhs_reg.dat.vec_ro as rhs_reg_vec: 
            #        rhs.axpy(1., rhs_reg_vec)
            # scale by the inverse mass matrix
            self.fems.tdens_inv_mass_matrix.solve(rhs, d)
            # set step size
            step = ctrl.set_step(d)
            print(utilities.msg_bounds(d,'tdens step')+f' dt={step:.2e}')
            ctrl.deltat = step
            print('step',step)
            # update
            tdens_vec.axpy(-step, d)

        # threshold from below tdens
        utilities.threshold_from_below(self.tdens_h, 0)

        # assign new tdens to solution
        sol.sub(1).assign(self.tdens_h)
        
        # compute pot associated to new tdens
        time_start = time.time()
        ierr = self.solve_pot_PDE(ctrl, sol)
        time_stop = time.time()
        ctrl.print_info(f'solve_pot_PDE time = {time_stop-time_start}', priority=2)
        
        return ierr            
        
    def iterate(self, ctrl, sol):
        '''
        Args:
         ctrl : Class with controls  (tol, max_iter, deltat, etc.)
         solution updated from time t^k to t^{k+1} solution updated from time t^k to t^{k+1} sol : Class with unkowns (pot,tdens, in this case)

        Returns:
         ierr : control flag. It is 0 if everthing worked.

        '''
        if (ctrl.time_discretization_method == 'mirror_descent'):
            ierr = self.mirror_descent(ctrl, sol)
            return ierr
        
        elif (ctrl.time_discretization_method == 'tdens_mirror_descent'):
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
    

    def save_inputs(self, file_name):
        '''
        Save to file the inputs of the solver
        '''
        utilities.save2pvd(
            [self.source,
             self.sink,
             self.img_observed,
             self.confidence],
             file_name)

def heat_spread(density, tau=0.1):
    '''
    Spread a density using the porous medium equation,
    :math:`\partial_t \rho - \nabla \cdot ( \nabla \rho)=0`.
    '''
    # create the function space
    mesh = density.function_space().mesh()
    space = FunctionSpace(mesh, 'CG', 1)
    new_density = Function(space)
    new_density.interpolate(density)
 
    test = TestFunction(space)
    PDE = ( 
        1/tau * (new_density - density) * test  * dx 
        + inner(grad(new_density), grad(test)) * dx
        )
    problem = NonlinearVariationalProblem(PDE, new_density)

    # set solver. One Newton step is required for m=1
    ctrl={
            'snes_rtol': 1e-16,
            'snes_atol': 1e-4,
            'snes_stol': 1e-16,
            'snes_type': 'newtonls',
            'snes_max_it': 20,
            'snes_monitor': None,
            # krylov solver controls
            'ksp_type': 'cg',
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
        
    # return the solution in the same function space of the input
    smooth_density = Function(density.function_space())
    smooth_density.interpolate(new_density)

    return smooth_density

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
