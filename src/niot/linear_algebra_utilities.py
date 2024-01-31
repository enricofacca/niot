from petsc4py import PETSc
from firedrake import MixedVectorSpaceBasis
from time import process_time
import sys


def study_eigenvalues(test_matrix):
    from slepc4py import SLEPc
    E = SLEPc.EPS(); E.create()
    E.setOperators(test_matrix)
    E.setProblemType(SLEPc.EPS.ProblemType.NHEP)
    E.setWhichEigenpairs(SLEPc.EPS.Which.ALL) #'smallest magnitude'
    E.setType(SLEPc.EPS.Type.LAPACK)
    E.setFromOptions()  

    E.solve()

    return E

def print_eigenvalues(test_matrix,E):
    
    Print = PETSc.Sys.Print

    Print()
    Print("******************************")
    Print("*** SLEPc Solution Results ***")
    Print("******************************")
    Print()

    its = E.getIterationNumber()
    Print("Number of iterations of the method: %d" % its)

    eps_type = E.getType()
    Print("Solution method: %s" % eps_type)

    nev, ncv, mpd = E.getDimensions()
    Print("Number of requested eigenvalues: %d" % nev)

    tol, maxit = E.getTolerances()
    Print("Stopping condition: tol=%.4g, maxit=%d" % (tol, maxit))

    nconv = E.getConverged()
    Print("Number of converged eigenpairs %d" % nconv)
    if nconv > 0:
        #Create the results vectors
        vr, wr = test_matrix.getVecs()
        vi, wi = test_matrix.getVecs()
        
        Print()
        Print("        k          ||Ax-kx||/||kx|| ")
        Print("----------------- ------------------")
        for i in range(nconv):
            k = E.getEigenpair(i, vr, vi)
            error = E.computeError(i)
            if k.imag != 0.0:
                Print(" %9f%+9f j %12g" % (k.real, k.imag, error))
            else:
                Print(" %12f      %12g" % (k.real, error))
        Print()

def diag(diag_vector):
    """
    Return the diagonal matrix with the entries of diag_vec
    """
    D = PETSc.Mat().create()
    n=diag_vector.getSize()
    D.setSizes([n,n])
    D.setType('aij') 
    D.setUp()
    D.setDiagonal(diag_vector)
    return D


def sparse_inverse(sparse_matrix,prec_type='hypre'):
    # Create a KSP object to apply the inverse of the matrix
    prec = PETSc.PC().create()
    prec.setOperators(sparse_matrix)
    prec.setType(prec_type)
    prec.setFromOptions()
    prec.setUp()
    return prec

def as_prec(prec_object):
    """
    Return a PETSc PC object from a preconditioner object
    """
    prec = PETSc.PC().create()
    prec.setType('python')
    prec.setPythonContext(prec_object)
    prec.setFromOptions()
    prec.setUp()
    return prec

def chop(A, tol=1E-10):
    # remove (near) zeros from sparsity pattern
    A.chop(tol)
    B = PETSc.Mat().create(comm=A.comm)
    B.setType(A.getType())
    B.setSizes(A.getSizes())
    B.setBlockSize(A.getBlockSize())
    B.setUp()
    B.setOption(PETSc.Mat.Option.IGNORE_ZERO_ENTRIES, True)
    B.setPreallocationCSR(A.getValuesCSR())
    B.assemble()
    A.destroy()
    return B


def save2file(matrix_or_vector, filename,format='ascii'):
    """
    Save a PETSc vector to a file
    """
    if format=='binary':
        viewer = PETSc.Viewer().createBinary(filename, 'w')
    elif format=='ascii':
        viewer = PETSc.Viewer().createASCII(filename)
        viewer.pushFormat(PETSc.Viewer.Format.ASCII_MATLAB)
    matrix_or_vector.view(viewer)
    viewer.destroy()


# Hereafter we redefine the LinearSolver class 
# to work with matrix free matrices and preconditioners
#


"""

Readaption of LinearSolver class (firedrake/linear_solver.py)
for definition of class approximating inverse application 
without stopping in case of error and that can be used as preconditioner.

"""

from firedrake.exceptions import ConvergenceError
import firedrake.function as function
import firedrake.vector as vector
import firedrake.matrix as matrix
import firedrake.solving_utils as solving_utils
from firedrake import dmhooks
from firedrake.petsc import PETSc, OptionsManager
from firedrake.utils import cached_property
from firedrake.ufl_expr import action
import firedrake.variational_solver as vs


class LinSol(OptionsManager):

    DEFAULT_KSP_PARAMETERS = solving_utils.DEFAULT_KSP_PARAMETERS

    @PETSc.Log.EventDecorator()
    def __init__(self, A, sol, 
                P=None, Prec=None, appctx=None,
                solver_parameters=None,
                nullspace = None, transpose_nullspace=None,
                near_nullspace=None, options_prefix=None):
        """A linear solver for assembled systems (Ax = b).

        :arg A: a :class:`~.MatrixBase` (the operator).
        :arg sol: a :class:`~.Function containg the state varaible used to
            assemble the operator. It is used to assemble precondtioner 
            since is passed as "state" variable (see also).
        :arg appctx: Any extra information used in the assembler.  
        :arg P: an optional :class:`~.MatrixBase` to construct any
             preconditioner from; if none is supplied ``A`` is
             used to construct the preconditioner.
        :kwarg parameters: (optional) dict of solver parameters.
        :kwarg nullspace: an optional :class:`~.VectorSpaceBasis` (or
            :class:`~.MixedVectorSpaceBasis` spanning the null space
            of the operator.
        :kwarg transpose_nullspace: as for the nullspace, but used to
               make the right hand side consistent.
        :kwarg near_nullspace: as for the nullspace, but used to set
               the near nullpace.
        :kwarg options_prefix: an optional prefix used to distinguish
               PETSc options.  If not provided a unique prefix will be
               created.  Use this option if you want to pass options
               to the solver from the command line in addition to
               through the ``solver_parameters`` dict.

        .. note::

          Any boundary conditions for this solve *must* have been
          applied when assembling the operator.
        """
        if not isinstance(A, matrix.MatrixBase):
            raise TypeError("Provided operator is a '%s', not a MatrixBase" % type(A).__name__)
        if P is not None and not isinstance(P, matrix.MatrixBase):
            raise TypeError("Provided preconditioner is a '%s', not a MatrixBase" % type(P).__name__)

        #solver_parameters = solving_utils.set_defaults(solver_parameters,
        #                                               A.a.arguments(),
        #                                               ksp_defaults=self.DEFAULT_KSP_PARAMETERS)
        self.A = A
        self.comm = A.comm
        self.P = P if P is not None else A

        # Set up parameters mixin
        super().__init__(solver_parameters, options_prefix)
        
        self.A.petscmat.setOptionsPrefix(self.options_prefix)
        self.P.petscmat.setOptionsPrefix(self.options_prefix)

        # If preconditioning matrix is matrix-free, then default to jacobi
        if isinstance(self.P, matrix.ImplicitMatrix):
            self.set_default_parameter("pc_type", "jacobi")

        self.ksp = PETSc.KSP().create(comm=self.comm)

        W = self.test_space
        # DM provides fieldsplits (but not operators)
        self.ksp.setDM(W.dm)
        self.ksp.setDMActive(False)

        
        if nullspace is not None:
            nullspace._apply(self.A)
            if P is not None:
                nullspace._apply(self.P)

        if transpose_nullspace is not None:
            transpose_nullspace._apply(self.A, transpose=True)
            if P is not None:
                transpose_nullspace._apply(self.P, transpose=True)

        if near_nullspace is not None:
            near_nullspace._apply(self.A, near=True)
            if P is not None:
                near_nullspace._apply(self.P, near=True)

        self.nullspace = nullspace
        self.transpose_nullspace = transpose_nullspace
        self.near_nullspace = near_nullspace
        # Operator setting must come after null space has been
        # applied
        self.ksp.setOperators(A=self.A.petscmat, P=self.P.petscmat)
        # Set from options now (we're not allowed to change parameters
        # anyway).
        if (Prec is not None):
            if (P is not None):
                raisewarning("Both P and Prec are not None, P will be ignored")
            self.ksp.setPC(Prec)
        
        self.set_from_options(self.ksp)
        self.options_prefix = self.ksp.getOptionsPrefix()


        # linear MG doesn't need RHS, supply zero.
        self.lvp = vs.LinearVariationalProblem(a=A.a, L=0, u=sol, bcs=A.bcs)
        mat_type = A.mat_type
        if appctx is None:
            appctx = {}#solver_parameters.get("appctx", {})
        self.ctx = solving_utils._SNESContext(self.lvp,
                                     mat_type=mat_type,
                                     pmat_type=mat_type,
                                     appctx=appctx,
                                     options_prefix=options_prefix)

        
        
       
        self._tmp = self.A.petscmat.createVecLeft()
        self.reason = 0
        self.solved = 0
        self.cumulative_iterations = 0
        self.initial_res = 0.0
        self.final_res = 0.0
        self.last_cpu = 0.0

    @cached_property
    def test_space(self):
        return self.A.a.arguments()[0].function_space()


    @cached_property
    def trial_space(self):
        return self.A.a.arguments()[1].function_space()


    @cached_property
    def _rhs(self):
        from firedrake.assemble import OneFormAssembler

        u = function.Function(self.trial_space)
        b = function.Function(self.test_space)
        expr = -action(self.A.a, u)
        return u, OneFormAssembler(expr, tensor=b).assemble, b

    def _lifted(self, b):
        u, update, blift = self._rhs
        u.dat.zero()
        for bc in self.A.bcs:
            bc.apply(u)
        update()
        # blift contains -A u_bc
        blift += b
        for bc in self.A.bcs:
            bc.apply(blift)
        # blift is now b - A u_bc, and satisfies the boundary conditions
        return blift

    @PETSc.Log.EventDecorator()
    def solve_with_function(self, b, x):
        if not isinstance(x, (function.Function, vector.Vector)):
            raise TypeError("Provided solution is a '%s', not a Function or Vector" % type(x).__name__)
        if isinstance(b, vector.Vector):
            b = b.function
        if not isinstance(b, function.Function):
            raise TypeError("Provided RHS is a '%s', not a Function" % type(b).__name__)

        if len(self.trial_space) > 1 and self.nullspace is not None:
            self.nullspace._apply(self.trial_space.dof_dset.field_ises)
        if len(self.test_space) > 1 and self.transpose_nullspace is not None:
            self.transpose_nullspace._apply(self.test_space.dof_dset.field_ises,
                                            transpose=True)
        if len(self.trial_space) > 1 and self.near_nullspace is not None:
            self.near_nullspace._apply(self.trial_space.dof_dset.field_ises, near=True)

        if self.A.has_bcs:
            b = self._lifted(b)

        with x.dat.vec as sol, b.dat.vec_ro as rhs:
            self.solve(rhs,sol)
        

    def solve(self, rhs, sol):
        """
        Solve the linear system.
        Args:
            rhs (PETSc.Vec): The right-hand side vector.
            sol (PETSc.Vec): The solution vector.
        """
        if not self.ksp.getInitialGuessNonzero():
            sol.set(0.0)
        #print("initial guess non zeros", self.ksp.getInitialGuessNonzero())

        #print("sol", sol.array)

        time_start = process_time()
        with self.inserted_options(), dmhooks.add_hooks(self.ksp.dm, self,appctx=self.ctx):
            #rhs_p = rhs.getSubVector(self.test_space.dof_dset.field_ises[0])
            #rhs_td = rhs.getSubVector(self.test_space.dof_dset.field_ises[1])
            #rhs_s = rhs.getSubVector(self.test_space.dof_dset.field_ises[2])
            #print(rhs_s.array.sum())
            self.initial_res = self.true_residual(rhs,sol)
            self.ksp.solve(rhs, sol)
            #if (self.solved ==0):
            #    self.ksp.view()
            self.reason = self.ksp.getConvergedReason()
            self.last_iterations = self.ksp.getIterationNumber()
            self.last_pres = self.ksp.getResidualNorm() 
            self.final_res = self.true_residual(rhs,sol)
            
        time_stop = process_time()
        
        
        self.solved += 1
        self.cumulative_iterations += self.last_iterations 
        self.last_cpu = time_stop - time_start

    def true_residual(self,rhs,sol):
        self.A.petscmat.mult(sol,self._tmp)
        self._tmp.aypx(-1,rhs)
        return self._tmp.norm()/rhs.norm()

    def info(self):
        return info_ksp(self.ksp)

    def apply(self,pc,x,y):
        self.solve(x,y)


def info_ksp(ksp):
    """
    Return a one-line string with main info about last linear system solved
    """
    reason = ksp.getConvergedReason()
    last_iterations = ksp.getIterationNumber()
    last_pres = ksp.getResidualNorm() 
    h=ksp.getConvergenceHistory()
    A = ksp.getOperators()[0]
    rhs = ksp.getRhs()
    sol = ksp.getSolution()
    temp = A.createVecLeft()
    A.mult(sol,temp)
    temp.aypx(-1,rhs)
    rhs_norm = rhs.norm()
    real_res = temp.norm()/rhs_norm
    residuals = h[-(last_iterations+1):]  
    return str(ksp.getOptionsPrefix())+f' {solving_utils.KSPReasons[reason]} {last_iterations} {residuals[0]:.1e} {residuals[-1]:.1e} res={real_res:.1e} |rhs|={rhs_norm:.1e} pres {last_pres:.1e}' 

def attach_nullspace(A, nullspace = None, transpose_nullspace = None, near_nullspace = None):
    """
    Attach a nullspace information to the PETSCmatrix A.

    """
    if (nullspace is not None) :
        if isinstance(nullspace,MixedVectorSpaceBasis):
            if (nullspace._nullspace is None):
                nullspace._build_monolithic_basis()
            A.setNullSpace(nullspace._nullspace)
        else:
            A.setNullSpace(nullspace.nullspace(comm=A.comm))

    if (transpose_nullspace is not None) :
        if isinstance(transpose_nullspace,MixedVectorSpaceBasis):
            if (transpose_nullspace._nullspace is None):
                transpose_nullspace._build_monolithic_basis()
            A.setTransposeNullSpace(transpose_nullspace._nullspace)
        else:
            A.setTransposeNullSpace(transpose_nullspace.nullspace(comm=A.comm))

    if (near_nullspace is not None) :
        if isinstance(near_nullspace,MixedVectorSpaceBasis):
            if (near_nullspace._nullspace is None):
                near_nullspace._build_monolithic_basis()
            A.setNearNullSpace(near_nullspace._nullspace)
        else:
            A.setNearNullSpace(near_nullspace.nullspace(comm=A.comm))
    

        
    if (near_nullspace is not None):
        if ( isinstance(near_nullspace,MixedVectorSpaceBasis)
            and near_nullspace._nullspace is None):
            near_nullspace._build_monolithic_basis()
        A.setNearNullSpace(near_nullspace._nullspace)
    


class LinSolMatrix(OptionsManager):

    DEFAULT_KSP_PARAMETERS = solving_utils.DEFAULT_KSP_PARAMETERS

    @PETSc.Log.EventDecorator()
    def __init__(self, A, function_space, solver_parameters,
                nullspace = None, transpose_nullspace= None, near_nullspace= None, 
                P=None,  Prec=None, appctx=None,
                options_prefix=None):
        """A linear solver for assembled systems (Ax = b).

        :arg A: a :class:`~.MatrixBase` (the operator).
        :arg sol: a :class:`~.Function containg the state varaible used to
            assemble the operator. It is used to assemble precondtioner 
            since is passed as "state" variable (see also).
        :arg appctx: Any extra information used in the assembler.  
        :arg P: an optional :class:`~.MatrixBase` to construct any
             preconditioner from; if none is supplied ``A`` is
             used to construct the preconditioner.
        :kwarg parameters: (optional) dict of solver parameters.
        :kwarg nullspace: an optional :class:`~.VectorSpaceBasis` (or
            :class:`~.MixedVectorSpaceBasis` spanning the null space
            of the operator.
        :kwarg transpose_nullspace: as for the nullspace, but used to
               make the right hand side consistent.
        :kwarg near_nullspace: as for the nullspace, but used to set
               the near nullpace.
        :kwarg options_prefix: an optional prefix used to distinguish
               PETSc options.  If not provided a unique prefix will be
               created.  Use this option if you want to pass options
               to the solver from the command line in addition to
               through the ``solver_parameters`` dict.

        .. note::

          Any boundary conditions for this solve *must* have been
          applied when assembling the operator.
        """
        if not isinstance(A, PETSc.Mat):
            raise TypeError("Provided operator is a '%s', not a PetscMatrix" % type(A).__name__)
        if P is not None and not isinstance(P, PETSc.Mat):
            raise TypeError("Provided preconditioner is a '%s', not a PetscMatrix" % type(P).__name__)

        

        self.A = A
        self.comm = A.comm
        self.P = P if P is not None else A

        # store the space of functions on which the operator is defined
        self.function_space = function_space
        
        
        # Set up parameters mixin
        super().__init__(solver_parameters, options_prefix)
        
        
        
        self.A.setOptionsPrefix(self.options_prefix)
        self.P.setOptionsPrefix(self.options_prefix)

        
        # If preconditioning matrix is matrix-free, then default to jacobi
        self.ksp = PETSc.KSP().create(comm=self.comm)

        # DM provides fieldsplits (but not operators)
        self.ksp.setDM(function_space.dm)
        self.ksp.setDMActive(False)


        # attach nullspace information
        attach_nullspace(self.A,
        nullspace = nullspace, 
        transpose_nullspace=transpose_nullspace, 
        near_nullspace=near_nullspace)
        if P is not None:
            attach_nullspace(self.P,
            nullspace = nullspace,
            transpose_nullspace=transpose_nullspace,
            near_nullspace=near_nullspace)

        self.nullspace = nullspace
        self.transpose_nullspace = transpose_nullspace
        self.near_nullspace = near_nullspace

        # this operation attach the nullspace information to the IS
        # 
        self.kernel_in_init = False
        if (self.kernel_in_init):
            if len(self.function_space) > 1 and self.nullspace is not None:
                self.nullspace._apply(self.trial_space.dof_dset.field_ises)
            if len(self.function_space) > 1 and self.transpose_nullspace is not None:
                self.transpose_nullspace._apply(self.test_space.dof_dset.field_ises,
                                                transpose=True)
            if len(self.function_space) > 1 and self.near_nullspace is not None:
                self.near_nullspace._apply(self.trial_space.dof_dset.field_ises, near=True)




        # Operator setting must come after null space has been
        # applied
        self.ksp.setOperators(A=self.A, P=self.P)
        # Set from options now (we're not allowed to change parameters
        # anyway).
        if (Prec is not None):
            if (P is not None):
                raisewarning("Both P and Prec are not None, P will be ignored")
            self.ksp.setPC(Prec)
        
        if appctx is None:
            appctx = {}



        self.set_from_options(self.ksp)
        self.options_prefix = self.ksp.getOptionsPrefix()

        # 
        self._tmp = self.A.createVecLeft()
        self.reason = 0
        self.solved = 0
        self.cumulative_iterations = 0

    
    @PETSc.Log.EventDecorator()
    def solve(self, b, x):
        if not self.ksp.getInitialGuessNonzero():
            x.set(0.0)

        if (not self.kernel_in_init):
            if len(self.function_space) > 1 and self.nullspace is not None:
                self.nullspace._apply(self.function_space.dof_dset.field_ises)
            if len(self.function_space) > 1 and self.transpose_nullspace is not None:
                self.transpose_nullspace._apply(self.function_space.dof_dset.field_ises,
                                                transpose=True)
            if len(self.function_space) > 1 and self.near_nullspace is not None:
                self.near_nullspace._apply(self.function_space.dof_dset.field_ises, near=True)

        time_start = process_time()
        with self.inserted_options(), dmhooks.add_hooks(self.ksp.dm, self):
            self.initial_res = self.true_residual(b,x)
            self.ksp.solve(b, x)
            self.final_res = self.true_residual(b,x)
            self.reason = self.ksp.getConvergedReason()
            self.last_iterations = self.ksp.getIterationNumber()
        time_stop = process_time()
             
        self.solved += 1
        self.cumulative_iterations += self.last_iterations
        self.last_cpu = time_stop - time_start        

    def true_residual(self,rhs,sol):
        self.A.mult(sol,self._tmp)
        self._tmp.aypx(-1,rhs)
        bnorm = rhs.norm()
        if bnorm > 0.0:
            return self._tmp.norm()/bnorm
        else:
            return 0.0

    def info(self):
        return str(self.options_prefix)+f' | {solving_utils.KSPReasons[self.reason]}: {self.ksp.getIterationNumber()} {self.initial_res:.2e} {self.final_res:.2e} | {self.ksp.getResidualNorm():.2e}'
        
    def apply(self,x,y):
        self.solve(x,y)
    
    def mult(self,x,y):
        self.solve(x,y)

        