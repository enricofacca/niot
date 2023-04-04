from itertools import chain
from contextlib import ExitStack
from firedrake import dmhooks
from petsc4py import PETSc
from firedrake import FunctionSpace



"""
Define a variable to store the reason of the SNES solver
"""
def _make_reasons(reasons):
    return dict([(getattr(reasons, r), r)
                 for r in dir(reasons) if not r.startswith('_')])
SNESReasons = _make_reasons(PETSc.SNES.ConvergedReason())


def my_solve(solver):
    """
    Call the solve methodof the NonLinearVariational class without invoking 
    check_snes_convergence(self.snes) that would raise an error that will stop the program.
    Copy from https://github.com/firedrakeproject/firedrake/blob/master/firedrake/variational_solver.py
    """
    
    
    # Make sure the DM has this solver's callback functions
    solver._ctx.set_function(solver.snes)
    solver._ctx.set_jacobian(solver.snes)

    # Make sure appcontext is attached to the DM before we solve.
    dm = solver.snes.getDM()
    for dbc in solver._problem.dirichlet_bcs():
        dbc.apply(solver._problem.u)

    work = solver._work
    with solver._problem.u.dat.vec as u:
        u.copy(work)
        with ExitStack() as stack:
            # Ensure options database has full set of options (so monitors
            # work right)
            for ctx in chain((solver.inserted_options(), dmhooks.add_hooks(dm, solver, appctx=solver._ctx)),
                                solver._transfer_operators):
                stack.enter_context(ctx)
            solver.snes.solve(None, work)
        work.copy(u)
    solver._setup = True

def msg_bounds(vec,label):
    min = vec.min()[1]
    max = vec.max()[1]
    return "".join([f'{min:2.1e}','<=',label,'<=',f'{max:2.1e}'])


def my_newton(flag, f_norm, tol=1e-6, max_iter=10):
    """
    Create a reverse communication Newton solver.
    Flag 1: compute residual
    Flag 2: compute Newton step compute Jacobian
    Flag 3: solve linear system


    """
    if (flag == 0):
        return 1

    if (flag == 1):
        # Create a krylov solver
        snes = PETSc.SNES().create()
        snes.setType('newtonls')
        snes.setTolerances(max_it=max_iter, atol=tol, rtol=tol)
        snes.setConvergenceHistory()
        snes.setFromOptions()
        return snes

def getFunctionSpace(fun, index):
    """
    Return the function space (not indexed) of a function in a mixed space
    TODO: It works but maybe thare are better ways to do it 
    """
    subfun = fun.sub(index)
    return FunctionSpace(subfun.function_space().mesh(),subfun.ufl_element().family(),subfun.ufl_element().degree())