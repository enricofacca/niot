from itertools import chain
from contextlib import ExitStack
from firedrake import dmhooks
from petsc4py import PETSc
from firedrake import FunctionSpace
from firedrake import Citations
from firedrake import Function
from firedrake import conditional


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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def color(color,str):
    if (color == 'blue'):
        return f'{bcolors.OKBLUE}{str}{bcolors.ENDC}'
    elif (color == 'green'):
        return f'{bcolors.OKGREEN}{str}{bcolors.ENDC}'
    elif (color == 'red'):
        return f'{bcolors.FAIL}{str}{bcolors.ENDC}'
    elif (color == 'yellow'):
        return f'{bcolors.WARNING}{str}{bcolors.ENDC}'
    elif (color == 'cyan'):
        return f'{bcolors.OKCYAN}{str}{bcolors.ENDC}'
    elif (color == 'magenta'):
        return f'{bcolors.HEADER}{str}{bcolors.ENDC}'
    elif (color == 'bold'):
        return f'{bcolors.BOLD}{str}{bcolors.ENDC}'
    elif (color == 'underline'):
        return f'{bcolors.UNDERLINE}{str}{bcolors.ENDC}'
    else:
        return str

def threshold_from_below(func, lower_bound):
    """
    Limit a function from below
    Args:
        func (Function): function to be limited from below (in place)
        lower_bound (float): lower bound
    """
    temp = Function(func.function_space())
    temp.interpolate(conditional(func>lower_bound,func,lower_bound))
    func.assign(temp)


import re

# find all occurrences of a substring in a string

#find the position of all @ in the text
def include_citations(filename):
    """
    Include the in filename 
    to the citations system of Firedrake and Petsc
    (see https://www.firedrakeproject.org/citations.html)
    
    Args: 
        filename (str): name of the file containing the citations
    
    Returns:
        empty
    
    The code uses brute force code, but avoid using 
    libraries that are not in the standard python distribution.
    """

    # read text from citations.bib file
    s = open(filename, 'r').read()

    # function to get the position of all occurrences of a substring in a string
    def find_all(a_str, sub):
        start = 0
        while True:
            start = a_str.find(sub, start)
            if start == -1: return
            yield start
            start += len(sub) # use start += 1 to find overlapping matches

    # find the position of all @ in the text
    position=list(find_all(s,'@'))
    position.append(len(s))

    # split the text in a list of strings 
    # and get keywords and content of each bibtex entry
    entries = []
    for i in range(len(position)-1):
        bib=s[position[i]:position[i+1]]

        begin=bib.find('{')
        end=bib.find(',')
        keyword=bib[begin+1:end]

        end=bib.rfind('}')
        content=bib[0:end+1].strip()
        
        entries.append([keyword,content])

    for cit in entries:
        print("Adding citation: "+cit[0])
        print(cit[1])
        Citations().add(cit[0],cit[1]+'\n')