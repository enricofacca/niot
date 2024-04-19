from itertools import chain
from contextlib import ExitStack
from firedrake import dmhooks
from petsc4py import PETSc
from firedrake import FunctionSpace
from firedrake import Citations
from firedrake import Function, TestFunction, TrialFunction
from firedrake import assemble
from firedrake import dx
from firedrake import conditional
from firedrake import File
from firedrake import solving_utils

from firedrake import sqrt, jump, avg, conditional, gt

from pyadjoint import Block
from pyadjoint.overloaded_function import overload_function

import firedrake.adjoint as fire_adj

from firedrake.__future__ import interpolate


from firedrake import COMM_WORLD, COMM_SELF

import re
import os
import numpy as np





"""
Define a variable to store the reason of the SNES solver
"""
def _make_reasons(reasons):
    return dict([(getattr(reasons, r), r)
                 for r in dir(reasons) if not r.startswith('_')])
SNESReasons = _make_reasons(PETSc.SNES.ConvergedReason())




def msg_bounds(vec,label):
    min = vec.min()[1]
    max = vec.max()[1]
    return "".join([f'{min:2.1e}','<=',label,'<=',f'{max:2.1e}'])

def save2pvd(functions,filename):
    """
    Save a list of functions to a pvd file.
    The associated vtu file is renamed to filename.vtu
    While standard Firedrake pvd files have the extension _0.vtu
    """
    # check exension
    if (filename[-4:] != '.pvd'):
        raise ValueError("The filename must have extension .pvd")

    if (type(functions) != list):
        functions = [functions]
    out_file = File(filename)
    out_file.write(*functions)

    # get directory 
    directory = os.path.dirname(os.path.abspath(filename))
    filename = os.path.basename(filename)
    filename = filename[:-4]


    comm = functions[0].function_space().mesh().comm
    if comm.size == 1:
        firedrake_vtu_name = f'{directory}/{filename}/{filename}_0.vtu'
        new_vtu_name = f'{directory}/{filename}.vtu'
        print(new_vtu_name)
        try:
            os.rename(firedrake_vtu_name,new_vtu_name)
        except:
            print(f"Error renaming vtu file{firedrake_vtu_name}")
            pass
    else:
        if comm.rank == 0:
            firedrake_vtu_name = filename+'_0.pvtu'
            new_vtu_name = filename+'.pvtu'
            try:
                os.rename(firedrake_vtu_name,new_vtu_name)
            except:
                print(f"Error renaming vtu file{firedrake_vtu_name}")
            pass
            
            with open(new_vtu_name, "r") as sources:
                lines = sources.readlines()
            with open(new_vtu_name,'w') as sources:
                for line in lines:
                    sources.write(re.sub(r'_0_', '_', line))
            
            with open(filename+'.pvd', "r") as sources:
                lines = sources.readlines()
            with open(filename+'.pvd','w') as sources:
                for line in lines:
                    sources.write(re.sub(r'_0.', '.', line))    

        
        # each processor has a vtu file
        firedrake_vtu_name = f'{filename}_0_{comm.rank}.vtu'
        new_vtu_name = f'{filename}_{comm.rank}.vtu'
        try:
            os.rename(firedrake_vtu_name,new_vtu_name)
        except:
            print(f"Error renaming vtu file{firedrake_vtu_name}")
            pass
        
        comm.Barrier()



def get_subfunction(function, index):
    """
    Return the subfunction of a function in a mixed space
    """
    functions = function.subfunctions()
    function_spaces = function.function_space()
    function = Function(function_spaces[index])
    function.assign(functions[index])
    return function


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
    temp = assemble(interpolate(conditional(func>lower_bound,func,lower_bound),func.function_space()))
    func.assign(temp)



# find all occurrences of a substring in a string

#find the position of all @ in the text
def include_citations(filename):
    """
    Include bibtex entries in filename 
    to the citations system of Firedrake and Petsc
    (see https://www.firedrakeproject.org/citations.html)
    
    Args: 
        filename (str): name of the file containing the citations
    
    Returns:
        empty
    
    The code is hard written, but avoid using 
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
        Citations().add(cit[0],cit[1]+'\n')


def y_branch(coordinates, masses, alpha):
    """
    Given a forcing term with 3 Dirac masses and the branch exponent alpha,
    returns the optimal topology of the network and the points of the network.
    See 
    @article{xia2003optimal,
    title={Optimal paths related to transport problems},
    author={Xia, Qinglan},
    journal={Communications in Contemporary Mathematics},
    volume={5},
    number={02},
    pages={251--279},
    year={2003},
    publisher={World Scientific}
    }

    Args:
    2dcordinates:list of the coordinate of the forcing term
    masses: the values of the forcing term (they must sum to zero)
    alpha: branching exponent

    return:
    v: topology of the optimal network
    p: points in the optimal network( if the y-shape is optimal and there is
       a biforcation node, the node coordinate are appended to 2dcoordinates)  
    """

    points=np.array(coordinates)
    masses=abs(np.array(masses))

    print(points)

    
    # there are 3 masses combinations since sum(masses)=0
    # +, -, -
    # +, +, - => -, -, + (reverse flow)
    # -, +, + => +, -, - (reverse flow)
    # -, -, + 
    # but everthing can be reduced to the first configuration
    # where one source sends to two sink

    coord_O = points[0,:]
    coord_Q = points[1,:]
    coord_P = points[2,:]


    m_O=abs(masses[0])
    m_Q=abs(masses[1])
    m_P=abs(masses[2])

    OP=coord_P-coord_O
    OQ=coord_Q-coord_O
    QP=coord_P-coord_Q

    OQP=np.arccos( np.dot(-OQ, QP)/ ( np.sqrt(np.dot(OQ,OQ)) * np.sqrt(np.dot(QP,QP) )))
    QPO=np.arccos( np.dot(-OP,-QP)/ ( np.sqrt(np.dot(OP,OP)) * np.sqrt(np.dot(QP,QP) )))
    POQ=np.arccos( np.dot( OQ, OP)/ ( np.sqrt(np.dot(OQ,OQ)) * np.sqrt(np.dot(OP,OP) )))
    
    k_1=(m_P/m_O)**(2*alpha)
    k_2=(m_Q/m_O)**(2*alpha)

    theta_1=np.arccos( (k_2-k_1-1)/(2*np.sqrt(k_1))     )
    theta_2=np.arccos( (k_1-k_2-1)/(2*np.sqrt(k_2))     )
    theta_3=np.arccos( (1-k_1-k_2)/(2*np.sqrt(k_1*k_2)) )
    
    v=[];
    if (POQ>=theta_3):
        B_opt=coord_O
        v.append([0,1])
        v.append([0,2])
        p = cp(points)
    elif ( (OQP>=theta_1) & (POQ<theta_3)):
        B_opt=coord_Q
        v[:,0]=[0,1]
        v[:,1]=[1,2]
        p = cp(points)
    elif ( (QPO>=theta_2) & (POQ<theta_3)):
        B_opt=coord_P
        v[:,0]=[0,2]
        v[:,1]=[1,2]
        p = cp(points)
    else:
        QM=np.dot(OP,OQ)/np.dot(OP,OP) * OP - OQ
        PH=np.dot(OP,OQ)/np.dot(OQ,OQ) * OQ - OP
        
        R=(coord_O+coord_P)/2.0 - (np.cos(theta_1)/np.sin(theta_1))/2.0 * np.sqrt(np.dot(OP,OP)/ np.dot(QM,QM) )* QM
        S=(coord_O+coord_Q)/2.0 - (np.cos(theta_2)/np.sin(theta_2))/2.0 * np.sqrt(np.dot(OQ,OQ)/ np.dot(PH,PH) ) * PH
        RO=coord_O-R
        RS=S-R
        
        B_opt=2*( (1-np.dot(RO,RS) / np.dot(RS,RS)) *R + np.dot(RO,RS)/np.dot(RS,RS)*S)-coord_O
        
        #p_B=tuple([tuple(i) for i in B_opt])
        #p_B=[i for i in enumerate(B_opt)]
        p_B=B_opt.tolist()

        p = np.zeros([4,2])
        p[0:3,:] = points
        p[3,:] = p_B
        v.append([0,3])
        v.append([1,3])
        v.append([2,3])
        
    return p,v


def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)


def nested_get(dic, keys, default=None):
    """
    Get the value from a nested dictionary
    Args:
        dic (dict): dictionary
        keys (list): list of keys
        default (any): default value if the key is not found
    Returns:
        value (any): value of the nested dictionary (it can be a dictionary)
    
    Example:
        dic = {'a':{'b':{'c':1}}}
        keys = ['a','b','c']
        value = nested_get(dic,keys)
        print(value)
        >>> 1
    """
    if type(keys) != list:
        keys = [keys]
    d = dic
    for key in keys[:-1]:
        if key in d:
            d = d[key]
    try:        
        value = d[keys[-1]]
    except:
        print(keys)
        if default is None:
            raise KeyError
        value = default
    return value


def nested_set(dic, keys, value, create_missing=False):
    """
    Get the value from a nested dictionary
    Args:
        dic (dict): dictionary
        keys (list): list of keys
        value (any): value to be set
        create_missing (bool): if True, create the missing keys
    Returns:
        None (it modifies the dictionary in place)
    Example:
        dic = {'a':{'b':{'c':1}}}
        keys = ['a','b','c']
        nested_set(dic,keys,2)
        print(dic)
        >>> {'a': {'b': {'c': 2}}}
    """
    d = dic
    if type(keys) != list:
        keys = [keys]
    for key in keys[:-1]:
        if key in d:
            d = d[key]
        elif create_missing:
            d = d.setdefault(key, {})
        else:
            print(f'Key {key} not found in dictionary {value}')
            raise KeyError 
        
    if keys[-1] in d or create_missing:
        d[keys[-1]] = value
    else:
        print(f'Key {keys[-1]} not found in dictionary {value}')
        raise KeyError 
    
    return dic


def delta_h(space):
    # check if mesh is simplex
    mesh = space.mesh()
    if mesh.ufl_cell().is_simplex():
        raise ValueError('implemented for cartesian grids')   
    
    if mesh.geometric_dimension() == 2:
        # try:
        #     # try to get this information from the mesh 
        #     hx = mesh.Lx/mesh.nx
        #     hy = mesh.Ly/mesh.ny
        #     # check if the values are close numerically with numpy
        #     if np.isclose(hx,hy):
        #         delta_h = hx
        #     else:
        #         raise ValueError('hx and hy are not close numerically')
        # except:
        x,y = mesh.coordinates
        x_func = assemble(interpolate(x, space))
        y_func = assemble(interpolate(y, space))
        delta_h = sqrt(jump(x_func)**2 + jump(y_func)**2)
    elif mesh.geometric_dimension() == 3:
        x,y,z = mesh.coordinates
        x_func = assemble(interpolate(x, space))
        y_func = assemble(interpolate(y, space))
        z_func = assemble(interpolate(z, space))
        delta_h = sqrt(jump(x_func)**2 
                            + jump(y_func)**2 
                            + jump(z_func)**2)
    return delta_h

def cell2face_map(fun, approach="harmonic_mean"):
    """
    Compute a face value from a cell value
    """
    if approach == 'arithmetic_mean':
        return (fun('+') + fun('-')) / 2
    elif approach == 'harmonic_mean':
        avg_fun = avg(fun)
        return conditional( gt(avg_fun, 0.0), fun('+') * fun('-') / avg_fun, 0.0)          
    else:
        raise ValueError('Wrong approach passed. Only arithmetic_mean or harmonic_mean are implemented')
