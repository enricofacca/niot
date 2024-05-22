from firedrake import *
import numpy as np


def compute_mass_Neumann(mesh, Neumann):
    """
    Compute Neumann mass.
    mesh: mesh
    Neumann: list of tuples (fun, region). 
            fun maps into the same topoogical
            dimension as the mesh.
    return: mass_Neumann
    """
    mass_Neumann = 0.0
    # TODO: implement this
    #n = FacetNormal(mesh)
    #for bnd in Neumann:
    #    fun = bnd[0]
    #    region = bnd[1]
    #    mass_Neumann += assemble(inner(fun,n)*dS(region))
    return mass_Neumann


def balance(source, sink, Neumann=None, tolerance_imbalance=1e-12, mode='scale_sink'):
    """
    Balance the source and sink terms.
    """
    mass_source = assemble(source*dx)
    mass_sink = assemble(sink*dx)
    if Neumann is None:
        mass_Neumann = 0.0
    else:   
        mass_Neumann = compute_mass_Neumann(source.function_space().mesh(),Neumann)
    imbalance = mass_source - mass_sink + mass_Neumann 
    if abs(imbalance) > tolerance_imbalance :
        if mode == 'scale_sink':
            sink *= (mass_source+mass_Neumann) / mass_sink
        elif mode == 'scale_source':
            source = (mass_sink-mass_Neumann) / mass_source
        else:
            raise ValueError('Source and sink terms are not balanced')

class OTPInputs():
    """
    Class contatins the spatial inputs for the optimal transport problem.
    """
    def __init__(self,
                  source, sink, 
                  Neumann=None,
                  Dirichlet=None, 
                  kappa=1.0, 
                  tolerance_imbalance=1e-12):
        
        # store mesh
        self.mesh = source.function_space().mesh()
        if Neumann is None:
            self.Neumann = [[as_vector(np.zeros(self.mesh.topological_dimension()-1)),"on_boundary"]]
        else:
            self.Neumann = Neumann

        # create softcopy of the inputs
        self.source = source
        self.sink = sink
        self.Dirichlet = Dirichlet
        self.kappa = kappa

class BranchedTransportProblem():
    """
    Classc containg the optimal transport problem.
    """
    def __init__(self, source, sink, 
                  Neumann=None,
                  Dirichlet=None, 
                  kappa=1.0, 
                  tolerance_imbalance=1e-11,
                  gamma=0.5):
        # store spatial info
        self.mesh = source.function_space().mesh()
        if Neumann is None:
            self.Neumann = [[as_vector(np.zeros(self.mesh.topological_dimension()-1)),"on_boundary"]]
        else:
            self.Neumann = Neumann

        # create softcopy of the inputs
        self.source = source
        self.sink = sink
        self.Dirichlet = Dirichlet
        if isinstance(kappa,float):
            R = FunctionSpace(self.mesh, 'R', 0)
            self.kappa = Function(R,val=1,name='kappa')
        else:
            self.kappa = kappa
        
        # Check data consistency
        mass_source = assemble(self.source * dx)
        mass_sink = assemble(self.sink * dx)
        mass_Neumann = compute_mass_Neumann(self.mesh, self.Neumann)
        mass_balance = (mass_source - mass_sink + mass_Neumann)/max(mass_source,mass_sink)
        if ( self.Dirichlet is None 
            and abs(mass_balance) > tolerance_imbalance):
            raise ValueError(f'Source, sink, and Neumann terms are not balanced {mass_balance}')
        
        # branched transport exponent
        self.gamma = gamma
    
