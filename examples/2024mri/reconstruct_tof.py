import sys
import glob
import os
from copy import deepcopy as cp

import numpy as np
from niot import image2dat as i2d
from niot import utilities
from niot import optimal_transport as ot
from niot import NiotSolver
from memory_profiler import profile


from firedrake import *
from scipy.ndimage import zoom
import time
from firedrake import VTKFile as File

from firedrake.petsc import PETSc

import sys

import itertools
import argparse
import nibabel

import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(formatter={'float': '{:0.2e}'.format})


def build_meshes_from_numpy(data, mesh_type="simplicial",lengths=None,label_boundary=False):
    if lengths is None:
        lengths = [1.0,data.shape[1]/data.shape[0],data.shape[2]/data.shape[0]]
    
    mesh = i2d.build_mesh_from_numpy(data, mesh_type=mesh_type,lengths=lengths,label_boundary=label_boundary)
    if mesh_type == "simplicial":
        cartesian_mesh = i2d.cartesian_grid_3d(data.shape,lengths)
    else:
        cartesian_mesh = mesh
    return mesh, cartesian_mesh

def load_data(field, coarseness, data_folder="../../../mri/",mesh_type="simplicial"):

    data = np.load(f'{data_folder}/{field}.npy')

    

    PETSc.Sys.Print(f"Data shape: {data.shape}") 


    if coarseness > 1:
        PETSc.Sys.Print('coarsening image')
        data = zoom(data, (1/coarseness,1/coarseness,1/coarseness), order=0)
        PETSc.Sys.Print(data.shape)


    print(f"Data shape: {data.shape}")
    
    # create mesh
    PETSc.Sys.Print('building mesh')
    start = time.time()
    lengths = [1.0,data.shape[1]/data.shape[0],data.shape[2]/data.shape[0]]
    mesh = i2d.build_mesh_from_numpy(data, mesh_type=mesh_type,lengths=lengths,label_boundary=True)
    if mesh_type == "simplicial":
        cartesian_mesh = i2d.cartesian_grid_3d(data.shape,lengths)
    else:
        cartesian_mesh = mesh
    
    #mesh = Mesh("Cube-03.msh")
    #mesh.name = 'mesh'
    #mesh.init()
    end = time.time()
    PETSc.Sys.Print(f'mesh built {end-start}s')

    # convert to firedrake
    PETSc.Sys.Print('converting image to firedrake')
    start = time.time()
    tof_fire = i2d.numpy2firedrake(mesh, data, name="TOF")
    end = time.time()
    PETSc.Sys.Print(f'image converted {end-start}s')

    return tof_fire, cartesian_mesh
    

def btp_inputs(tof_fire):
    mesh = tof_fire.function_space().mesh()
    
    threshold_domain = 150
    threshold_network = 250
    
    
    # convert to firedrake
    DG0 = FunctionSpace(mesh,"DG",0)
    source = Function(DG0,name="source")
    sink = Function(DG0,name="sink")
    x,y,z = SpatialCoordinate(mesh)
    #source.interpolate(conditional(z<15,1,0)*conditional(tof_fire>250,1,0))
    source.assign(0.0)
    # above 150 define the approximate support of absortion
    # above 250 is remove beacuse where we know we have blood vessels
    sink.interpolate(conditional(tof_fire>threshold_domain,1,0) 
                     * conditional(tof_fire<threshold_network,1,0))


    mass_source = assemble(source*dx)
    mass_sink = assemble(sink*dx)

    #source /= mass_source
    sink /= mass_sink


    PETSc.Sys.Print(f"{mass_source=:.2e} {mass_sink=:.2e}")


    
    
    corrupted = Function(DG0,name="corrupted")
    corrupted.interpolate(conditional(tof_fire> threshold_network,1,0)* tof_fire)

    return source, sink, corrupted

    


def labels(fem,
           gamma,wd,wr,
           corrupted_as_initial_guess,
           confidence,
           tdens2image, 
           method):
    label= [
        f'fem{fem}',
        f'gamma{gamma:.1e}',
        f'wd{wd:.1e}',
        f'wr{wr:.1e}',
        f'ini{corrupted_as_initial_guess:.1e}',
        f'conf{confidence}']
    if tdens2image['type'] == 'identity':
        label.append(f'mu2iidentity')
    elif tdens2image['type'] == 'heat':
        label.append(f"mu2iheat{tdens2image['sigma']:.1e}")
    elif tdens2image['type'] == 'pm':
        label.append(f"mu2ipm{tdens2image['sigma']:.1e}")
    else:
        raise ValueError(f'Unknown tdens2image {tdens2image}')
    label.append(f"scaling{tdens2image['scaling']:.1e}")  
    if method is not None:
        if method == 'tdens_mirror_descent_explicit':
            short_method = 'te'
        elif method == 'tdens_mirror_descent_semi_implicit':
            short_method = 'tsi'
        elif method == 'gfvar_gradient_descent_explicit':
            short_method = 'ge'
        elif method == 'gfvar_gradient_descent_semi_implicit':
            short_method = 'gsi'
        elif method == 'tdens_logarithmic_barrier':
            short_method = 'tlb'
        else:
            raise ValueError(f'Unknown method {method}')
    label.append(f'method{short_method}')
    return label

def setup_problem(source, sink, inlet):
    
    mesh = source.function_space().mesh()
    
    # Define the branched transport problem
    gamma=0.5
    
    inlet_pressure = Function(inlet.function_space())
    inlet_pressure.assign(0.0)
    weak_Dirichlet = [(Constant(0.0), ds_b, inlet)]
    btp = ot.BranchedTransportProblem(source, sink, gamma=gamma, 
                                      Dirichlet = None,
                                      weak_Dirichlet = weak_Dirichlet)

    return btp


def setup_solver(btp, 
                 corrupted, 
                 fem="DG0DG0",
                 gamma=0.5, 
                 wd=1e-2,
                 wr=0.0, 
                 corrupted_as_initial_guess=0,
                 confidence='ONE',
                 tdens2image={
                     'type':'identity',
                     'scaling':1e0
                 },
                 method='tdens_mirror_descent_explicit'  ,
                 directory='out/',
                 labels_problem=['unnamed'],
                 comm=COMM_SELF,
                 verbose=0):


   
    mesh = btp.source.function_space().mesh()

    confidence = Function(btp.source.function_space())
    confidence.assign(1.0)


    

    niot_solver = NiotSolver(btp, 
                             corrupted,  
                             confidence=confidence, 
                             spaces = fem,
                             cell2face = 'harmonic_mean',
                             setup=False,
                             )


    # Setup the solver's parameters

    # inpainting
    niot_solver.ctrl_set('discrepancy_weight', wd)
    niot_solver.ctrl_set('regularization_weight', wr)
    niot_solver.ctrl_set(['tdens2image'], tdens2image)

    # optimization
    niot_solver.ctrl_set('optimization_tol', 1e-5)
    niot_solver.ctrl_set('constraint_tol', 1e-5)
    niot_solver.ctrl_set('max_iter', 1)
    niot_solver.ctrl_set('max_restart', 4)
    niot_solver.ctrl_set('verbose', 2)

     # time discretization
    if method is None:
        method = 'tdens_mirror_descent_explicit'
    niot_solver.ctrl_set(['dmk','type'], method)
    if 'tdens' in method:
        if method == 'tdens_logarithmic_barrier':
            pass
        else:
            niot_solver.ctrl_set(['dmk',method,'gradient_scaling'], 'dmk')
        

    # time step
    deltat_control = {
        'type': 'adaptive2',
        'lower_bound': 1e-13,
        'upper_bound': 5e-2,
        'expansion': 1.1,
        'contraction': 0.5,
    }
    niot_solver.ctrl_set(['dmk',method,'deltat'], deltat_control)
    
    return niot_solver



   
#
# common setup
#
fems = ["DG0DG0"]
wr = [0.0]
method = [
    "tdens_mirror_descent_explicit",
]

def figure1():
    #
    # Combinations producting the data for Figure 2
    #
    gamma = [0.5] # 
    wd = [1e-1,1e0]  # set the discrepancy to zero
    ini = [0]
    # the following are not influent since wd=weight discrepancy is zero
    conf = ["ONE"]
    maps = [
        #{"type": "identity", "scaling": 1/20},
        {"type": "identity", "scaling": 1/10},
    ]
    parameters = [
        fems,
        gamma,
        wd,
        wr,
        ini,
        conf,
        maps,
        method,
    ]
    combinations = list(itertools.product(*parameters))

    return combinations


def downsample(data,coarseness,mode="zoom"):
    """
    Downsample data by a factor of coarseness
    Args:
    data: numpy 2d or 3d array
    coarseness: int
    mode: str
        "zoom": uses scipy.ndimage.zoom
        "max" : uses max of neighbours cells
    Returns:
        data: numpy 2d or 3d array
    """
    if coarseness <= 1:
        return data
      
    PETSc.Sys.Print('coarsening image')
    if mode == "zoom":
        factors = tuple([1 if n==1 else 1/coarseness for n in data.shape])
        data = zoom(data, factors, order=0)
        PETSc.Sys.Print(data.shape)
        return data

    elif mode == "max":
        # the extra boundary required
        pad_shape = tuple([(0, int(coarseness*np.ceil(n/coarseness)-n)) for n in data.shape])

        # pad with -inf
        data = np.pad(data, pad_shape, mode='constant', constant_values=-np.inf)
        

        # coarseness = 2
        # (nx//2,2,ny//2,2,nz//2,2)
        dim = data.ndim
        reshaped_shape = sum([[n//coarseness, coarseness] for n in data.shape],[])
        
        
        # reshape gathering neighbours cells and get the max
        extra_dim = tuple([2*i+1 for i in range(data.ndim)])
        
        data = data.reshape(reshaped_shape).max(axis=extra_dim)
        
        return data

def indices_restrict(data, lengths, xyz_bounds):
    """ 
    Assuming that LX=1
    """
    indices_bounds = []
    for axis_index in range(len(data.shape)):
        n_axis = data.shape[axis_index]
        len_axis = lengths[axis_index]
        if xyz_bounds[axis_index] is None:
            indices_bounds.append([0,n_axis])
        else:
            lower, upper = xyz_bounds[axis_index]
            indices_bound = [max(0,int(lower/len_axis*n_axis)),min(int(upper/len_axis*n_axis),n_axis)]
            indices_bounds.append( indices_bound)
                

    return np.array(indices_bounds)

def restrict(data, indices_bounds):
    """ 
    Assuming that LX=1
    """
    data = data[indices_bounds[0][0]:indices_bounds[0][1],
                indices_bounds[1][0]:indices_bounds[1][1],
                indices_bounds[2][0]:indices_bounds[2][1]]
    data = np.ascontiguousarray(data)
    return data




def select_slice():
    # get bottom slice of tof
    tof_bottom_np = np.flipud(tof_np[:,:,0])
    tof_bottom_np /= tof_bottom_np.max()
    tof_bottom_np[tof_bottom_np<0.25] = 0
    # save as vtr and png
    #i2d.numpy2vtr(tof_bottom_np, lengths[0:2], f"{out_directory}/tof", name='tof')
    i2d.numpy2image(tof_bottom_np, f"{out_directory}/tof_bottom.png") 


@profile
def experiment(args):

    field = "TOF"
    coarseness = args.c



    results = args.out
    if not  os.path.exists(results):
        os.mkdir(results)

    test_case = f"{field}_{coarseness:02}"
    out_directory = results+test_case
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)

    # load tof data
    tof_data = nibabel.load(args.mri+'TOF.nii.gz')
    tof_np = tof_data.get_fdata()
    hx, hy, hz = tof_data.header['pixdim'][1:4]
    lengths = np.array([float(tof_np.shape[0]*hx), float(tof_np.shape[1]*hy), float(tof_np.shape[2]*hz)])
    
    # load t1 data
    t1_data = nibabel.load(args.mri+'/T1.nii.gz')
    t1_np = t1_data.get_fdata() 
    PETSc.Sys.Print(f"Data shape: {tof_np.shape} Lengths: {lengths}")

    # load inlet data
    tof_inlet_np = i2d.image2numpy(f"{args.mri}/tof_inlets.png")
    tof_inlet_np = np.flipud(tof_inlet_np)
    tof_inlet_np = tof_inlet_np.reshape((tof_inlet_np.shape[0],tof_inlet_np.shape[1],1),order='F', copy=True)
    
    
    
    

    # restrict data
    restrict_domain = True
    if restrict_domain:
        indices_bounds = indices_restrict(tof_np,
                                        lengths=lengths,
                                        xyz_bounds=[
                                            [args.xmin,args.xmax],
                                            [args.ymin,args.ymax],
                                            [args.zmin,args.zmax]
                                        ])    
        tof_np = restrict(tof_np,indices_bounds)
        t1_np = restrict(t1_np,indices_bounds)
        tof_inlet_np = restrict(tof_inlet_np,indices_bounds)  
        
        
        lengths = np.array(tof_np.shape)*np.array([hx,hy,hz])
        
        PETSc.Sys.Print(f"Data shape: {tof_np.shape} Lengths: {lengths} After restriction ")

    # coarsen data
    mode = "max"
    if coarseness > 1:
        tof_np = downsample(tof_np,coarseness,mode)
        t1_np = downsample(t1_np,coarseness,mode)
        tof_inlet_np = downsample(tof_inlet_np,coarseness,mode)
        PETSc.Sys.Print(f"Coarse Data shape: {t1_np.shape}")
    
    
    
    # saving inputs in vtr
    PETSc.Sys.Print("start saving inputs as vtr")
    start = time.time()
    i2d.numpy2vtr(t1_np, lengths, f"{out_directory}/t1", name='t1')
    i2d.numpy2vtr(tof_np, lengths, f"{out_directory}/tof", name='tof')
    i2d.numpy2vtr(tof_inlet_np, 
                [lengths[0],lengths[1],hz],
                f"{out_directory}/tof_inlet", name='tof_inlet')
    PETSc.Sys.Print("saved inputs in "+f'{out_directory}/inputs'+f" in {time.time()-start:.2f}s")

    # create mesh
    time0 = time.time()
    mesh_type = "cartesian" if fems[0]=="DG0DG0" else "simplicial"
    mesh, cartesian_mesh =  build_meshes_from_numpy(tof_np, mesh_type=mesh_type,lengths=lengths)
    PETSc.Sys.Print(f"Mesh built in {time.time()-time0:.2f}s")


    PETSc.Sys.Print("converting into firedrake")
    tof = i2d.numpy2firedrake(cartesian_mesh, tof_np, name="TOF")
    t1 = i2d.numpy2firedrake(cartesian_mesh, t1_np, name="T1")
    tof_inlet_3d_np = np.zeros_like(tof_np)
    tof_inlet_3d_np[:,:,0] = tof_inlet_np[:,:,0]
    inlets = i2d.numpy2firedrake(cartesian_mesh, tof_inlet_3d_np, name="Inlets")
    PETSc.Sys.Print(f"converted into firedrake in {time.time()-time0:.2f}s")

    # free memory
    tof_np = None
    t1_np = None
    tof_inlet_np = None


    # setup problem
    source, sink, corrupted = btp_inputs(tof)
    btp = setup_problem(source, sink, inlets)

    save_inputs_as_pvd = False
    if save_inputs_as_pvd:
        PETSc.Sys.Print("start saving inputs as pvd")
        start = time.time()
        out_file = File(f'{out_directory}/inputs.pvd')
        out_file.write(tof,source,sink,corrupted,t1)
        PETSc.Sys.Print("saved inputs in "+f'{out_directory}/inputs.pvd'+f" in {time.time()-start:.2f}s")
    

    #setup controls
    combinations = figure1()
    
    for combination in combinations:
    
        label = "_".join(labels(*combination))

        PETSc.Sys.Print(label)

        label_dir = os.path.join(out_directory,label)
        if not os.path.exists(label_dir):
            os.mkdir(label_dir)

        
        # setup solvers
        niot_solver = setup_solver( btp, corrupted, *combinations[0])

        ierr = niot_solver.solve()

        # save solution
        pot, tdens, vel = niot_solver.get_otp_solution(niot_solver.sol)

        


        DQ0 = FunctionSpace(cartesian_mesh,"DQ",0)
        tdens_grid = Function(DQ0, name="tdens_grid")
        pot_grid = Function(DQ0, name="pot_grid")

        tdens_grid.interpolate(tdens)
        pot_grid.interpolate(pot)


        reconstruction = Function(niot_solver.fems.tdens_space)
        reconstruction.interpolate(niot_solver.tdens2image(tdens) )
        reconstruction.rename('reconstruction','Reconstruction')

        niot_solver = None

        #filename = f'{label_dir}/reconstruction.pvd'
        #out_file = VTKFile(filename,mode='w')
        #out_file.write(pot, tdens)
        #PETSc.Sys.Print(f"{ierr=}. Saved solution to "+filename)

        tdens_np = i2d.firedrake2numpy(tdens)
        pot_np = i2d.firedrake2numpy(pot)

        i2d.numpy2vtr(tdens_np, lengths, f"{out_directory}/tdens", name='tdens')
        i2d.numpy2vtr(pot_np, lengths, f"{out_directory}/pot", name='pot')


        exit()
        numpy_name = f"tdens_{cartesian_mesh.comm.rank:02d}.npy"
        path = os.path.join(label_dir,numpy_name)
        with tdens_grid.dat.vec_ro as v:
            v_np = v.array
            v_np.tofile(path)

        numpy_name = f"pot_{cartesian_mesh.comm.rank:02d}.npy"
        path = os.path.join(label_dir,numpy_name)
        print(numpy_name)
        with pot_grid.dat.vec_ro as v:
            v_np = v.array
            v_np.tofile(path)

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Reconstruct network')
    #parser.add_argument("--field", type=str, default='TOF', help="TOF")
    parser.add_argument("--c", type=int, default=1, help="coarseing factor")
    parser.add_argument("--mri", type=str, default="./mri/", help="directory with mri data")
    parser.add_argument("--out", type=str, default="./results_dirichlet/", help="output directory")
    parser.add_argument("--xmin", type=float, default=0.0, help="Lower bound x")
    parser.add_argument("--xmax", type=float, default=1000.0, help="Upper bound x")
    parser.add_argument("--ymin", type=float, default=0.0, help="Lower bound y")
    parser.add_argument("--ymax", type=float, default=1000.0, help="Upper bound y")
    parser.add_argument("--zmin", type=float, default=0.0, help="Lower bound z")
    parser.add_argument("--zmax", type=float, default=1000.0, help="Upper bound z")
    
    args, unknown = parser.parse_known_args()

    

    experiment(args)