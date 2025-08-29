from copy import deepcopy as cp
import json 
import numpy as np
from niot import image2dat as i2d
from niot import utilities
from niot import optimal_transport as ot
from niot import NiotSolver
from niot import SpaceDiscretization
import os
from scipy.ndimage import zoom
from firedrake import COMM_WORLD, VTKFile, Function, PETSc
import pytest
import argparse
import time

def load_input(example, nref, mask, network_file, comm=COMM_WORLD):
    # btp inputs
    img_sources = f'{example}/source.png'
    img_sinks = f'{example}/sink.png'
    
    np_source = i2d.image2numpy(img_sources,normalize=True,invert=True)
    np_sink = i2d.image2numpy(img_sinks,normalize=True,invert=True)
    
    # taking just the support of the sources and sinks
    np_source[np.where(np_source>0.0)] = 1.0
    np_sink[np.where(np_sink>0.0)] = 1.0

    # balancing the mass
    nx, ny = np_source.shape
    hx, hy = 1.0/nx, 1.0/ny
    mass_source = np.sum(np_source)*hx*hy
    mass_sink = np.sum(np_sink)*hx*hy
    if abs(mass_source-mass_sink)>1e-16:
        np_sink *= mass_source/mass_sink 

    
    # load image or numpy array
    try:
        img_networks = f'{example}/{network_file}'
        np_network = i2d.image2numpy(img_networks,normalize=True,invert=True)
        #PETSc.Sys.Print(f'using {img_networks}',comm=comm)
    except: 
        np_network = np.load(f'{example}/{network_file}')
        #if np_network.ndim == 2 and i2d.convention_2d_flipud:
        #np_network = np.flipud(np_network)
    np_mask = i2d.image2numpy(f'{example}/{mask}',normalize=True,invert=True)

    if nref != 0:
        PETSc.Sys.Print(f"Refining mesh {nref} times",comm=comm)
        np_source = zoom(np_source, 2**nref, order=0, mode='nearest')
        np_sink = zoom(np_sink, 2**nref, order=0, mode='nearest')
        np_network = zoom(np_network, 2**nref, order=0, mode='nearest')
        np_mask = zoom(np_mask, 2**nref, order=0, mode='nearest')


    return np_source, np_sink, np_network, np_mask


@pytest.fixture
def inputs_data(nref=0):
    #return build_inputs(0)


#def build_inputs(nref=0):
    
    PETSc.Sys.Print(f"Reading inputs")
    # set problem case
    example_directory = os.path.abspath(f"../../examples/2024FaccaNordbottenHanson/data/y_net_frog200/")
    nref = nref
    mask_file = "mask_medium.png"
    network_file = "network.png"

    # load input and set numpy quanities
    np_source, np_sink, np_network, np_mask = load_input(example_directory, nref, mask_file, network_file)
    np_corrupted = np_network * (1-np_mask) # Corrupt the network
    np_confidence = (1.0 - np_mask) 
    #np_confidence[:] = 1.0
    np_confidence_one = cp(np_confidence)
    np_confidence_one[:] = 1.0



    # Create mesh
    h = 1.0 / np_corrupted.shape[1]
    mesh = i2d.build_mesh_from_numpy(np_corrupted.shape, mesh_type="cartesian", lengths=[1.0,np_corrupted.shape[0]*h])

    # Convert numpy arrays to firedrake functions    
    source = i2d.numpy2firedrake(mesh, np_source, name="source")
    sink = i2d.numpy2firedrake(mesh, np_sink, name="sink")
    network = i2d.numpy2firedrake(mesh, np_network, name="network")
    corrupted = i2d.numpy2firedrake(mesh, np_corrupted, name="corrupted")
    mask = i2d.numpy2firedrake(mesh, np_mask, name="mask")
    confidence = i2d.numpy2firedrake(mesh, np_confidence, name="confidence")
    one = i2d.numpy2firedrake(mesh, np_confidence_one, name="one")

    # Define the branched transport problem
    ot.balance(source, sink)
    btp = ot.BranchedTransportProblem(source, sink, gamma=0.5)

    yield  {"btp": btp,
        "corrupted": corrupted,
        "confidence": confidence,
        "one": one}


@pytest.mark.parametrize("discr_norm", ["l2","dual_h1"])
@pytest.mark.parametrize("wd", [2,10])
@pytest.mark.parametrize("wp", [1e-1,5])
@pytest.mark.parametrize("conf", ["mask","one"])
def test_gradients(inputs_data, discr_norm, wd, wp, conf):
    """
    Check take all gradeint are computed correctly
    using or not the adjoint method
    """
    btp = inputs_data["btp"]
    corrupted = inputs_data["corrupted"]
    if conf == "mask":
        confidence = inputs_data["confidence"]
    else:
        confidence = inputs_data["one"]
    
    
    # identity map
    t2i_map = {"type":"identity", "scaling":1.0 }

    start = time.time()
    niot_solver = NiotSolver(btp, 
                            corrupted,  
                            confidence=confidence, 
                            spaces = "DG0DG0",
                            cell2face = 'harmonic_mean',
                            setup = False,
                            )

    # just the computation of the gradient 
    niot_solver.ctrl_set("discrepancy_weight",wd)
    niot_solver.ctrl_set("penalization_weight",wp)
    niot_solver.ctrl_set("discrepancy_norm",discr_norm)
    niot_solver.ctrl_set("max_iter",2)
    niot_solver.ctrl_set("verbose",0)
    niot_solver.ctrl_set("use_adjoint",False)
    niot_solver.ctrl_set("tdens2image",t2i_map)
    niot_solver.setup()
    niot_solver.solve()
    end = time.time() 
    PETSc.Sys.Print(f"Time without adjoint {end-start}")


    # get gradeint cofunciton
    gradient_cofun = niot_solver.gradient_lagrangian

    # save h1_dual_pot
    #VTKFile("h1_dual_pot.pvd").write(niot_solver.pot_dual_h1)
    gradient = Function(corrupted.function_space(), name="gradient")
    with gradient.dat.vec as g, niot_solver.gradient_lagrangian.dat.vec as g_niot:
        niot_solver.fems.inv_tdens_mass_matrix.mult(g_niot, g)
    #VTKFile("gradient.pvd").write(gradient)

    PETSc.Sys.Print()
    # adjoint method
    PETSc.Sys.Print("Using the adjoint method")
    PETSc.Sys.Print()

    start = time.time()
    niot_solver_adj = NiotSolver(btp, 
                            corrupted,  
                            confidence=confidence, 
                            spaces = "DG0DG0",
                            cell2face = 'harmonic_mean',
                            setup = False,
                            )


    # just the computation of the gradient 
    niot_solver_adj.ctrl_set("discrepancy_weight",wd)
    niot_solver_adj.ctrl_set("discrepancy_norm",discr_norm)
    niot_solver_adj.ctrl_set("penalization_weight",wp)
    niot_solver_adj.ctrl_set("max_iter",2)
    niot_solver_adj.ctrl_set("verbose",0)
    niot_solver_adj.ctrl_set("use_adjoint",True)
    niot_solver_adj.ctrl_set("tdens2image",t2i_map)
    niot_solver_adj.setup()
    niot_solver_adj.solve()

    end = time.time() 
    PETSc.Sys.Print(f"Time without adjoint {end-start}")
    
    # get gradeint cofunciton
    gradient_cofun_adj = niot_solver_adj.gradient_lagrangian
    
    # save h1_dual_pot
    #VTKFile("h1_dual_pot_adj.pvd").write(niot_solver_adj.pot_dual_h1)

    gradient_adj = Function(corrupted.function_space(), name="gradient_adj")
    with gradient_adj.dat.vec as g_adj, niot_solver_adj.gradient_lagrangian.dat.vec as g_niot_adj:
        niot_solver_adj.fems.inv_tdens_mass_matrix.mult(g_niot_adj, g_adj)
    #VTKFile("gradient_adj.pvd").write(gradient_adj)

    PETSc.Sys.Print()
    # adjoint method
    PETSc.Sys.Print("Compare gradients")
    PETSc.Sys.Print()

    with gradient_cofun.dat.vec as g, gradient_cofun_adj.dat.vec as g_adj:
        print(utilities.msg_bounds(g,"g"))
        print(utilities.msg_bounds(g_adj,"g_adj"))
        diff = g - g_adj
        diff_norm = diff.norm() / g.norm()

    PETSc.Sys.Print(f"relative diff norm functions = {diff_norm}")
    assert(diff_norm < 1e-6)

    with gradient.dat.vec as g, gradient_adj.dat.vec as g_adj:
        print(utilities.msg_bounds(g,"g"))
        print(utilities.msg_bounds(g_adj,"g_adj"))
        diff = g - g_adj
        diff_norm = diff.norm() / g.norm()

    PETSc.Sys.Print(f"relative diff norm functions = {diff_norm}")
    assert(diff_norm < 1e-4)

if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--nref", type=int, default=0, help="number of mesh refinement")
    args = argparser.parse_args()
    nref = args.nref
    
    # set problem case
    input_dic = build_inputs(nref)
    

    test_gradients(input_dic, discr_norm="dual_h1", wd=2, wp=10, conf="one")


