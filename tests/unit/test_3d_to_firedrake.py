"""
This test is to check the conversion of a 3D numpy array 
to a firedrake function and back to a numpy array.
It should work also in MPI mode, where the numpy array is
copied to all processes and while the firedrake structures are
distributed across processes.
"""

import numpy as np
from firedrake import VTKFile, FunctionSpace, Function, COMM_SELF, PETSc, assemble, dx, SpatialCoordinate, conditional
from niot import image2dat as i2d
import numpy as np
from mpi4py.MPI import MAX,MIN

import pytest

save_pvd = True
save_vtr = False
nx=3
ny=5
nz=7
example=np.zeros([nx,ny,nz])

x_array = np.linspace(1,nx+1,nx+1)[:-1]
y_array = np.linspace(nx+1,nx+ny+1,ny+1)[:-1]
z_array = np.linspace(nx+ny+1,nx+ny+nz+1,nz+1)[:-1]

example[:,0,0] = 1.0#x_array[:]
example[0,:,0] = 2.0#y_array[:]
example[0,0,:] = 3.0#z_array[:]
example[0,0,0] = 6.0


lengths = [1,1,1]
def explicit_expression(x,y,z, lengths):
    Lx,Ly,Lz = lengths
    return (3 * conditional(x<Lx/nx,1,0)*conditional(y<Ly/ny,1,0)
            + 2 * conditional(x<Lx/nx,1,0)*conditional(z<Lz/nz,1,0)
            + 1 * conditional(y<Ly/ny,1,0)*conditional(z<Lz/nz,1,0) )






# save numpy array to vtr
i2d.numpy2vtr(example, lengths, "test3d_np", names='source')

# define cartesian mesh
def check_3d_numpy_simplicial_mesh(mesh, example):
   # get number of cells
    ncells = mesh.num_cells()
    # get number of vertices
    nvertices = mesh.num_vertices()
    
    if mesh.comm.size == 1:
        assert(ncells == 6 * np.prod(example.shape)) # 6 tetrahedra per cartesian cell
        assert(nvertices == np.prod([n+1 for n in example.shape]))

def check_3d_numpy_cartesian_mesh(mesh, example):
    # get number of cells
    if hasattr(mesh,"extruded"):
        ncells = mesh.num_cells() * (mesh.layers-1)
        nvertices = mesh.num_vertices() * (mesh.layers)
    else:
        ncells = mesh.num_cells() 
        nvertices = mesh.num_vertices()

    
    # get number of vertices
    if mesh.comm.size == 1:
        assert(ncells == np.prod(example.shape))
        assert(nvertices == np.prod([n+1 for n in example.shape]))



def check_coordinates(mesh, lengths):   
    np_coordinates = mesh.coordinates.dat.data_ro
    x_max_p = np.max(np_coordinates[:,0])
    x_min_p = np.min(np_coordinates[:,0])
    #PETSc.Sys.Print(f"rank {mesh.comm.rank} {x_min_p:.2f}<=x<={x_max_p:.2f}",comm=COMM_SELF)
    x_min = mesh.comm.allreduce(x_min_p, op=MIN)
    x_max = mesh.comm.allreduce(x_max_p, op=MAX)
    assert( np.isclose(x_min,0.0))
    assert( np.isclose(x_max,lengths[0]))

    y_min_p = np.min(np_coordinates[:,1])
    y_max_p = np.max(np_coordinates[:,1])
    #PETSc.Sys.Print(f"rank {mesh.comm.rank} {y_min_p:.2f}<=y<={y_max_p:.2f}",comm=COMM_SELF)

    y_min = mesh.comm.allreduce(y_min_p, op=MIN)
    y_max = mesh.comm.allreduce(y_max_p, op=MAX)
    assert( np.isclose(y_min,0.0))
    assert( np.isclose(y_max,lengths[1]))
    
    z_min_p = np.min(np_coordinates[:,2])
    z_max_p = np.max(np_coordinates[:,2])
    #PETSc.Sys.Print(f"rank {mesh.comm.rank} {z_min_p:.2f}<=z<={z_max_p:.2f}",comm=COMM_SELF)
    z_min = mesh.comm.allreduce(z_min_p, op=MIN)
    z_max = mesh.comm.allreduce(z_max_p, op=MAX)
    assert( np.isclose(z_min,0.0))
    assert( np.isclose(z_max,lengths[2]))

mesh_types = ['simplicial','cartesian','cartesian_extruded']
arrays = [example]
lenghts_box = [lengths,[1.0,2.0,3.0]]

@pytest.mark.parametrize("mesh_type", mesh_types)
@pytest.mark.parametrize("example", arrays)
@pytest.mark.parametrize("lengths", lenghts_box)
def test_3d_mesh(mesh_type, example, lengths):
    print(f"Testing {mesh_type}")
    if mesh_type == 'simplicial':
        mesh = i2d.build_mesh_from_numpy(example, mesh_type='simplicial',lengths=lengths)  
        check_3d_numpy_simplicial_mesh(mesh, example)
        check_coordinates(mesh, lengths)
    elif "cartesian" in mesh_type :
        extrude = ("extruded" in mesh_type)
        print(f"extrude {extrude}")
        mesh = i2d.build_mesh_from_numpy(example, mesh_type='cartesian',lengths=lengths, extrude=extrude)
        print(f"mesh {mesh}")
        
def h_length(V,mode):
    mesh = V.mesh()
    if (mode == 'cell_distance'):
        x,y = mesh.coordinates
        x_func = interpolate(x, V)
        y_func = interpolate(y, V)
        h_len = sqrt(jump(x_func)**2 + jump(y_func)**2)
    elif mode == 'cell_diameter':
        # implemation as
        alpha = 1.0
        h = CellDiameter(mesh)/sqrt(2)
        h_len = avg(h)/alpha
    else:
        raise ValueError('mode must be - center_distance, cell_diameter')
    return h_len

        
    # check_3d_numpy_cartesian_mesh(mesh, example)
    # check_coordinates(mesh, lengths)

    # # convert to firedrake
    # source_fire = i2d.numpy2firedrake(mesh, example, 'source',lengths=lengths)
    # cartesian_mesh = i2d.build_mesh_from_numpy(example, mesh_type='cartesian',lengths=lengths)
    # test_source = Function(FunctionSpace(cartesian_mesh, "DG", 0))
    # source_fire_dg0 = Function(FunctionSpace(cartesian_mesh, "DG", 0))
    # source_fire_dg0.interpolate(source_fire)

    # # test against explicit expression
    # x,y,z = SpatialCoordinate(cartesian_mesh)
    # test_source.interpolate( explicit_expression(x,y,z,lengths) )   
    # assert (np.isclose( assemble((source_fire_dg0 - test_source)**2*dx), 0.0))

    # # convert to numpy
    # source_numpy = i2d.firedrake2numpy(source_fire_dg0)

    # assert(np.allclose(example,source_numpy))


if __name__ == '__main__':
    test_3d_mesh(mesh_types[0], example, lengths)
    test_3d_mesh(mesh_types[1], example, lengths)
    test_3d_mesh(mesh_types[2], example, lengths)

    # create
    mesh = i2d.build_mesh_from_numpy(example, mesh_type='simplicial',lengths=lengths)  
    check_3d_numpy_simplicial_mesh(mesh, example)
    check_coordinates(mesh, lengths)

    # create cartesian mesh
    cartesian_mesh = i2d.build_mesh_from_numpy(example, mesh_type='cartesian',lengths=lengths, extrude=False)
    print(f"{cartesian_mesh.ufl_cell().is_simplex()=}")

    check_3d_numpy_cartesian_mesh(cartesian_mesh, example)
    check_coordinates(cartesian_mesh, lengths)


    # convert to firedrake
    source_fire = i2d.numpy2firedrake(mesh, example, 'source',lengths=lengths)
    test_source = Function(FunctionSpace(cartesian_mesh, "DG", 0))
    source_fire_dg0 = Function(FunctionSpace(cartesian_mesh, "DG", 0))
    source_fire_dg0.interpolate(source_fire)


    # test against explicit expression
    x,y,z = SpatialCoordinate(cartesian_mesh)
    test_source.interpolate( explicit_expression(x,y,z,lengths) )   
    assert (np.isclose( assemble((source_fire_dg0 - test_source)**2*dx), 0.0))


    # convert to numpy
    source_numpy = i2d.firedrake2numpy(source_fire_dg0)

    assert(np.allclose(example,source_numpy))

    if save_pvd:
        out_file = VTKFile('test3d_firedrake.pvd')
        out_file.write(source_fire)


    if save_vtr:
        i2d.numpy2vtr(source_numpy, [1,1,1], "test3d_npfiredrakenp", names='image')

