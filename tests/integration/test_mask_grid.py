from niot import image2dat as i2d
import numpy as np
from mpi4py.MPI import SUM, COMM_WORLD
from firedrake import assemble, dS, Function, FunctionSpace, VTKFile, dx, ExtrudedMesh, mesh, PETSc
from firedrake import conditional
import nibabel
import pytest

input_array = np.array([[0, 1, 1, 1 ],
                        [1, 1, 0, 1 ]],dtype=int)


base_array = np.array([[0, 0, 1, 1 ],
                        [1, 2, 0, 1 ]])



height_array = np.array([[0, 3, 4, 1 ],
                        [1, 2, 0, 1 ]])



expected_topology = np.array([
    [0, 1,  6,  5],
    [1, 2,  7,  6],
    [2, 3,  8,  7],
    [4, 5, 10,  9],
    [5, 6, 11, 10],
    [7, 8, 13, 12]], dtype=int)

# expected coordinate with hx = hy = 1.0
expected_coordinates = np.array([
    [1., 2., 3., 4., 0., 1., 2., 3., 4., 0., 1., 2., 3., 4.],
    [2., 2., 2., 2., 1., 1., 1., 1., 1., 0., 0., 0., 0., 0.,]
    ], dtype=float).T

expected_edges = np.array([
    [0,4],
    [0,1],
    [1,2],
    [2,5],
    [3,4]], dtype=int)






# def test_mask2grid(mask):
#     print("mask")
#     print(mask)
#     topol, coordinates, edges = i2d.topol_coords_edges_from_mask(mask, Lx=4, Ly=2, 
#                                                                  invert_rows_columns=True,
#                                                                  flip_up_down=True)

#     print("topol")
#     print(topol.shape)
#     print(topol)
#     print(expected_topology)
#     print("xy")
#     print(coordinates.shape)
#     print(coordinates) 
#     print(expected_coordinates)
#     print("edges")
#     print(edges.shape)
#     print(edges)
#     print(expected_edges)
#     mesh = i2d.mesh_from_topology(topol, coordinates, reorder=False)
#     VTKFile("mask2_grid.pvd").write(mesh)


#     assert np.array_equal(topol, expected_topology)
#     assert np.array_equal(coordinates, expected_coordinates)
#     assert np.array_equal(edges, expected_edges)


# def test_mask2mesh(mask):
#     Lx = 4.0
#     Ly = 2.0
#     hx = 2.0
#     hy = 1.0
#     Lx = mask.shape[0]*hx
#     Ly = mask.shape[1]*hy
#     hx, hy = Lx/mask.shape[0], Ly/mask.shape[1]
#     print("mask")
#     print(mask)
#     print(mask.shape)
#     invert = False
#     topol, coordinates, edges = i2d.topol_coords_edges_from_mask(mask, Lx=Lx, Ly=Ly,
#                                                                  invert_rows_columns=invert,
#                                                                  flip_up_down=False)
#     print("topol")
#     print(topol.shape)
#     print(topol)
#     print("xy")
#     print(coordinates.shape)
#     print(coordinates)
#     print("edges")
#     print(edges.shape)
#     print(edges)
    
    
#     mesh2d = i2d.mesh_from_topology(topol, coordinates, reorder=False)
#     VTKFile("mask2_grid.pvd").write(mesh2d)


#     if invert:
#         mesh2d.nx = mask.shape[1]
#         mesh2d.ny = mask.shape[0]
#     else:
#         mesh2d.nx = mask.shape[0]
#         mesh2d.ny = mask.shape[1]
    
#     mesh2d.xmin = 0.0 
#     mesh2d.ymin = 0.0
#     if invert:
#         mesh2d.xmax = Lx
#         mesh2d.ymax = Ly
#         mask2 = mask
#     else:
#         mask2 = mask
#         mesh.xmax = Lx
#         mesh.ymax = Ly

#     mask2 = np.copy(mask)
#     mask2[0,:] = 0.0
#     mask_fire = i2d.numpy2firedrake(mesh2d, mask2, "mask", invert_rows_columns=invert)
#     VTKFile("mask_cut.pvd").write(mask_fire)

#     total_ncells = assemble(mask_fire*dx) # just to check it works
#     print("total cells", total_ncells)


    
#     print("num cells", mesh2d.num_cells())
#     print("num vertices", mesh2d.num_vertices())
#     print("num edges", mesh2d.num_edges())
#     print('num_facets', mesh2d.num_facets())
#     print('num faces', mesh2d.num_faces())
    

#     print(edges.shape)
#     #print(dir(mesh))

#     REAL = FunctionSpace(mesh2d, "R", 0)
#     one = Function(REAL, name="one")
#     one.assign(1.0)
#     internal_face_size = assemble(one*dS)
    
#     local_ncells = mesh2d.num_cells()    
#     # global number of cells with MPI
#     #total_ncells = COMM_WORLD.allreduce(local_ncells, op=SUM)
#     #print("total cells", total_ncells)
    
    
#     assert(np.isclose(total_ncells,mask.sum()*hx*hy,atol=1e-12))
#     assert(mesh2d.num_vertices() == coordinates.shape[0])
#     if np.isclose(hx, hy, rtol=1e-10):
#         assert(np.isclose(internal_face_size,edges.shape[0]*hx,rtol=1e-10))
#         print("internal faces", internal_face_size, edges.shape[0]*hx)
    

    
#     #assert(mesh.num_edges() == edges.shape[0])


test_nx=3
test_ny=5
test_nz=7
example_easy=np.zeros([test_nx,test_ny,test_nz])


example_easy[:,0,0] = 1.0#x_array[:]
example_easy[0,:,0] = 2.0#y_array[:]
example_easy[0,0,:] = 3.0#z_array[:]
example_easy[0,0,0] = 6.0



@pytest.mark.parametrize("example", [example_easy])
def test_example(example, invert_rows_columns=True, variable_layer=False):    
    lengths = [0.4,50.0,0.1]
    
    binary = np.zeros_like(example, dtype=int)
    binary[example>0] = 1
    
    mask2d, base2d, height2d = i2d.mask_base_height(binary)
    full_mesh2d = i2d.build_mesh_from_numpy(mask2d.shape, 
                                             lengths=lengths,
                                             comm=COMM_WORLD)
    full_height_fire = i2d.numpy2firedrake(full_mesh2d, height2d, "height")
    

    selected_mesh2d = i2d.mesh_from_2d_mask(mask2d, 
                                            lengths[0:2], 
                                            invert_rows_columns=invert_rows_columns,
                                            comm=COMM_WORLD)
    selected_height_fire = i2d.numpy2firedrake(selected_mesh2d, height2d, "selected_height")
    

    full_height_np = i2d.firedrake2numpy(full_height_fire)
    selected_height_np = i2d.firedrake2numpy(selected_height_fire)
    assert(np.allclose(full_height_np, height2d, rtol=1e-10, atol=1e-10))
    assert(np.allclose(selected_height_np, height2d, rtol=1e-10, atol=1e-10))
    PETSc.Sys.Print("2d PASSED")
    
    

    full_mesh3d = i2d.build_mesh_from_numpy(binary.shape,
                                        lengths=lengths,
                                        comm=COMM_WORLD)
    selected_mesh3d = i2d.mesh_from_3d_mask(binary,
                                   lengths=lengths,
                                   variable_layer=variable_layer,
                                  comm=COMM_WORLD)
    binary = np.zeros_like(example, dtype=int)
    binary[example>0] = 1
    

    
    # create firedrake function
    selected_example_fd = i2d.numpy2firedrake(selected_mesh3d, example, "selected_example")
    full_example_fd = i2d.numpy2firedrake(full_mesh3d, example, "example")
    
    selected_example_np = i2d.firedrake2numpy(selected_example_fd)
    full_example_np = i2d.firedrake2numpy(full_example_fd)
    
    assert(np.allclose(selected_example_np, example, rtol=1e-10))
    assert(np.allclose(full_example_np, example, rtol=1e-10))
    PETSc.Sys.Print("3d PASSED")




if __name__ == "__main__":
    # parse arguments
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nref", type=int, default=0, help="number of uniform refinements")
    parser.add_argument("--variable", type=int, default=0, help="which test to run")
    args = parser.parse_args()

    nref = args.nref
    mask = input_array
    if nref > 0:
        mask = np.kron(mask, np.ones((2**nref, 2**nref), dtype=int))
    #test_mask2grid(mask)
    input_array = np.array([
        [0, 1, 1, 1 ],
        [1, 1, 0, 1 ]
        ],dtype=int)

    from PIL import Image
    im = Image.fromarray(mask.astype(np.uint8)*255) #*255 to convert 0/1 to 0/255
    im.save("PIL_array.png")

    # save with mathplotlib
    import matplotlib.pyplot as plt
    plt.imsave("matplotlib_array.png", mask, cmap='gray', vmin=0, vmax=1)
    
    test_nx=3
    test_ny=5
    test_nz=7
    example=np.zeros([test_nx,test_ny,test_nz])


    example[:,0,0] = 1.0#x_array[:]
    example[0,:,0] = 2.0#y_array[:]
    example[0,0,:] = 3.0#z_array[:]
    example[0,0,0] = 6.0
    
    test_nx=2*2**nref
    test_ny=3*2**nref
    test_nz=5#*2**nref
    x = np.linspace(0, 1, test_nx)
    y = np.linspace(0, 1, test_ny)
    z = np.linspace(0, 1, test_nz)
    xv, yv, zv = np.meshgrid(x, y, z)
    #print(xv)
    r = 0.6
    sphere = (xv-0.5)**2 + (yv-0.5)**2 + (zv-0.5)**2 - r**2

    example = np.zeros_like(sphere, dtype=int)
    example[sphere<0] = 1
    example[sphere>=0] = 0
    example[0:test_nx//2-1,0:test_ny//2,:] = 0
    variable_layer = args.variable> 0
    test_example(example, invert_rows_columns=True, variable_layer=variable_layer)
    #print(sphere)
    #print(example)
    #test_mask2mesh(mask)
    