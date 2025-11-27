from firedrake import *
import numpy as np
from niot import image2dat as i2d
from niot.utilities import color


# load topology, and coordinates from text file
cells = np.loadtxt("topol.txt", dtype=int)
coords = np.loadtxt("xy_coords.txt", dtype=np.float64)

# load partition from text file
size_partitions = np.loadtxt("size_partitions.txt", dtype=np.int32)
flat_partions = np.loadtxt("flat_partitions.txt", dtype=np.int32)





# load 3d mask from text file
mask2d = np.loadtxt(f"mask3d_0.txt", dtype=np.int32)
print("mask2d shape", mask2d.shape)
mask3d = np.zeros((mask2d.shape[0], mask2d.shape[1],5), dtype=np.int32) 
print("mask3d shape", mask3d.shape)
mask3d[:,:,0] = mask2d
for i in range(1,5):
    mask3d[:,:,i] = np.loadtxt(f"mask3d_{i}.txt", dtype=np.int32)


from niot import image2dat as i2d
mask, base, height = i2d.mask_base_height(mask3d)
comm = COMM_WORLD

np.savetxt(f'mask_base.txt',mask,fmt='%d')
np.savetxt(f'base.txt',base,fmt='%d')
np.savetxt(f'height.txt',height,fmt='%d')



# if comm.size == 1 :
#     distribution_parameters = None
#     variable_layers = np.loadtxt("variable_layers.txt")
# elif comm.size == 2:
#     if comm.rank == 0:
#         partitions = (size_partitions, flat_partions)#, list(range(ncells)))
#     else:
#         partitions = (None, None)  
#     distribution_parameters = {
#         "partition": partitions,
#         "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)
#         #"overlap_type": (DistributedMeshOverlapType.NONE, 0)
#         }

#     # load variable from text file
#     file = f"variable_layers_{comm.rank}.txt"
#     variable_layers = np.loadtxt(file)
# else:
#     raise ValueError("This test is designed to run with 1 (works fine) or 2 (bug appears) MPI processes")


def mesh2d_from_mask(mask, lenghts):
    """
    Given and input array with 0/1 values, 
    2d-mesh describing the support with quadrilaters.
    """

    # 
    mask = mask.T
    nx, ny = mask.shape
    hx, hy = lenghts[0] / nx, lenghts[1] / ny
      
    #print(f"Creating mesh from mask")
    active_cells = np.where(mask>0)
    i_cells, j_cells = active_cells
    ncell = i_cells.size
    
    #
    # topology 
    #

    # nx = 2
    # ny = 3
    # alternatively the transpose
    # 3 -- 7 -- 11
    # |  2 |  5 |
    # 2 -- 6 -- 10
    # |  1 |  4 |
    # 1 -- 5 -- 9
    # |  0 |  3 |
    # 0 -- 4 -- 8
    nodes_in_cells = [
        i_cells * (ny + 1) + j_cells, # SE
        (i_cells + 1) * (ny + 1) + j_cells, # SW
        (i_cells + 1) * (ny + 1) + j_cells + 1, # NW  
        i_cells * (ny + 1) + j_cells + 1,  #NE
        ]

    # list of active node
    nodes_in_cells = np.asarray(nodes_in_cells).swapaxes(0,-1).reshape(-1, 4)
    nodes = np.unique(nodes_in_cells.flatten())
    

    # Define the new numbering and the inverse of the active nodes
    inverse = np.zeros((nx+1)*(ny+1),dtype=int)
    inverse[:] = -1
    inverse[nodes] = np.arange(nodes.size)
    new_nodes_in_cells = inverse[nodes_in_cells]
    
    
    # Define the coordinates of the nodes (note the ny+1)
    irow, jcol = np.array([nodes % (ny+1), nodes // (ny+1)])
    xy_coord = np.array([hx * jcol, hy * irow]).T
    
    
    debug = True
    if debug:
        for i in range(ncell):
            local_nodes = new_nodes_in_cells[i,:]
            print(f"cell {i:4d} : {new_nodes_in_cells[i,0]:4d} {new_nodes_in_cells[i,1]:4d} {new_nodes_in_cells[i,2]:4d} {new_nodes_in_cells[i,3]:4d}")
            print(f" {xy_coord[local_nodes[0],0]:8.4f} {xy_coord[local_nodes[0],1]:8.4f}"
                + f" {xy_coord[local_nodes[1],0]:8.4f} {xy_coord[local_nodes[1],1]:8.4f}"
                + f" {xy_coord[local_nodes[2],0]:8.4f} {xy_coord[local_nodes[2],1]:8.4f}"
                + f" {xy_coord[local_nodes[3],0]:8.4f} {xy_coord[local_nodes[3],1]:8.4f}"
                    )
            

    name = mesh.DEFAULT_MESH_NAME
    dim = 2
    plex = mesh.plex_from_cell_list(
            dim, new_nodes_in_cells, xy_coord, comm, mesh._generate_default_mesh_topology_name(name)
        )
    

    mesh2d = mesh.Mesh(
        plex,
        reorder=False,
        distribution_parameters=None,
        name=name,
        distribution_name=None,
        permutation_name=None,
        comm=comm,
    )
        
    return mesh2d


def mesh3d_from_mask(mask3d, lengths):
    # compute 2d-ararys mask, base, height
    mask, base, height = i2d.mask_base_height(mask3d)
    selected_mesh2d = mesh2d_from_mask(mask, lengths[0:2])

    Lx, Ly = lengths[0], lengths[1]
    nx, ny = mask3d.shape[1], mask3d.shape[0]
    hx, hy = Lx/nx, Ly/ny
    
    
    def get_local_to_grid_indices_map(mesh, hx, hy):
        
        """"
        get (i,j) indices of the center of mass
        """
        DQ0 = FunctionSpace(mesh, 'DQ', 0)
        W = VectorFunctionSpace(DQ0.ufl_domain(), DQ0.ufl_element())
        centroid_coordinates = assemble(interpolate(DQ0.ufl_domain().coordinates, W))
        coord = centroid_coordinates.dat.data_ro_with_halos
        indices = np.array([coord[:,0]/hx, coord[:,1]/hy]).astype(int).T
        return indices



    # Varaible layers need to be consistent with the distribution
    # of the 2D mesh
    indices = get_local_to_grid_indices_map(selected_mesh2d, hx, hy)

    base_flat  = base[indices[:,1],indices[:,0]]
    height_flat = height[indices[:,1],indices[:,0]]
    #height_flat = 5 - base_flat

    variable_layers = np.vstack(
         (base_flat,
          height_flat)
          ).T.tolist()
      
    # create the 3D extruded mesh
    selected_mesh3d = ExtrudedMesh(selected_mesh2d, 
                                    layers=variable_layers, 
                                    layer_height=lengths[2]/mask3d.shape[2])

    return selected_mesh3d



lengths = [16, 24, 1.0]

#mesh3d = i2d.mesh_from_3d_mask(mask3d, 
#                               lengths, 
#                               variable_layer=True, 
#                                invert_rows_columns=True,
#                                comm=COMM_WORLD)


mesh3d = mesh3d_from_mask(mask3d, lengths)



mesh2d = mesh3d._base_mesh
VTKFile("mesh2d.pvd").write(mesh2d)
VTKFile("mesh3d.pvd").write(mesh3d)


for rank in range(comm.size):
    comm.Barrier()
    if rank == comm.rank:
        print("coord check zero", rank)
        #print(selected_mesh3d.coordinates.dat.data)
        for i,c in enumerate(mesh2d.coordinates.dat.data_with_halos):
            if np.linalg.norm(c) < 1e-10:
                print(c," index ", i)
        
        #print(selected_mesh3d.coordinates.dat.data)
        for i,c in enumerate(mesh3d.coordinates.dat.data):
            if np.linalg.norm(c) < 1e-10:
                print(color("red",f"{c} index {i}"))

print(mask2d.shape)
print(mask2d)