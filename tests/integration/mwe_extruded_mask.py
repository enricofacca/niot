from firedrake import *
import numpy as np

# load topology, and coordinates from text file
cells = np.loadtxt("topol.txt", dtype=int)
coords = np.loadtxt("xy_coords.txt", dtype=np.float64)

# load partition from text file
size_partitions = np.loadtxt("size_partitions.txt", dtype=np.int32)
flat_partions = np.loadtxt("flat_partitions.txt", dtype=np.int32)

comm = COMM_WORLD
if comm.size == 1 :
    distribution_parameters = None
    variable_layers = np.loadtxt("variable_layers.txt")
elif comm.size == 2:
    if comm.rank == 0:
        partitions = (size_partitions, flat_partions)#, list(range(ncells)))
    else:
        partitions = (None, None)  
    distribution_parameters = {
        "partition": partitions,
        "overlap_type": (DistributedMeshOverlapType.VERTEX, 1)
        #"overlap_type": (DistributedMeshOverlapType.NONE, 0)
        }

    # load variable from text file
    file = f"variable_layers_{comm.rank}.txt"
    variable_layers = np.loadtxt(file)
else:
    raise ValueError("This test is designed to run with 1 (works fine) or 2 (bug appears) MPI processes")





# create mesh
plex = mesh.plex_from_cell_list(
        2, cells, coords, comm, mesh._generate_default_mesh_topology_name(mesh.DEFAULT_MESH_NAME)
    )

mesh2d = mesh.Mesh(
    plex,
    reorder=False,
    distribution_parameters=distribution_parameters,
    name=mesh.DEFAULT_MESH_NAME,
    distribution_name=None,
    permutation_name=None,
    comm=comm,
)
VTKFile("mesh2d.pvd").write(mesh2d)

# extrude mesh
mesh3d = ExtrudedMesh(mesh2d, layers=variable_layers, layer_height=1.0)
VTKFile("mesh3d.pvd").write(mesh3d)


for rank in range(comm.size):
    comm.Barrier()
    if rank == comm.rank:
        print("coord check zero", rank)
        #print(selected_mesh3d.coordinates.dat.data)
        for i,c in enumerate(mesh2d.coordinates.dat.data):
            if np.linalg.norm(c) < 1e-10:
                print(c," index ", i)
        
        print("coord check zero", rank)
        #print(selected_mesh3d.coordinates.dat.data)
        for i,c in enumerate(mesh3d.coordinates.dat.data):
            if np.linalg.norm(c) < 1e-10:
                print(c," index ", i)