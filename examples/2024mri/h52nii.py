from firedrake import *
from firedrake.petsc import PETSc
from time import time
import os
import nibabel
import numpy as np
from niot import image2dat as i2d
import gc
import argparse
from pyop2.mpi import (
    MPI, COMM_WORLD, temp_internal_comm
)

comm = COMM_WORLD
funs = []

def h52nii(h5_file, di_name="./"):
    # Load the mesh from the HDF5 file
    with CheckpointFile(h5_file, 'r',comm=comm) as afile:
        # get mesh 
        start = time()
        PETSc.Sys.Print(f"Start mesh ")
        mesh = afile.load_mesh("relabeled_mesh")

        # get info of original image
        affine = afile.get_attr("/info/", "affine")
        PETSc.Sys.Print(f" affine ")
        offset = afile.get_attr("/info/", "offset")
        PETSc.Sys.Print(f" offeet")
        voxel_size = afile.get_attr("/info/", "voxel_size")
        PETSc.Sys.Print(f" voxel_size ")
        dimensions = afile.get_attr("/info/", "dimensions")
        PETSc.Sys.Print(f" dimensions ")
        lenghts = voxel_size*dimensions


    # create cartesian mesh for interpolation
    cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,
                                            lengths=lenghts,
                                            offset=offset)

    DG0_cartesian = FunctionSpace(cartesian_mesh, "DG", 0)
    DG0 = FunctionSpace(mesh, "DG", 0)
    interpolate_fun = Function(DG0, name="interpolatator_fun")
    interpolator = interpolate(interpolate_fun, DG0_cartesian, 
                                allow_missing_dofs=True,  
                                default_missing_val=1e30)

    # We have a set of points with corresponding data from elsewhere which vary
    # from rank to rank
    # Get the centroid coordinates
    #DQ0 = FunctionSpace(cartesian_mesh, 'DQ', 0)
    #W = VectorFunctionSpace(DQ0.ufl_domain(), DQ0.ufl_element())
    #centroid_coordinates = assemble(interpolate(DQ0.ufl_domain().coordinates, W))
    #xyz = centroid_coordinates.dat.data
    #vom = VertexOnlyMesh(mesh, xyz, redundant=False, missing_points_behaviour="ignore")
    # P0DG is the only function space you can make on a vertex-only mesh
    #P0DG_input_ordering = FunctionSpace(vom.input_ordering, "DG", 0)
    #f_at_input_points = Function(P0DG_input_ordering)


    # We interpolate the other way this time
    #f_at_input_points.interpolate(f_at_points)


    
    if comm.rank == 0:
        nx, ny, nz = dimensions
        hx, hy, hz = voxel_size
        x = np.linspace(offset[0]+hx/2.0, offset[0]+lenghts[0]-hx/2.0, nx)
        y = np.linspace(offset[1]+hy/2.0, offset[1]+lenghts[1]-hy/2.0, ny)
        z = np.linspace(offset[2]+hy/2.0, offset[2]+lenghts[2]-hz/2.0, nz)
        xv, yv, zv = np.meshgrid(x, y, z)
        points = np.vstack([xv.ravel(), yv.ravel(), zv.ravel()]).T
    else:
        points = np.zeros((0, 3), dtype=np.float64)

    # create the vertex-only mesh for f evaluation    
    vom = VertexOnlyMesh(mesh, points, redundant=True, missing_points_behaviour="ignore")
    P0DG = FunctionSpace(vom, "DG", 0)
    f_at_input_points = Function(P0DG, name=f"f_at_point")

    coords = vom.coordinates.dat.data
    print("rank",comm.rank, "shape", coords.shape)

    

    def coord2index(x, y, z):
        ix = np.fix((x - offset[0]) / voxel_size[0]).astype(int)
        iy = np.fix((y - offset[1]) / voxel_size[1]).astype(int)    
        iz = np.fix((z - offset[2]) / voxel_size[2]).astype(int)
        return ix, iy, iz
    
    ix, iy, iz = coord2index(coords[:,0], coords[:,1], coords[:,2])
    
    for j in range(comm.size):
        if comm.rank == j:
            for i in range(2):  
                print("rank", comm.rank, " point ", coords[i,:]," index ", ix[i], iy[i], iz[i])
        comm.barrier()


    # Create a P0DG function on the input ordering vertex-only mesh
    # P0DG_input_ordering = FunctionSpace(vom.input_ordering, "DG", 0)
    # point_data_input_ordering = Function(P0DG_input_ordering)

    # # We can safely set the values of this function, knowing that the data will
    # # be in the same order and on the same MPI rank as point_locations_from_elsewhere
    # point_data_input_ordering.dat.data_wo[:] = points
    # print("rank", comm.rank,"size", point_data_input_ordering.dat.data_ro.size)

    # # Interpolate puts this data onto the original vertex-only mesh
    # point_data = assemble(interpolate(point_data_input_ordering, P0DG))

    
    # Interpolation performs point evaluation
    # [test_vertex_only_mesh_manual_example 2]
    # f_at_points = assemble(interpolate(f, P0DG))


    function_np = np.zeros((dimensions[0], dimensions[1], dimensions[2]))
    function_np[:] = -1e30
    
    # the 'totals' array will hold the sum of each 'data' array
    if comm.rank==0:
        # only processor 0 will actually get the data
        global_data = np.zeros_like(function_np)
    else:
        global_data = None
                           

    funs = []
    with CheckpointFile(h5_file, 'r',comm=mesh.comm) as afile:
        # get all functions in the file
        scalar_functions_name = afile._get_function_name_function_space_name_map(afile._get_mesh_name_topology_name_map()[mesh.name], mesh.name)


    for fun_name in scalar_functions_name.keys():
        with CheckpointFile(h5_file, 'r',comm=mesh.comm) as afile:
            start = time()  
            fun = afile.load_function(mesh, fun_name)
            #funs.append(fun)
            PETSc.Sys.Print(f" Loaded {fun_name} in {time()-start:.2f} seconds")

            # interpolate to cartesian grid
            #PETSc.Sys.Print(f" Interpolating {fun.name()} to cartesian grid")
            #start = time()

            #fun_cartesian = Function(DG0_cartesian, name=fun.name())
            #fun_cartesian.interpolate(fun, 
            #                        allow_missing_dofs=True,  
            #                        default_missing_val=+1e30)
            #PETSc.Sys.Print(f" Interpolated {fun.name()} in {time()-start:.2f} seconds")
            #function_np = i2d.firedrake2numpy(fun_cartesian)
            
            #f_at_points = assemble(interpolate(f, P0DG))

            # We interpolate the other way this time
            #f_at_input_points.interpolate(f_at_points)


    
    #for fun in funs:
        #print(f_at_input_points.dat.data_ro)
        #print(dir(P0DG))
        # We interpolate the other way this time
        #f_at_input_points.interpolate(f_at_points)


        f_at_input_points.dat.data_wo[:] = -1e30 
        start = time()
        PETSc.Sys.Print(f" Interpolation {fun.name()} to input points",end="")
        f_at_input_points.interpolate(fun)
        PETSc.Sys.Print(f" - done in {time()-start:.2f} seconds")
        print("rank", comm.rank, " after interpolation ", f_at_input_points.dat.data_ro.max())


        start = time()
        function_np[ix, iy, iz] = f_at_input_points.dat.data_ro
        PETSc.Sys.Print(f" assigned numpy array for {fun.name() } in {time()-start:.2f} seconds")
        print("rank", comm.rank, " after assignment ", function_np.max())


        start = time()
        PETSc.Sys.Print(f" Creating numpy array for {fun.name() }",end="")
        with temp_internal_comm(mesh.comm) as icomm:
            #global_data = icomm.allreduce(function_np, op=MPI.MAX)#, root=0)
             
            icomm.Reduce(
                [function_np, MPI.DOUBLE],
                [global_data, MPI.DOUBLE],
                op = MPI.MAX,
            root = 0
             )
        PETSc.Sys.Print(f" created numpy array for {fun.name() } in {time()-start:.2f} seconds")
        
        # # use MPI to get the totals 
        # mesh.comm.Reduce(
        #     [function_np, MPI.DOUBLE],
        #     [global_data, MPI.DOUBLE],
        #     op = MPI.MAX,
        # root = 0
        # )
        # print("after reduce")


        # convert in numpy array
        

        # save as nii
        start = time()
        if mesh.comm.rank == 0:
            filename = os.path.join(di_name, f"{fun.name()}.nii.gz")
            nibabel.save(nibabel.Nifti1Image(global_data, affine), filename)
        mesh.comm.barrier()
        PETSc.Sys.Print(f" Saved {fun.name()} to NIfTI in {time()-start:.2f} seconds")
        #function_np = None
        #gc.collect()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert HDF5 mesh and functions to NIfTI format.")
    parser.add_argument("--h5", type=str, help="Path to the input HDF5 file.")    
    parser.add_argument("--out", type=str, default="./", help="Directory to save the output NIfTI files.")
    args = parser.parse_args()

    h52nii(args.h5, args.out)