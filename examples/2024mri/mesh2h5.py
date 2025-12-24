import nibabel 
import argparse
import numpy as np
from connected_components_tof import connected_components, main_network_equal_one
import os
from scipy.ndimage import gaussian_filter
#import pygalmesh
from firedrake import *
from niot import image2dat as i2d
from mwe import MyRelabeledMesh
import meshio
from firedrake import CheckpointFile, COMM_WORLD, PETSc, VTKFile, DirichletBC, FunctionSpace, Function, TestFunction, TrialFunction, inner, grad, dx, conditional
import localthickness as lt
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation
import time
from mpi4py import MPI
import gc

def bounding_box(mesh):
    lower = mesh.coordinates.dat.data.min(axis=0)
    upper = mesh.coordinates.dat.data.max(axis=0)
    
    global_lower = []
    global_upper = []
    for i in range(mesh.geometric_dimension()):
        global_lower.append(mesh.comm.allreduce(lower[i], op=MPI.MIN))
        global_upper.append(mesh.comm.allreduce(upper[i], op=MPI.MAX))
    
    return np.array(global_lower), np.array(global_upper)

def read_nifti_file(filepath, mesh, lengths, offset, name, output_function):
    """Reads a NIfTI file and returns the image data as a numpy array."""
    PETSc.Sys.Print(f"Reading NIfTI file: {filepath}", end="")    
    start  = time.time()
    img = nibabel.load(filepath)
    data_np = img.get_fdata()
    PETSc.Sys.Print(f" - completed in {time.time()-start:.2e} s")

    start  = time.time()
    PETSc.Sys.Print("Intepolate from numpy", end="")    
    i2d.numpy2firedrake(mesh, data_np, name=name, lengths=lengths, offset=offset, output_function=output_function)
    PETSc.Sys.Print(f" - completed in {time.time()-start:.2e} s")
    data_np = None  # Free memory
    gc.collect()
    

def setup(mri_directory, 
          out_directory,
            mesh_tpye="simplex",
            save_h5=False,
            save_pvd=False,
            test_dirichlet_bc = False
          ):

    def load_indo_data(mri_directory):
        
        # load tof data and get basic info
        dir_nii = mri_directory
        tof_data = nibabel.load(f"{dir_nii}/TOF.nii.gz")
        dimensions = tof_data.header.get_data_shape()[:3]

        hx, hy, hz = tof_data.header['pixdim'][1:4]
        voxel_size = np.array([hx, hy, hz])
        affine = tof_data.affine

        return voxel_size, dimensions, affine

    def load_data(dir_nii):
        # get brain mask
        nii_file = f"{dir_nii}/brain_mask_smooth.nii.gz"
        brain_mask_data = nibabel.load(nii_file)
        brain_mask_np = brain_mask_data.get_fdata()
          
        

        # load tof data
        # tof_np = tof_data.get_fdata()
        file_nii = f"{dir_nii}/TOF.nii.gz"
        tof_data = nibabel.load(file_nii)
        tof_np = tof_data.get_fdata()

        # load tof data
        # tof_np = tof_data.get_fdata()
        file_nii = f"{dir_nii}/aseg.nii.gz"
        data_aseg = nibabel.load(file_nii)
        aseg_np = data_aseg.get_fdata()
        

        # load main network
        file_nii = f"{dir_nii}/T1.nii.gz"
        t1_data = nibabel.load(file_nii)
        t1_np = t1_data.get_fdata()


        return voxel_size, tof_data.affine, tof_np, brain_mask_np, t1_np, aseg_np
    
    def load_preprocessed(out_directory):
        """
        Load preprocessed data from nii.gz files.
        Returns:
            main_network_np: numpy array of main network
            sink_support_np: numpy array of sink support
            skeleton_np: numpy array of skeleton
            thickness_np: numpy array of thickness
            tof_clean_np: numpy array of smoothed tof
        """
        main_network_data = nibabel.load(f"{out_directory}/main_network.nii.gz")
        main_network_np = main_network_data.get_fdata()

        inlets_np = nibabel.load(f"{out_directory}/inlets.nii.gz").get_fdata()
        external_network_np = nibabel.load(f"{out_directory}/external_network.nii.gz").get_fdata()

        sink_support_data = nibabel.load(f"{out_directory}/sink_support.nii.gz")
        sink_support_np = sink_support_data.get_fdata()

        skeleton_data = nibabel.load(f"{out_directory}/skeleton.nii.gz")
        skeleton_np = skeleton_data.get_fdata()

        thickness_data = nibabel.load(f"{out_directory}/thickness.nii.gz")
        thickness_np = thickness_data.get_fdata()

        tof_clean_data = nibabel.load(f"{out_directory}/tof_clean.nii.gz")
        tof_clean_np = tof_clean_data.get_fdata()
        voxel_size = main_network_data.header['pixdim'][1:4]
        dimensions = main_network_np.shape
        
        lengths = np.array(dimensions) * voxel_size
        affine = main_network_data.affine
        offset = affine[:3, 3]

        PETSc.Sys.Print(f" Prepoccesed voxel_size: {voxel_size}, lengths: {lengths}, offset: {offset}")

        return main_network_np, sink_support_np, skeleton_np, thickness_np, tof_clean_np, external_network_np, inlets_np

    voxel_size, dimensions, affine = load_indo_data(mri_directory)
    hx, hy, hz = voxel_size
    lengths = np.array([hx, hy, hz]) * np.array(dimensions)
    offset = np.array(affine[:3, 3])

    PETSc.Sys.Print(f" Inputs shape {dimensions}")
    PETSc.Sys.Print(f" voxel_size: {voxel_size}")
    PETSc.Sys.Print(f" lengths: x {lengths[0]:.10e}, y {lengths[1]:.10e}, z {lengths[2]:.10e}")
    PETSc.Sys.Print(f" offset:  x {offset[0]:.10e}, y {offset[1]:.10e}, z {offset[2]:.10e}")
    
    # preprocess data
    main_network_np, sink_support_np, skeleton_np, thickness_np, tof_clean_np, external_network_np , inlets_np = load_preprocessed(out_directory)
    
    # reload the mesh from file
    if mesh_tpye == "simplex":
        mesh_name = "brain_main"
    else:
        mesh_name = "brain_hexa_main"
    nproc = PETSc.COMM_WORLD.getSize()
    try:
        meshfile = os.path.join(out_directory, f"{mesh_name}_{nproc:04d}.h5")
        PETSc.Sys.Print(f"Loading mesh from {meshfile}")
        if not os.path.exists(meshfile):
            PETSc.Sys.Print(f"Mesh in {meshfile} not found")
            raise FileNotFoundError
        start = time.time() 
        with CheckpointFile(meshfile, 'r', comm=COMM_WORLD) as afile:
            mesh = afile.load_mesh("mesh")
        PETSc.Sys.Print(f"Mesh loaded in {time.time()-start:.2e} s")
    except:
        PETSc.Sys.Print("Not found .h5 file")
        
    try:
        filename = os.path.join(out_directory,f"{mesh_name}.e")
        PETSc.Sys.Print(f" reading {filename} file")
        start = time.time() 
        mesh = Mesh(filename)
        PETSc.Sys.Print(f"completed in {time.time()-start:.2e} s")
    except:
        PETSc.Sys.Print("Not found .e file")
        PETSc.Sys.Print("reading mesh from .msh file")
        start = time.time() 
        mesh = Mesh(os.path.join(out_directory,f"{mesh_name}.msh"))
        PETSc.Sys.Print(f"completed in {time.time()-start:.2e} s")
    
    PETSc.Sys.Print("Bounding box mesh")
    lower, upper = bounding_box(mesh)
    PETSc.Sys.Print(f" {lower[0]:.10e}<= x <>{upper[0]:.10e} lx{upper[0]-lower[0]:.10e}. Lx={lengths[0]:.10e}")
    PETSc.Sys.Print(f" {lower[1]:.10e}<= y <>{upper[1]:.10e} ly{upper[1]-lower[1]:.10e} Ly={lengths[1]:.10e}")
    PETSc.Sys.Print(f" {lower[2]:.10e}<= z <>{upper[2]:.10e} lz{upper[2]-lower[2]:.10e} Lz={lengths[2]:.10e}") 
    
    # PETSc.Sys.Print("Shifting coordinates")
    # mesh.coordinates.dat.data[:, 0] -= offset[0]
    # mesh.coordinates.dat.data[:, 1] -= offset[1]
    # mesh.coordinates.dat.data[:, 2] -= offset[2]
    
    # PETSc.Sys.Print("Offset completed")
    # lower, upper = bounding_box(mesh)
    # PETSc.Sys.Print(f" {lower[0]:.10e}<= x <>{upper[0]:.10e}. Lx={lengths[0]:.10e}")
    # PETSc.Sys.Print(f" {lower[1]:.10e}<= y <>{upper[1]:.10e}. Ly={lengths[1]:.10e}")
    # PETSc.Sys.Print(f" {lower[2]:.10e}<= z <>{upper[2]:.10e}. Lz={lengths[2]:.10e}")
        
    
    def interpolate_from_numpy(target_mesh, data_np, lengths, name):
        start  = time.time()
        PETSc.Sys.Print("Intepolate main network from numpy", end="")    
        data_mesh = i2d.numpy2firedrake(target_mesh, data_np, name=name,lengths=lengths)
        PETSc.Sys.Print(f" - completed in {time.time()-start:.2e} s")
        return data_mesh
        
    def interpolate_from_cartesian(target_mesh, cartesian_mesh, data_np, name):
        start  = time.time()
        PETSc.Sys.Print("Intepolate main network from cartesian", end="")    
        DG0 = FunctionSpace(target_mesh, "DG", 0)
        data_mesh = Function(DG0, name=name)
        data_cartesian = i2d.numpy2firedrake(cartesian_mesh, data_np, name=name)
        data_mesh.interpolate(main_network_cartesian)
        PETSc.Sys.Print(f" - completed in {time.time()-start:.2e} s")
        return data_mesh, data_cartesian 


    DG0 = FunctionSpace(mesh, "DG", 0)
    main_network_mesh = Function(DG0, name="main_network")
    filepath = os.path.join(out_directory, "main_network.nii.gz")
    read_nifti_file(filepath, mesh, lengths, offset, "main_network",  output_function = main_network_mesh)

    
    my = False
    remark_mesh = True
    if mesh_tpye == "cartesian":
        remark_mesh = False
     
    lower, upper = bounding_box(mesh)
    if remark_mesh:
        if my:
            zmin = lower[2]
            start  = time.time()
            PETSc.Sys.Print("intepolate marker for relabeled mesh", end="")
            marker_space = FunctionSpace(mesh, "DQ", 0)
            main_network_indicator = Function(marker_space, name="main_network_indicator")
            x,y,z = mesh.coordinates
            main_network_indicator.interpolate(main_network_mesh * conditional(abs(z-zmin) < hx,1,0))
            PETSc.Sys.Print(f" - completed in {time.time()-start:.2e} s")
                
            start  = time.time()
            PETSc.Sys.Print("Relabeled mesh", end="")
            relabeled_mesh = MyRelabeledMesh(mesh, [main_network_indicator], 
                                                [99],
                                                boundary_only=True,
                                                name="relabeled_mesh")
            PETSc.Sys.Print(f" - completed in {time.time()-start:.2e} s")

            lower, upper = bounding_box(relabeled_mesh)
            PETSc.Sys.Print("Bounding box lower after relabeling:")
            PETSc.Sys.Print(f" {lower[0]:.10e}<= x <>{upper[0]:.10e}. Lx={lengths[0]:.10e}")
            PETSc.Sys.Print(f" {lower[1]:.10e}<= y <>{upper[1]:.10e}. Ly={lengths[1]:.10e}")
            PETSc.Sys.Print(f" {lower[2]:.10e}<= z <>{upper[2]:.10e}. Lz={lengths[2]:.10e}")
        
            # PETSc.Sys.Print("Shifting coordinates. I do not know why relabeled mesh shift them back")
            # relabeled_mesh.coordinates.dat.data[:, 0] -= offset[0]
            # relabeled_mesh.coordinates.dat.data[:, 1] -= offset[1]
            # relabeled_mesh.coordinates.dat.data[:, 2] -= offset[2]
            # PETSc.Sys.Print("Offset completed")
        else:
            zmin = lower[2]
            start  = time.time()
            PETSc.Sys.Print("intepolate marker for relabeled mesh", end="")
            marker_space = FunctionSpace(mesh, "HDiv Trace", 0)
            main_network_bottom_indicator = Function(marker_space, name="main_network_indicator")
            x,y,z = mesh.coordinates
            main_network_bottom_indicator.interpolate(main_network_mesh * conditional(abs(z-zmin) < hx,1,0))
            start  = time.time()
            PETSc.Sys.Print(f" done in {time.time()-start:.2e} s")
            PETSc.Sys.Print("Relabeled mesh", end="")
            relabeled_mesh = RelabeledMesh(mesh, [main_network_bottom_indicator], 
                                                [99],
                                                name="relabeled_mesh")
            PETSc.Sys.Print(f" - completed in {time.time()-start:.2e} s")
            lower, upper = bounding_box(relabeled_mesh)
            PETSc.Sys.Print("Bounding box lower after relabeling:")
            PETSc.Sys.Print(f" {lower[0]:.10e}<= x <>{upper[0]:.10e}. Lx={lengths[0]:.10e}")
            PETSc.Sys.Print(f" {lower[1]:.10e}<= y <>{upper[1]:.10e}. Ly={lengths[1]:.10e}")
            PETSc.Sys.Print(f" {lower[2]:.10e}<= z <>{upper[2]:.10e}. Lz={lengths[2]:.10e}")
        
            # PETSc.Sys.Print("Shifting coordinates. I do not know why relabeled mesh shift them back")
            # relabeled_mesh.coordinates.dat.data[:, 0] -= offset[0]
            # relabeled_mesh.coordinates.dat.data[:, 1] -= offset[1]
            # relabeled_mesh.coordinates.dat.data[:, 2] -= offset[2]
            # PETSc.Sys.Print("Offset completed")
    else:
        relabeled_mesh = mesh
        relabeled_mesh.name = "relabeled_mesh"



    PETSc.Sys.Print(f" Relabeled mesh created")
    lower, upper = bounding_box(relabeled_mesh)
    new_lengths = upper - lower
    PETSc.Sys.Print(f" {lower[0]:.10e}<= x <>{upper[0]:.10e}. Lx={lengths[0]:.10e} newLx={new_lengths[0]:.10e}")
    PETSc.Sys.Print(f" {lower[1]:.10e}<= y <>{upper[1]:.10e}. Ly={lengths[1]:.10e} newLy={new_lengths[1]:.10e}")
    PETSc.Sys.Print(f" {lower[2]:.10e}<= z <>{upper[2]:.10e}. Lz={lengths[2]:.10e} newLz={new_lengths[2]:.10e}")
    

    # # 
    # PETSc.Sys.Print(" Interpolate numpy data to relabeled mesh")
    # tof_mesh = i2d.numpy2firedrake(relabeled_mesh, tof_np, name='tof',lengths=lengths)
    # tof_smooth_mesh = i2d.numpy2firedrake(relabeled_mesh, tof_clean_np, name='tof_smooth',lengths=lengths)
    # t1_mesh = i2d.numpy2firedrake(relabeled_mesh, t1_np, name='t1',lengths=lengths)
    # brain_mask_mesh = i2d.numpy2firedrake(relabeled_mesh, brain_mask_np, name='brain_mask',lengths=lengths)
    # main_network_mesh = i2d.numpy2firedrake(relabeled_mesh, main_network_np, name='main_network',lengths=lengths)
    # sink_support_mesh = i2d.numpy2firedrake(relabeled_mesh, sink_support_np, name='sink_support',lengths=lengths)
    # skeleton_mesh = i2d.numpy2firedrake(relabeled_mesh, skeleton_np, name='skeleton',lengths=lengths)
    # thickness_mesh = i2d.numpy2firedrake(relabeled_mesh, thickness_np, name='thickness',lengths=lengths)
    # inlets_mesh = i2d.numpy2firedrake(relabeled_mesh, inlets_np, name='inlets',lengths=lengths)

    # PETSc.Sys.Print("Shifting coordinates of relabeled mesh")
    # relabeled_mesh.coordinates.dat.data[:, 0] += offset[0]
    # relabeled_mesh.coordinates.dat.data[:, 1] += offset[1]
    # relabeled_mesh.coordinates.dat.data[:, 2] += offset[2]
    # PETSc.Sys.Print("Offset completed")
   
    

    
    if save_h5:
        n_proc = COMM_WORLD.size
        if mesh_tpye == "simplex":
            h5_filename = os.path.join(mri_directory,
                                f"inputs_nproc{n_proc}.h5")
        else:
            h5_filename = os.path.join(mri_directory,
                                f"inputs_hexa_nproc{n_proc}.h5")
        PETSc.Sys.Print(f"Saving to {h5_filename}", end="")
        with CheckpointFile(h5_filename, 'w', comm=COMM_WORLD) as afile:
            afile.save_mesh(relabeled_mesh,"relabeled_mesh")
            PETSc.Sys.Print(f" mesh ")

            afile.save_function(main_network_mesh)
            PETSc.Sys.Print(f" main_network ")
        
            for filepath, name in [
                (os.path.join(out_directory, "inlets.nii.gz"), "inlets"),
                (os.path.join(out_directory, "thickness.nii.gz"),  "thickness"),
                (os.path.join(out_directory, "skeleton.nii.gz"),  "skeleton"),
                (os.path.join(out_directory, "sink_support.nii.gz"), "sink_support"),
                (os.path.join(out_directory, "external_network.nii.gz"), "external_network"),
                (os.path.join(out_directory, "tof_clean.nii.gz"), "tof"),
                (os.path.join(mri_directory, "T1.nii.gz"), "t1"),
                (os.path.join(mri_directory, "TOF.nii.gz"), "tof"),
                (os.path.join(mri_directory, "brain_mask_smooth.nii.gz"), "brain_mask_smooth"),
                ]:
                read_nifti_file(filepath, mesh, lengths, offset, name, main_network_mesh)

                afile.save_function(main_network_mesh)
                PETSc.Sys.Print(f" {name} ")

            # afile.save_function(tof_mesh)
            # PETSc.Sys.Print(f" tof ", end="")
            # afile.save_function(t1_mesh)
            # PETSc.Sys.Print(f" t1 ", end="")
            # afile.save_function(brain_mask_mesh)
            # PETSc.Sys.Print(f" brain_mask ", end="")
            # afile.save_function(main_network_mesh)
            # PETSc.Sys.Print(f" main_network ", end="")
            # afile.save_function(sink_support_mesh)
            # PETSc.Sys.Print(f" sink_support ", end="")
            # afile.save_function(skeleton_mesh)
            # PETSc.Sys.Print(f" skeleton ", end="")
            # afile.save_function(thickness_mesh)
            # PETSc.Sys.Print(f" thickness ", end="")
            # afile.save_function(inlets_mesh)
            # PETSc.Sys.Print(f" inlets ", end="")


            PETSc.Sys.Print(f" Include info ")
            afile.require_group("/info/")
            afile.set_attr("/info/", "affine", affine)
            PETSc.Sys.Print(f" affine ")
            afile.set_attr("/info/", "offset", offset)
            PETSc.Sys.Print(f" offeset")
            afile.set_attr("/info/", "voxel_size", voxel_size)
            PETSc.Sys.Print(f" voxel_size ")
            afile.set_attr("/info/", "dimensions", dimensions)
            PETSc.Sys.Print(f" dimensions ")
            PETSc.Sys.Print(f" - done")

    if test_dirichlet_bc:
        PETSc.Sys.Print("Shifting coordinates of relabeled mesh")
        relabeled_mesh.coordinates.dat.data[:, 0] -= offset[0]
        relabeled_mesh.coordinates.dat.data[:, 1] -= offset[1]
        relabeled_mesh.coordinates.dat.data[:, 2] -= offset[2]
        PETSc.Sys.Print("Offset completed")


        V = FunctionSpace(relabeled_mesh, "CG", 1)
        test = TestFunction(V)
        trial = TrialFunction(V)
        a = inner(grad(trial), grad(test)) * dx
        L = sink_support_mesh * test * dx

        start = time.time()
        solution = Function(V,name="solution")
        problem = LinearVariationalProblem(a, L, solution, bcs=[DirichletBC(V, 0.0, 99)])
        solver = LinearVariationalSolver(problem,
                                solver_parameters={
                                    "ksp_type": "cg",
                                    "ksp_rtol": 1e-6,
                                    "pc_type": "hypre",
                                    "ksp_monitor_true_residual": None})

        solver.solve()
        PETSc.Sys.Print(f"Dirichlet BC test solve completed in {time.time()-start:.2e} s")
        data4pvd.append(solution)
        DG0_Cartesian = FunctionSpace(cartesian_mesh, "DG", 0)
        solution_cartesian = assemble(interpolate(solution, DG0_Cartesian, allow_missing_dofs=True,  default_missing_val=-99))
        solution_np = i2d.firedrake2numpy(solution_cartesian)
        if cartesian_mesh.comm.rank == 0:
            outfilename = os.path.join(out_directory,f"solution_dirichlet.nii.gz")
            nibabel.save(nibabel.Nifti1Image(solution_np, affine), outfilename)
        cartesian_mesh.comm.barrier()

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mri', type=str)
    parser.add_argument('--mesh', type=str, default="simplex", help="type of mesh: [simplex], cartesian")
    parser.add_argument('--out', type=str)
    parser.add_argument('--h5', action='store_true')
    parser.add_argument('--pvd', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    setup(args.mri, args.out, args.mesh, args.h5, args.pvd, args.test)
    
    
