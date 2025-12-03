import nibabel 
import argparse
import numpy as np
from connected_components_tof import connected_components, main_network_equal_one
import os
from scipy.ndimage import gaussian_filter
import pygalmesh
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

def bounding_box(mesh):
    lower = mesh.coordinates.dat.data.min(axis=0)
    upper = mesh.coordinates.dat.data.max(axis=0)
    
    global_lower = []
    global_upper = []
    for i in range(mesh.geometric_dimension()):
        global_lower.append(mesh.comm.allreduce(lower[i], op=MPI.MIN))
        global_upper.append(mesh.comm.allreduce(upper[i], op=MPI.MAX))
    
    return np.array(global_lower), np.array(global_upper)

def setup(mri_directory, 
          out_directory,
            save_h5=False,
            save_pvd=False
            test_dirichlet_bc = False
          ):

    def load_data(mri_directory):
        save_npy = False
        # load tof data and get basic info
        dir_nii = mri_directory
        tof_data = nibabel.load(f"{dir_nii}/TOF.nii.gz")
        dimensions = tof_data.header.get_data_shape()[:3]

        print(f"Data shape: {dimensions=}")
        hx, hy, hz = tof_data.header['pixdim'][1:4]
        lengths = np.array([float(dimensions[0]*hx), 
                            float(dimensions[1]*hy), 
                            float(dimensions[2]*hz)])
        voxel_size = (hx, hy, hz)

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

        sink_support_data = nibabel.load(f"{out_directory}/sink_support.nii.gz")
        sink_support_np = sink_support_data.get_fdata()

        skeleton_data = nibabel.load(f"{out_directory}/skeleton.nii.gz")
        skeleton_np = skeleton_data.get_fdata()

        thickness_data = nibabel.load(f"{out_directory}/thickness.nii.gz")
        thickness_np = thickness_data.get_fdata()

        tof_clean_data = nibabel.load(f"{out_directory}/tof_clean.nii.gz")
        tof_clean_np = tof_clean_data.get_fdata()

        return main_network_np, sink_support_np, skeleton_np, thickness_np, tof_clean_np

    voxel_size, affine, tof_np, brain_mask_np, t1_np, aseg_np = load_data(mri_directory)
    hx, hy, hz = voxel_size
    dimensions = tof_np.shape
    lengths = np.array([float(dimensions[0]*hx), 
                        float(dimensions[1]*hy), 
                        float(dimensions[2]*hz)])
    offset = affine[:3, 3]
    PETSc.Sys.Print(f" voxel_size: {voxel_size}, lengths: {lengths}, offset: {offset}")
    
    # preprocess data
    main_network_np, sink_support_np, skeleton_np, thickness_np, tof_clean_np = load_preprocessed(out_directory)
    
    # reload the mesh from file
    nproc = PETSc.COMM_WORLD.getSize()
    try:
        meshfile = os.path.join(out_directory, f"brain_main_{nproc:04d}.h5")
        PETSc.Sys.Print(f"Loading mesh from {meshfile}")
        if not os.path.exists(meshfile):
            PETSc.Sys.Print(f"Mesh in {meshfile} not found")
            raise FileNotFoundError
        start = time.time() 
        with CheckpointFile(meshfile, 'r', comm=COMM_WORLD) as afile:
            mesh = afile.load_mesh("mesh")
        PETSc.Sys.Print(f"Mesh loaded in {time.time()-start:.2e} s")
    except:
        PETSc.Sys.Print("reading mesh from .msh file")
        start = time.time() 
        mesh = Mesh(os.path.join(out_directory,"brain_main.msh"))
        PETSc.Sys.Print(f"completed in {time.time()-start:.2e} s")
    
    PETSc.Sys.Print("Offset completed")
    lower, upper = bounding_box(mesh)
    PETSc.Sys.Print(f" {lower[0]:.2e}<= x <>{upper[0]:.2e}. Lx={lengths[0]:.2e}")
    PETSc.Sys.Print(f" {lower[1]:.2e}<= y <>{upper[1]:.2e}. Ly={lengths[1]:.2e}")
    PETSc.Sys.Print(f" {lower[2]:.2e}<= z <>{upper[2]:.2e}. Lz={lengths[2]:.2e}") 
    

    PETSc.Sys.Print("Shifting coordinates")
    mesh.coordinates.dat.data[:, 0] -= offset[0]
    mesh.coordinates.dat.data[:, 1] -= offset[1]
    mesh.coordinates.dat.data[:, 2] -= offset[2]
    PETSc.Sys.Print("Offset completed")
    lower, upper = bounding_box(mesh)
    PETSc.Sys.Print(f" {lower[0]:.2e}<= x <>{upper[0]:.2e}. Lx={lengths[0]:.2e}")
    PETSc.Sys.Print(f" {lower[1]:.2e}<= y <>{upper[1]:.2e}. Ly={lengths[1]:.2e}")
    PETSc.Sys.Print(f" {lower[2]:.2e}<= z <>{upper[2]:.2e}. Lz={lengths[2]:.2e}")
    
    cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths)
    PETSc.Sys.Print("Cartesian mesh")
    lower, upper = bounding_box(cartesian_mesh)
    PETSc.Sys.Print(f" {lower[0]:.2e}<= x <>{upper[0]:.2e}. Lx={lengths[0]:.2e}")
    PETSc.Sys.Print(f" {lower[1]:.2e}<= y <>{upper[1]:.2e}. Ly={lengths[1]:.2e}")
    PETSc.Sys.Print(f" {lower[2]:.2e}<= z <>{upper[2]:.2e}. Lz={lengths[2]:.2e}")
    

    interpolate_cartesian = False
    if interpolate_cartesian:
        PETSc.Sys.Print("Numpy to Firedrake Functions on Cartesian grid", end="")
        tof_cartesian = i2d.numpy2firedrake(cartesian_mesh, tof_np, name='tof')
        tof_smooth_cartesian = i2d.numpy2firedrake(cartesian_mesh, tof_clean_np, name='tof_smooth')
        t1_cartesian = i2d.numpy2firedrake(cartesian_mesh, t1_np, name='t1')
        brain_mask_cartesian = i2d.numpy2firedrake(cartesian_mesh, brain_mask_np, name='brain_mask')
        main_network_cartesian = i2d.numpy2firedrake(cartesian_mesh, main_network_np, name='main_network')
        sink_support_cartesian = i2d.numpy2firedrake(cartesian_mesh, sink_support_np, name='sink_support')
        skeleton_cartesian = i2d.numpy2firedrake(cartesian_mesh, skeleton_np, name='skeleton')
        thickness_cartesian = i2d.numpy2firedrake(cartesian_mesh, thickness_np, name='thickness')
        PETSc.Sys.Print(f" - completed in {time.time()-start:.2e}")
        
    
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

    main_network_mesh = interpolate_from_numpy(mesh, main_network_np, lengths, name="main_network")
    #main_network_mesh2, main_network_cartesian2 = interpolate_from_cartesian(mesh, cartesian_mesh, main_network_np, name="main_network2")
    
    #assemble_diff = assemble((main_network_mesh - main_network_mesh2)**2 * dx)
    #PETSc.Sys.Print(f"Difference between two interpolation methods for main network: {assemble_diff:.2e}")
    
    # relabeled mesh to mark the inlet boundary
    my = False
    if my:
        zmin = 0
        start  = time.time()
        PETSc.Sys.Print("intepolate marker for relabeled mesh", end="")
        marker_space = FunctionSpace(mesh, "HDiv Trace", 0)
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
        PETSc.Sys.Print(f"Bounding box lower: {lower}, upper: {upper}")

        PETSc.Sys.Print("Shifting coordinates. I do not know why relabeled mesh shift them back")
        relabeled_mesh.coordinates.dat.data[:, 0] -= offset[0]
        relabeled_mesh.coordinates.dat.data[:, 1] -= offset[1]
        relabeled_mesh.coordinates.dat.data[:, 2] -= offset[2]
        PETSc.Sys.Print("Offset completed")
    else:
        zmin = 0
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
        PETSc.Sys.Print(f" {lower[0]:.2e}<= x <>{upper[0]:.2e}. Lx={lengths[0]:.2e}")
        PETSc.Sys.Print(f" {lower[1]:.2e}<= y <>{upper[1]:.2e}. Ly={lengths[1]:.2e}")
        PETSc.Sys.Print(f" {lower[2]:.2e}<= z <>{upper[2]:.2e}. Lz={lengths[2]:.2e}")
    
        PETSc.Sys.Print("Shifting coordinates. I do not know why relabeled mesh shift them back")
        relabeled_mesh.coordinates.dat.data[:, 0] -= offset[0]
        relabeled_mesh.coordinates.dat.data[:, 1] -= offset[1]
        relabeled_mesh.coordinates.dat.data[:, 2] -= offset[2]
        PETSc.Sys.Print("Offset completed")

    PETSc.Sys.Print(f" Relabeled mesh created")
    lower, upper = bounding_box(relabeled_mesh)
    PETSc.Sys.Print(f" {lower[0]:.2e}<= x <>{upper[0]:.2e}. Lx={lengths[0]:.2e}")
    PETSc.Sys.Print(f" {lower[1]:.2e}<= y <>{upper[1]:.2e}. Ly={lengths[1]:.2e}")
    PETSc.Sys.Print(f" {lower[2]:.2e}<= z <>{upper[2]:.2e}. Lz={lengths[2]:.2e}")
    

    # save as pvd
    use_cartesian_interpolation = False
    if use_cartesian_interpolation:
        DG0 = FunctionSpace(relabeled_mesh, "DG", 0)
        tof_mesh = Function(DG0, name="tof")
        tof_smooth_mesh = Function(DG0, name="tof_smooth")
        t1_mesh = Function(DG0, name="t1")
        main_network_mesh = Function(DG0, name="main_network")
        sink_support_mesh = Function(DG0, name="sink_support")
        skeleton_mesh = Function(DG0, name="skeleton")
        thickness_mesh = Function(DG0, name="thickness")
        brain_mask_mesh = Function(DG0, name="brain_mask")
        
        start  = time.time()
        PETSc.Sys.Print("intepolate tof", end="")
        tof_mesh.interpolate(tof_cartesian)
        PETSc.Sys.Print(f" - completed in {time.time()-start:.2e} s")
        tof_smooth_mesh.interpolate(tof_smooth_cartesian)
        t1_mesh.interpolate(t1_cartesian)
        main_network_mesh.interpolate(main_network_cartesian)
        sink_support_mesh.interpolate(sink_support_cartesian)
        skeleton_mesh.interpolate(skeleton_cartesian)
        thickness_mesh.interpolate(thickness_cartesian)
        brain_mask_mesh.interpolate(brain_mask_cartesian)
        test = TestFunction(DG0)
        size = assemble(test *dx)
        size_mesh = Function(DG0, name="size_mesh")
        with size_mesh.dat.vec as s, size.dat.vec as h:
            h.copy(s)
    else:
        tof_mesh = i2d.numpy2firedrake(relabeled_mesh, tof_np, name='tof',lengths=lengths)
        tof_smooth_mesh = i2d.numpy2firedrake(relabeled_mesh, tof_clean_np, name='tof_smooth',lengths=lengths)
        t1_mesh = i2d.numpy2firedrake(relabeled_mesh, t1_np, name='t1',lengths=lengths)
        brain_mask_mesh = i2d.numpy2firedrake(relabeled_mesh, brain_mask_np, name='brain_mask',lengths=lengths)
        main_network_mesh = i2d.numpy2firedrake(relabeled_mesh, main_network_np, name='main_network',lengths=lengths)
        sink_support_mesh = i2d.numpy2firedrake(relabeled_mesh, sink_support_np, name='sink_support',lengths=lengths)
        skeleton_mesh = i2d.numpy2firedrake(relabeled_mesh, skeleton_np, name='skeleton',lengths=lengths)
        thickness_mesh = i2d.numpy2firedrake(relabeled_mesh, thickness_np, name='thickness',lengths=lengths)
    
    lower, upper = bounding_box(relabeled_mesh)
    PETSc.Sys.Print(f" {lower[0]:.2e}<= x <>{upper[0]:.2e}. Lx={lengths[0]:.2e}")
    PETSc.Sys.Print(f" {lower[1]:.2e}<= y <>{upper[1]:.2e}. Ly={lengths[1]:.2e}")
    PETSc.Sys.Print(f" {lower[2]:.2e}<= z <>{upper[2]:.2e}. Lz={lengths[2]:.2e}")
    

    PETSc.Sys.Print("Shifting coordinates of relabeled mesh")
    relabeled_mesh.coordinates.dat.data[:, 0] += offset[0]
    relabeled_mesh.coordinates.dat.data[:, 1] += offset[1]
    relabeled_mesh.coordinates.dat.data[:, 2] += offset[2]
    PETSc.Sys.Print("Offset completed")

    lower, upper = bounding_box(cartesian_mesh)
    PETSc.Sys.Print(f" {lower[0]:.2e}<= x <>{upper[0]:.2e}. Lx={lengths[0]:.2e}")
    PETSc.Sys.Print(f" {lower[1]:.2e}<= y <>{upper[1]:.2e}. Ly={lengths[1]:.2e}")
    PETSc.Sys.Print(f" {lower[2]:.2e}<= z <>{upper[2]:.2e}. Lz={lengths[2]:.2e}")
    
    data4pvd = [tof_mesh,
                            brain_mask_mesh,
                            main_network_mesh,
                            sink_support_mesh,
                            tof_smooth_mesh]


    
    if test_dirichlet_bc:
        V = FunctionSpace(relabeled_mesh, "CG", 1)
        test = TestFunction(V)
        trial = TrialFunction(V)
        a = inner(grad(trial), grad(test)) * dx
        L = - sink_support_mesh * test * dx

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
        solution_cartesian = i2d.firedrake2numpy(solution, dimensions, fill=-99)
        solution_np = i2d.firedrake2numpy(solution_cartesian, dimensions, lengths)
        if cartesian_mesh.comm.rank == 0:
            outfilename = os.path.join(out_directory,f"solution_dirichlet.nii.gz")
            nibabel.save(nibabel.Nifti1Image(solution_np, affine), outfilename)
        cartesian_mesh.comm.barrier()

    if save_pvd:
        outfilename = os.path.join(out_directory,f"inputs.pvd")
        VTKFile(outfilename).write(*data4pvd)

    if save_h5:
        n_proc = COMM_WORLD.size
        h5_filename = os.path.join(mri_directory,
                                f"inputs_nproc{n_proc}.h5")
        PETSc.Sys.Print(f"Saving to {h5_filename}", end="")
        print("name",relabeled_mesh.name)
        with CheckpointFile(h5_filename, 'w', comm=COMM_WORLD) as afile:
            afile.save_mesh(relabeled_mesh,"relabeled_mesh")
            PETSc.Sys.Print(f" mesh ", end="")
            afile.save_function(tof_mesh)
            PETSc.Sys.Print(f" tof ", end="")
            afile.save_function(t1_mesh)
            PETSc.Sys.Print(f" t1 ", end="")
            afile.save_function(brain_mask_mesh)
            PETSc.Sys.Print(f" brain_mask ", end="")
            afile.save_function(main_network_mesh)
            PETSc.Sys.Print(f" main_network ", end="")
            afile.save_function(sink_support_mesh)
            PETSc.Sys.Print(f" sink_support ", end="")
            afile.save_function(skeleton_mesh)
            PETSc.Sys.Print(f" skeleton ", end="")
            afile.save_function(thickness_mesh)
            PETSc.Sys.Print(f" thickness ", end="")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mri', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--h5', action='store_true')
    parser.add_argument('--pvd', action='store_true')
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    setup(args.mri, args.out, args.h5, args.pvd, args.test)
    
    
