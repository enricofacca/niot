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

def setup(mri_directory, 
          out_directory,
        threshold_tof_4_main_network,
        blur_tof_4_main_network = 0.0, 
        blur_tof_4_mesh = 0.0, 
            build=True,
          save_h5=False,
            firedrake_conversion=True,
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
    PETSc.Sys.Print("reading mesh from .msh file")
    start = time.time() 
    mesh = Mesh(os.path.join(out_directory,"brain_main.msh"))
    PETSc.Sys.Print(f"completed in {time.time()-start:.2e} s")
    PETSc.Sys.Print("Shifting coordinates")
    mesh.coordinates.dat.data[:, 0] -= offset[0]
    mesh.coordinates.dat.data[:, 1] -= offset[1]
    mesh.coordinates.dat.data[:, 2] -= offset[2]
    PETSc.Sys.Print("Offset completed")
    PETSc.Sys.Print(f" xmin {mesh.coordinates.dat.data[:,0].min():.2f}, xmax {mesh.coordinates.dat.data[:,0].max():.2f}")
    PETSc.Sys.Print(f" ymin {mesh.coordinates.dat.data[:,1].min():.2f}, ymax {mesh.coordinates.dat.data[:,1].max():.2f}")
    PETSc.Sys.Print(f" zmin {mesh.coordinates.dat.data[:,2].min():.2f}, zmax {mesh.coordinates.dat.data[:,2].max():.2f}")
    PETSc.Sys.Print(f" lengths: {lengths}") 
    
    PETSc.Sys.Print("Numpy to Firedrake Functions on Cartesian grid", end="")
    cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths)
    tof_cartesian = i2d.numpy2firedrake(cartesian_mesh, tof_np, name='tof')
    tof_smooth_cartesian = i2d.numpy2firedrake(cartesian_mesh, tof_clean_np, name='tof_smooth')
    t1_cartesian = i2d.numpy2firedrake(cartesian_mesh, t1_np, name='t1')
    brain_mask_cartesian = i2d.numpy2firedrake(cartesian_mesh, brain_mask_np, name='brain_mask')
    main_network_cartesian = i2d.numpy2firedrake(cartesian_mesh, main_network_np, name='main_network')
    sink_support_cartesian = i2d.numpy2firedrake(cartesian_mesh, sink_support_np, name='sink_support')
    skeleton_cartesian = i2d.numpy2firedrake(cartesian_mesh, skeleton_np, name='skeleton')
    thickness_cartesian = i2d.numpy2firedrake(cartesian_mesh, thickness_np, name='thickness')
    PETSc.Sys.Print(f" - completed in {time.time()-start:.2e}")
    
    
    
    # start = time.time()
    # PETSc.Sys.Print("Numpy to Firedrake Functions on Cartesian grid", end="")
    # cartesian_mesh =  i2d.cartesian_grid_3d(dimensions, lengths)
    # tof_cartesian = i2d.numpy2firedrake(cartesian_mesh, tof_np, name='tof')
    # tof_smooth_cartesian = i2d.numpy2firedrake(cartesian_mesh, tof_clean_np, name='tof_smooth')
    # t1_cartesian = i2d.numpy2firedrake(cartesian_mesh, t1_np, name='t1')
    # brain_mask_cartesian = i2d.numpy2firedrake(cartesian_mesh, brain_mask_np, name='brain_mask')
    # main_network_cartesian = i2d.numpy2firedrake(cartesian_mesh, main_network_np, name='main_network')
    # sink_support_cartesian = i2d.numpy2firedrake(cartesian_mesh, sink_support_np, name='sink_support')
    # skeleton_cartesian = i2d.numpy2firedrake(cartesian_mesh, skeleton_np, name='skeleton')
    # thickness_cartesian = i2d.numpy2firedrake(cartesian_mesh, thickness_np, name='thickness')
    # PETSc.Sys.Print(f" - completed in {time.time()-start:.2e}")

    # save as pvd
    # DG0 = FunctionSpace(mesh, "DG", 0)
    # tof_mesh = Function(DG0, name="tof")
    # tof_smooth_mesh = Function(DG0, name="tof_smooth")
    # t1_mesh = Function(DG0, name="t1")
    # main_network_mesh = Function(DG0, name="main_network")
    # sink_support_mesh = Function(DG0, name="sink_support")
    # skeleton_mesh = Function(DG0, name="skeleton")
    # thickness_mesh = Function(DG0, name="thickness")
    # brain_mask_mesh = Function(DG0, name="brain_mask")
        
    DG0 = FunctionSpace(mesh, "DG", 0)
    main_network_mesh = Function(DG0, name="main_network")
    main_network_mesh.interpolate(main_network_cartesian)
    main_network_mesh = i2d.numpy2firedrake(mesh, main_network_np, name='main_network',lengths=lengths)
        
    # relabeled mesh to mark the inlet boundary
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
    

    # save as pvd
    DG0 = FunctionSpace(relabeled_mesh, "DG", 0)
    tof_mesh = i2d.numpy2firedrake(relabeled_mesh, tof_np, name='tof',lengths=lengths)
    tof_smooth_mesh = i2d.numpy2firedrake(relabeled_mesh, tof_clean_np, name='tof_smooth',lengths=lengths)
    t1_mesh = i2d.numpy2firedrake(relabeled_mesh, t1_np, name='t1',lengths=lengths)
    brain_mask_mesh = i2d.numpy2firedrake(relabeled_mesh, brain_mask_np, name='brain_mask',lengths=lengths)
    main_network_mesh = i2d.numpy2firedrake(relabeled_mesh, main_network_np, name='main_network',lengths=lengths)
    sink_support_mesh = i2d.numpy2firedrake(relabeled_mesh, sink_support_np, name='sink_support',lengths=lengths)
    skeleton_mesh = i2d.numpy2firedrake(relabeled_mesh, skeleton_np, name='skeleton',lengths=lengths)
    thickness_mesh = i2d.numpy2firedrake(relabeled_mesh, thickness_np, name='thickness',lengths=lengths)
    
    PETSc.Sys.Print("Shifting coordinates of relabeled mesh")
    relabeled_mesh.coordinates.dat.data[:, 0] += offset[0]
    relabeled_mesh.coordinates.dat.data[:, 1] += offset[1]
    relabeled_mesh.coordinates.dat.data[:, 2] += offset[2]
    PETSc.Sys.Print("Offset completed")





    test_dirichlet_bc = False
    if test_dirichlet_bc:
        V = FunctionSpace(relabeled_mesh, "CG", 1)
        test = TestFunction(V)
        trial = TrialFunction(V)
        a = inner(grad(trial), grad(test)) * dx
        L = sink_support_mesh * test * dx

        solution = Function(V,name="solution")
        problem = LinearVariationalProblem(a, L, solution, bcs=[DirichletBC(V, 0.0, 99)])
        solver = LinearVariationalSolver(problem,
                                solver_parameters={
                                    "ksp_type": "cg",
                                    "ksp_rtol": 1e-6,
                                    "pc_type": "hypre"})
        solver.solve()
        VTKFile("direchlet.pvd").write(solution,sink_support_mesh)

    outfilename = os.path.join(out_directory,f"inputs.pvd")
    VTKFile(outfilename).write(tof_mesh,
                            brain_mask_mesh,
                            main_network_mesh,
                            sink_support_mesh,
                            tof_smooth_mesh)

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
    args = parser.parse_args()

    setup(args.mri, args.out, args.h5)
    
    
