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
import local_thickness as lt
from skimage.morphology import skeletonize


def setup(mri_directory, threshold, blur = 0.0, blur_tof = 0.0, build=True):
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
    
    # get brain mask
    nii_file = f"{dir_nii}/brain_mask_smooth.nii.gz"
    brain_mask_data = nibabel.load(nii_file)
    brain_mask_np = brain_mask_data.get_fdata()  
    

    # load tof data
    # tof_np = tof_data.get_fdata()
    file_nii = f"{dir_nii}/TOF.nii.gz"
    tof_data = nibabel.load(file_nii)
    tof_np = tof_data.get_fdata()

    # blur tof 
    if blur > 0:
        print(f" - applying gaussian blur {blur:.2e}",end="")
        tof_np = gaussian_filter(tof_np, sigma=blur*hx)
        print(f" - done",end="")

    # separe connected components
    labels_np, nlabels = connected_components(tof_np, threshold)
    labels_np = main_network_equal_one(labels_np, nlabels, tof_np)


    # save as nifti 
    main_network = np.zeros_like(labels_np, dtype=np.uint8)
    main_network[labels_np == 1] = 1
    outfilename = f"main_network_mesh.nii.gz"
    print(f"Saving main network {outfilename}")
    nibabel.save(nibabel.Nifti1Image(main_network, tof_data.affine), 
                 outfilename)
    

    # get the skeleton of main network
    skeleton_np = skeletonize(main_network)
    skeleton_np = skeleton_np.astype(np.uint8)

    # compute local thickness of the main network
    thickness_np = lt.local_thickness(main_network)
    # scale by thickness 
    thickness_np *= hx

    

    # load tof data
    # tof_np = tof_data.get_fdata()
    file_nii = f"{dir_nii}/aseg.nii.gz"
    data_aseg = nibabel.load(file_nii)
    aseg_np = data_aseg.get_fdata()
    

    # load main network
    file_nii = f"{dir_nii}/T1.nii.gz"
    t1_data = nibabel.load(file_nii)
    t1_np = t1_data.get_fdata()
    
    # blur tof 
    if blur_tof > 0:
        print(f" - applying gaussian blur {blur_tof:.2e}",end="")
        tof_smooth_np = gaussian_filter(tof_np, sigma=blur_tof*hx)
        print(f" - done",end="")

    

    # get the sink
    sink_support_np = np.zeros_like(aseg_np, dtype=np.uint8)
    # label described in https://surfer.nmr.mgh.harvard.edu/fswiki/FsTutorial/AnatomicalROI/FreeSurferColorLUT
    empty_markers = [4, # left-lateral ventricle
                     5, # left-inf-lat-vent
                     14, # 3rd ventricle
                     15, # 4th ventricle
                     24, # CSF
                     43, # right-lateral ventricle
                     44, # right-inf-lat-ventricle 
                     ]
    
    sink_support_np[aseg_np > 0] = 1
    for label in empty_markers:
        sink_support_np[aseg_np == label] = 0
    # remove main network from sink
    sink_support_np[main_network > 0 ] = 0


    cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths)
    tof_cartesian = i2d.numpy2firedrake(cartesian_mesh, tof_np, name='tof')
    t1_cartesian = i2d.numpy2firedrake(cartesian_mesh, t1_np, name='t1')
    main_network_cartesian = i2d.numpy2firedrake(cartesian_mesh, main_network, name='main_network')
    sink_support_cartesian = i2d.numpy2firedrake(cartesian_mesh, sink_support_np, name='sink_support')
    skeleton_cartesian = i2d.numpy2firedrake(cartesian_mesh, skeleton_np, name='skeleton')
    thickness_cartesian = i2d.numpy2firedrake(cartesian_mesh, thickness_np, name='thickness')



    #
    # # Build mesh
    #
    mask = brain_mask_np.copy()
    label_sink = 4
    mask[sink_support_np > 0 ] = label_sink
    
    
    label_tof = 3
    t = 0.175 * tof_smooth_np.max()
    mask[tof_smooth_np > t ] = label_tof
    
    
    label_main = 2
    # remove everything outside domain
    mask[brain_mask_np < 1 ] = 0
    # restore main_network
    mask[main_network > 0 ] = label_main 
    mask = mask.astype(np.uint8)

    voxel_size = (hx, hy, hz)

    if build:
        mesh = pygalmesh.generate_from_array(
            mask,
            voxel_size, 
            max_facet_distance=0.2,
            max_cell_circumradius={
                "default": 2.0, 
                label_main: 0.5,
                label_tof: 0.5,
                label_sink: 2.0
            },
        )
        mesh.write("brain_main.vtu")
    
        writer = partial(meshio.gmsh.write, fmt_version="2.2", binary=True)
        writer("brain_main.msh", mesh)

    
    mesh = Mesh("brain_main.msh")
    zmin = cartesian_mesh.zmin
    print(f"{zmin=}")
    DG0 = FunctionSpace(mesh, "DG", 0)
    main_network_mesh = Function(DG0, name="main_network_mesh")
    main_network_mesh.interpolate(main_network_cartesian)
    

    # relabeled mesh to mark the inlet boundary
    marker_space = FunctionSpace(mesh, "HDiv Trace", 0)
    main_network_indicator = Function(marker_space, name="main_network_indicator")
    x,y,z = mesh.coordinates
    main_network_indicator.interpolate(main_network_mesh * conditional(z-zmin < hx,1,0)) 
    relabeled_mesh = MyRelabeledMesh(mesh, [main_network_indicator], 
                                     [99],
                                     boundary_only=True,
                                    name="relabeled_mesh")
    VTKFile("labeled_mesh.pvd").write(relabeled_mesh)
    
    # save as pvd
    VTKFile("tof_mesh.pvd").write(tof_mesh)
    DG0 = FunctionSpace(relabeled_mesh, "DG", 0)
    tof_mesh = Function(DG0, name="tof")
    t1_mesh = Function(DG0, name="t1")
    main_network_mesh = Function(DG0, name="main_network")
    sink_support_mesh = Function(DG0, name="sink_support")
    skeleton_mesh = Function(DG0, name="skeleton")
    thickness_mesh = Function(DG0, name="thickness")
    
    tof_mesh.interpolate(tof_cartesian)
    t1_mesh.interpolate(t1_cartesian)
    main_network_mesh.interpolate(main_network_cartesian)
    sink_support_mesh.interpolate(sink_support_cartesian)
    skeleton_mesh.interpolate(skeleton_cartesian)
    thickness_mesh.interpolate(thickness_cartesian)
    
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

    n_proc = COMM_WORLD.size
    h5_filename = f"inputs_nproc{n_proc}.h5"
    PETSc.Sys.Print(f"Saving to {h5_filename}", end="")
    print("name",relabeled_mesh.name)
    with CheckpointFile(h5_filename, 'w', comm=COMM_WORLD) as afile:
        afile.save_mesh(relabeled_mesh,"relabeled_mesh")
        PETSc.Sys.Print(f" mesh ", end="")
        afile.save_function(tof_mesh)
        PETSc.Sys.Print(f" tof ", end="")
        afile.save_function(t1_mesh)
        PETSc.Sys.Print(f" t1 ", end="")
        afile.save_function(main_network_mesh)
        PETSc.Sys.Print(f" main_network ", end="")
        afile.save_function(sink_support_mesh)
        PETSc.Sys.Print(f" sink_support ", end="")
        afile.save_function(skeleton_mesh)
        PETSc.Sys.Print(f" skeleton ", end="")
        afile.save_function(thickness_mesh)
        PETSc.Sys.Print(f" thickness ", end="")

    outfilename = f"mask_mesh.nii.gz"
    print(f"Saving main network {outfilename}")
    nibabel.save(nibabel.Nifti1Image(mask, tof_data.affine), 
                 outfilename)


        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mri', type=str)
    parser.add_argument('--threshold', type=float, default=250,
                        help="Threshold for Tof. Default is 250.")
    parser.add_argument('--blur_main', type=float, default=0.0, 
                        help="Blur for connected components. If 0, no blur is applied.")
    parser.add_argument('--blur_tof', type=float, default=0.0, 
                        help="Blur for connected components. If 0, no blur is applied.")
    parser.add_argument('--read', action='store_true')
    args = parser.parse_args()

    setup(args.mri, args.threshold, args.blur_main, args.blur_tof, not args.read)
    
    
