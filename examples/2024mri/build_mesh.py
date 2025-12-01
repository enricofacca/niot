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
            threshold_tof_4_main_network,
            blur_tof_4_main_network = 0.0, 
            blur_tof_4_mesh = 0.0, 
            build=True,
          save_h5=False):

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
    
    voxel_size, affine, tof_np, brain_mask_np, t1_np, aseg_np = load_data(mri_directory)
    hx, hy, hz = voxel_size


    def set_main_network(tof_np, threshold_tof, blur_tof, hx):
        """
        Get the main network, its skeleton, and local thickness.
        Blur is applied before connected components analysis and improve skeletonization.
        """
        # blur tof, separate main network, and keep largest connected component
        if blur_tof> 0:
            print(f" - applying gaussian blur {blur_tof:.2e}")
            tof_main_network_np = gaussian_filter(tof_np, sigma=blur_tof * hx)
        else:
            tof_main_network_np = tof_np.copy()
        labels_np, nlabels = connected_components(tof_main_network_np, threshold_tof)
        labels_np = main_network_equal_one(labels_np, nlabels, tof_np)


        # save as nifti 
        main_network = np.zeros_like(labels_np, dtype=np.uint8)
        main_network[labels_np == 1] = 1
        
        # get the skeleton of main network
        skeleton_np = skeletonize(main_network)
        skeleton_np = skeleton_np.astype(np.uint8)

        # compute local thickness of the main network
        thickness_np = lt.local_thickness(main_network)
        # scale by thickness 
        thickness_np *= hx

        return main_network, skeleton_np, thickness_np
    
    
    def set_tof4mesh(tof_np, options_dict):
        """
        Prepocess tof of assign a label to the mesh generation
        """
        mode = options_dict.get("mode","gaussian_blur")
        if mode == "gaussian":
            suboption = options_dict.get("gaussian")
            blur_tof = suboption.get("blur",0.0)
            hx = suboption.get("hx",1.0)
            if abs(blur_tof) < 1e-10:
                return tof_smooth_np
            else:
                print(f" - applying gaussian blur {blur_tof:.2e}",end="")
                tof_smooth_np = gaussian_filter(tof_np, sigma=blur_tof*hx)
                print(f" - done",end="")
                return tof_smooth_np
        elif mode == "dilation":
            suboption = options_dict.get("dilation")
            mask = suboption.get("mask",None)
            iterations = suboption.get("iterations",2)
            threshold = suboption.get("threshold", 180)
            structure = tof_np > threshold
            tof_smooth_np = binary_dilation(structure,iterations=iterations,mask=mask)
            print(f" - done",end="")
            return tof_smooth_np
        else:
            raise ValueError(f"Unknown mode {mode} for tof preprocessing")        
        
    
    def set_sink_support(aseg_np, main_network, dilatation_iterations, mask_brain_np):
        """
        set the sink support based on aseg
        """
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
        sink_support_np = binary_dilation(sink_support_np,
                                        iterations=dilatation_iterations,mask=mask_brain_np)
        sink_support_np = sink_support_np.astype(dtype=np.uint8)
        # remove main network from sink
        sink_support_np[main_network > 0 ] = 0
        

        return sink_support_np
    
    # process parameters
    main_network_np, skeleton_np, thickness_np = set_main_network(tof_np, 
                                                                  threshold_tof_4_main_network, 
                                                                  blur_tof_4_main_network, hx)
    sink_support_np = set_sink_support(aseg_np, main_network_np, 2, brain_mask_np)
    

    i2d.save_slice(sink_support_np, output_dir = "support_slices")
    options_dict = {"mode" : "dilation",
                    "gaussian": {
                        "blur": blur_tof_4_mesh,
                        "hx": hx},
                    "dilation": {
                        "mask": brain_mask_np,
                        "iterations": 3,
                        "threshold": 180
                        }
                    }

    tof_clean_np = gaussian_filter(tof_np, sigma = hx * 1.5) 
    tof_smooth_np = set_tof4mesh(tof_clean_np, options_dict).astype(np.float32)
    


    # save as nifti
    for var, name in zip([main_network_np, skeleton_np, thickness_np, sink_support_np],
                            ["main_network", "skeleton", "thickness", "sink_support"]):
        outfilename = os.path.join(mri_directory,f"{name}_blur{blur_tof_4_main_network:.2e}_t{threshold_tof_4_main_network:.2e}.nii.gz")
        print(f"Saving main network {outfilename}")
        nibabel.save(nibabel.Nifti1Image(var, affine), outfilename)
    
    outfilename = os.path.join(mri_directory,
                               f"tof_mesh.nii.gz")
    print(f"Saving smoothed tof {outfilename}")
    nibabel.save(nibabel.Nifti1Image(tof_smooth_np, affine), outfilename)

    outfilename = os.path.join(mri_directory,
                               f"tof_clean.nii.gz")
    print(f"Saving smoothed tof {outfilename}")
    nibabel.save(nibabel.Nifti1Image(tof_clean_np, affine), outfilename)

    #
    # Build mesh
    #
    def build_mesh(voxel_size, brain_mask_np, sink_support_np, tof_support_np, main_network):
        mask = brain_mask_np.copy()
        label_sink = 4
        mask[sink_support_np > 0 ] = label_sink
        
        # blur tof
        label_tof = 3
        mask[tof_support_np > 0 ] = label_tof
        
        
        label_main = 2
        # remove everything outside domain
        mask[brain_mask_np < 1 ] = 0
        # restore main_network
        mask[main_network > 0 ] = label_main 
        mask = mask.astype(np.uint8)

        outfilename = os.path.join(mri_directory,
                               f"mask_mesher.nii.gz")
        print(f"Saving mask mesher {outfilename}")
        nibabel.save(nibabel.Nifti1Image(mask, affine), outfilename)
        
        
        PETSc.Sys.Print("volex size:", hx, hy, hz)
        scale = 2
        mesh_pygal = pygalmesh.generate_from_array(
                mask,
                voxel_size, 
                max_facet_distance=scale*hx,
                max_cell_circumradius={
                    "default": scale*16*hx, 
                    label_main: scale*hx,
                    label_tof: scale*hx,
                    label_sink: scale*8*hx
                },
            )
        return mesh_pygal, mask
        
    if build:
        mesh_pygal, mask_np = build_mesh(voxel_size,
                                         brain_mask_np,
                                         sink_support_np,
                                         tof_smooth_np,
                                         main_network_np)
        mesh_pygal.write(os.path.join(mri_directory,"brain_main.vtu"))

        
        
        writer = partial(meshio.gmsh.write, fmt_version="2.2", binary=True)
        writer(os.path.join(mri_directory,"brain_main.msh"), mesh_pygal)
    
    dimensions = tof_np.shape
    lengths = np.array([float(dimensions[0]*hx), 
                        float(dimensions[1]*hy), 
                        float(dimensions[2]*hz)])
    cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths)
    tof_cartesian = i2d.numpy2firedrake(cartesian_mesh, tof_np, name='tof')
    tof_smooth_cartesian = i2d.numpy2firedrake(cartesian_mesh, tof_smooth_np, name='tof_smooth')
    t1_cartesian = i2d.numpy2firedrake(cartesian_mesh, t1_np, name='t1')
    brain_mask_cartesian = i2d.numpy2firedrake(cartesian_mesh, brain_mask_np, name='brain_mask')
    main_network_cartesian = i2d.numpy2firedrake(cartesian_mesh, main_network_np, name='main_network')
    sink_support_cartesian = i2d.numpy2firedrake(cartesian_mesh, sink_support_np, name='sink_support')
    skeleton_cartesian = i2d.numpy2firedrake(cartesian_mesh, skeleton_np, name='skeleton')
    thickness_cartesian = i2d.numpy2firedrake(cartesian_mesh, thickness_np, name='thickness')
    





    # reload the mesh from file
    PETSc.Sys.Print("reading mesh from .msh file")
    start = time.time() 
    mesh = Mesh(os.path.join(mri_directory,"brain_main.msh"))
    PETSc.Sys.Print(f"completed in {time.time()-start:.2e} s")
    zmin = 0.0
    DG0 = FunctionSpace(mesh, "DG", 0)
    main_network_mesh = Function(DG0, name="main_network_mesh")
    main_network_mesh.interpolate(main_network_cartesian)
        

    # relabeled mesh to mark the inlet boundary
    marker_space = FunctionSpace(mesh, "HDiv Trace", 0)
    main_network_indicator = Function(marker_space, name="main_network_indicator")
    x,y,z = mesh.coordinates
    main_network_indicator.interpolate(main_network_mesh * conditional(z-zmin < hx,1,0))
    elem = main_network_indicator.topological.function_space().ufl_element()
    PETSc.Sys.Print(elem.family(),elem.degree())
    relabeled_mesh = MyRelabeledMesh(mesh, [main_network_indicator], 
                                     [99],
                                     boundary_only=True,
                                    name="relabeled_mesh")
    PETSc.Sys.Print("Relabeled mesh created")

    # save as pvd
    DG0 = FunctionSpace(relabeled_mesh, "DG", 0)
    tof_mesh = Function(DG0, name="tof")
    tof_smooth_mesh = Function(DG0, name="tof_smooth")
    t1_mesh = Function(DG0, name="t1")
    main_network_mesh = Function(DG0, name="main_network")
    sink_support_mesh = Function(DG0, name="sink_support")
    skeleton_mesh = Function(DG0, name="skeleton")
    thickness_mesh = Function(DG0, name="thickness")
    brain_mask_mesh = Function(DG0, name="brain_mask")
    
    
    tof_mesh.interpolate(tof_cartesian)
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

    PETSc.Sys.Print("Interpolation completed")
    
    #VTKFile("labeled_mesh.pvd").write(relabeled_mesh)
    # shift coordinate of the relabeled mesh
    offset = affine[:3, 3]
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

    outfilename = os.path.join(mri_directory,f"inputs_blur{blur_tof_4_main_network:.2e}_thr{threshold_tof_4_main_network:.2e}_TOF_blur{blur_tof_4_mesh:.2e}.pvd")
    VTKFile(outfilename).write(tof_mesh,
                               brain_mask_mesh,
                               main_network_mesh,
                               sink_support_mesh,size_mesh,tof_smooth_mesh)

    if save_h5:
        n_proc = COMM_WORLD.size
        h5_filename = os.path.join(mri_directory,
                                   f"inputs_nproc{n_proc}" + 
                                   f"_MAIN_blur{blur_tof_4_main_network:.2e}" +
                                   f"_thr{threshold_tof_4_main_network:.2e}_TOF_blur{blur_tof_4_mesh:.2e}.h5")
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
    parser.add_argument('--threshold', type=float, default=250,
                        help="Threshold for Tof. Default is 250.")
    parser.add_argument('--blur_main', type=float, default=0.0, 
                        help="Blur for connected components. If 0, no blur is applied.")
    parser.add_argument('--blur_mesh', type=float, default=0.0, 
                        help="Blur for connected components. If 0, no blur is applied.")
    parser.add_argument('--read', action='store_true')
    parser.add_argument('--h5', action='store_true')
    args = parser.parse_args()

    setup(args.mri, args.threshold, args.blur_main, args.blur_mesh, not args.read, args.h5)
    
    
