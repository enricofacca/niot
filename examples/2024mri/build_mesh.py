import nibabel 
import argparse
import numpy as np
from connected_components_tof import connected_components, main_network_equal_one, find_external_network
import os
from scipy.ndimage import gaussian_filter
from firedrake import *
from niot import image2dat as i2d
from mwe import MyRelabeledMesh
import meshio
from firedrake import CheckpointFile, COMM_WORLD, PETSc, VTKFile, DirichletBC, FunctionSpace, Function, TestFunction, TrialFunction, inner, grad, dx, conditional
import localthickness as lt
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation
import time


def export_voxel_to_gmsh(array_3d, voxel_size=1.0):
    """
    Converts a 3D binary numpy array into a Gmsh (.msh) file using hexahedral elements.
    Only cells where array_3d == 1 are converted into mesh elements.
    
    Parameters:
    -----------
    array_3d : np.ndarray
        3D array of 0s and 1s.
    filename : str
        Output path for the .msh file.
    voxel_size : float
        The physical side length of each cube.
    """
    
    # 1. Identify active voxel indices (where value is 1)
    # Using np.argwhere returns an (N, 3) array of [z, y, x]
    z_idx, y_idx, x_idx = np.where(array_3d == 1)
    num_cubes = len(z_idx)
    
    if num_cubes == 0:
        print("Warning: The provided array is empty (all zeros). No mesh generated.")
        return

    # 2. Generate unique nodes
    # A cube at (i, j, k) has 8 vertices. 
    # To avoid duplicate nodes at shared corners, we define the global grid of possible nodes.
    nx, ny, nz = array_3d.shape
    
    # The coordinate grid for nodes (vertices) has dimensions (N+1)
    # We only want to export nodes that are actually part of an active cube.
    # However, for simplicity and performance in smaller/medium grids, 
    # we can map cube indices to a global node indexing system.
    
    def get_node_idx(iz, iy, ix):
        return iz * (ny + 1) * (nx + 1) + iy * (nx + 1) + ix

    # Define the 8 relative offsets for a hexahedron in Gmsh ordering (Type 5)
    # Gmsh Hexahedron node ordering:
    # 0: (0,0,0), 1: (1,0,0), 2: (1,1,0), 3: (0,1,0)  <- Bottom face
    # 4: (0,0,1), 5: (1,0,1), 6: (1,1,1), 7: (0,1,1)  <- Top face
    offsets = np.array([
        [0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0],
        [1, 0, 0], [1, 0, 1], [1, 1, 1], [1, 1, 0]
    ])
    
    # Construct elements array (num_cubes, 8)
    # We add the offsets to our base (z, y, x) indices
    cells_nodes = []
    for dx, dy, dz in offsets:
        cells_nodes.append(get_node_idx(x_idx + dx, y_idx + dy, z_idx + dz))
    
    # Stack to get (num_cubes, 8)
    hexa_cells = np.stack(cells_nodes, axis=1)
    
    # 3. Collect unique nodes and remap
    unique_node_indices, inverse_map = np.unique(hexa_cells, return_inverse=True)
    hexa_cells_remapped = inverse_map.reshape(hexa_cells.shape)
    
    # 4. Calculate physical coordinates for unique nodes
    # Reconstruct (x, y, z) from the flat unique_node_indices
    u_iz = unique_node_indices // ((ny + 1) * (nx + 1))
    remainder = unique_node_indices % ((ny + 1) * (nx + 1))
    u_iy = remainder // (nx + 1)
    u_ix = remainder % (nx + 1)
    
    points = np.stack([u_ix, u_iy, u_iz], axis=1).astype(float) * voxel_size
    
    # 5. Create meshio object and write
    # 'hexahedron' is the meshio key for 8-node bricks
    cells = [("hexahedron", hexa_cells_remapped)]
    
    
    mesh = meshio.Mesh(points=points, cells=cells)

    return mesh


def setup(mri_directory, 
          out_directory,
        threshold_tof_4_main_network,
        blur_tof_4_main_network = 0.0, 
        blur_tof_4_mesh = 0.0, 
            build=True,
          save_h5=False,
            firedrake_conversion=True,
            build_tof_mesh=False,
            cell_type="tetrahedron"
          ):

    def load_data(mri_directory):
        save_npy = False
        # load tof data and get basic info
        dir_nii = mri_directory
        tof_data = nibabel.load(f"{dir_nii}/TOF.nii.gz")
        dimensions = tof_data.header.get_data_shape()[:3]

        print(f"Data shape: {dimensions=}")
        hx, hy, hz = tof_data.header['pixdim'][1:4]
        lengths = np.array([hx,hy,hz]) * np.array(dimensions)
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

    dimensions = tof_np.shape
    lengths = np.array([hx,hy,hz]) * np.array(dimensions)
    
    print(f"{hx=:.10e} {hy=:.10e} {hz=:.10e}")
    print(f"voxel_size: {voxel_size}, lengths: {lengths}")
    h_new = lengths/np.array(dimensions)
    print(f"New voxel size: {h_new[0]=:.10e}, {h_new[1]=:.10e}, {h_new[2]=:.10e}")

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
        
        # define main network 
        main_network = np.zeros_like(labels_np, dtype=np.uint8)
        main_network[labels_np == 1] = 1
        
        # get the skeleton of main network
        skeleton_np = skeletonize(main_network)
        skeleton_np = skeleton_np.astype(np.uint8)

        # compute local thickness of the main network
        thickness_np = lt.local_thickness(main_network)
        # scale by thickness 
        thickness_np *= hx

        # find external network
        external_network_np = find_external_network(labels_np)

        return main_network, skeleton_np, thickness_np, external_network_np
    
    
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
        
    
    def set_sink_support(aseg_np, main_network):
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

        # remove main network from sink
        sink_support_np[main_network_np > 0 ] = 0

        return sink_support_np
    
    # process parameters
    main_network_np, skeleton_np, thickness_np, external_network_np = set_main_network(tof_np, 
                                                                  threshold_tof_4_main_network, 
                                                                  blur_tof_4_main_network, hx)
    sink_support_np = set_sink_support(aseg_np, main_network_np)
    
    # in tof there are small isolated components that we do not want to fit
    # so we apply a slight gaussian blur to remove them
    tof_clean_np = gaussian_filter(tof_np, sigma = hx * 1.5)
    

    # save as nifti
    data = [
        (main_network_np, "main_network"),
        (skeleton_np, "skeleton"),
        (thickness_np, "thickness"),
        (sink_support_np, "sink_support"),
        (tof_clean_np, "tof_clean"),
        (external_network_np, "external_network"),
    ]
    for var, name in data:
        outfilename = os.path.join(out_directory,f"{name}.nii.gz")
        print(f"Saving {outfilename}")
        nibabel.save(nibabel.Nifti1Image(var, affine), outfilename)
    
    #
    # Build mesh
    #
    def build_mesh(voxel_size, brain_mask_np, sink_support_np, tof_support_np, main_network_np, include_sink=True):
        # We need to preprocess the data

        # dilate sink support to avoid small features
        sink_support_mesh_np = binary_dilation(sink_support_np,
                                        iterations=2,mask=brain_mask_np)
        sink_support_mesh_np = sink_support_mesh_np.astype(dtype=np.uint8)

        # we create a neighborhood of the tof where we expect
        # the tof to be reconstructed
        # we set the region where tof must be reconstructed 
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
        tof_neigh_np = set_tof4mesh(tof_support_np, options_dict).astype(np.float32)
    
    
        mask = brain_mask_np.copy()
        if include_sink:
            label_sink = 4
        else:
            label_sink = 0
        mask[sink_support_mesh_np > 0 ] = label_sink
        
        # blur tof
        label_tof = 3
        mask[tof_neigh_np > 0 ] = label_tof
        
        
        label_main = 2
        # remove everything outside domain
        mask[brain_mask_np < 1 ] = 0
        # restore main_network
        mask[main_network_np > 0 ] = label_main 
        mask = mask.astype(np.uint8)


        import pygalmesh        
        PETSc.Sys.Print("volex size:", hx, hy, hz)
        scale = 1
        mesh_pygal = pygalmesh.generate_from_array(
                mask,
                voxel_size, 
                #max_facet_distance=scale*hx,
                max_circumradius_edge_ratio=4.0,
                max_cell_circumradius={
                    "default": scale*16*hx, 
                    label_main: scale*hx,
                    label_tof: scale*hx,
                    label_sink: scale*8*hx
                },
            )
        return mesh_pygal, mask
        
    if build and cell_type == "tetrahedron":
        mesh_pygal, mask_np = build_mesh(voxel_size,
                                         brain_mask_np,
                                         sink_support_np,
                                         tof_clean_np,
                                         main_network_np)
        # recenter mesh
        coordinate = mesh_pygal.points

        # move the coordinate where they exceed the domain [0, lengths[0]], [0, lengths[1]], [0, lengths[2]]
        coordinate[:,0] = np.clip(coordinate[:,0], 0.0, lengths[0])
        coordinate[:,1] = np.clip(coordinate[:,1], 0.0, lengths[1])
        coordinate[:,2] = np.clip(coordinate[:,2], 0.0, lengths[2])

        xmin, ymin, zmin = coordinate.min(axis=0)
        xmax, ymax, zmax = coordinate.max(axis=0)
        print(f"lengths: {lengths}")
        print(f"Mesh bounds after recentering: \n x[{xmin:.2f}, {xmax:.2f}],\n y[{ymin:.2f}, {ymax:.2f}],\n z[{zmin:.2f}, {zmax:.2f}]")
        

        offset = np.array(affine[:3, 3])
        print(f"Offset: x {offset[0]:.10e}, y {offset[1]:.10e}, z {offset[2]:.10e}")
        coordinate[:, 0] += offset[0]
        coordinate[:, 1] += offset[1]
        coordinate[:, 2] += offset[2]
        print("Mesh info:")
        print(f"coordinate_shape: {coordinate.shape}")
        print(f"Mesh has {len(coordinate)} points and {len(mesh_pygal.cells_dict['tetra'])} tetrahedra")

        # save mesh as vtu and msh
        mesh_pygal.write(os.path.join(out_directory,"brain_main.vtu"))
        writer = partial(meshio.gmsh.write, fmt_version="2.2", binary=True)        
        writer(os.path.join(out_directory,"brain_main.msh"), mesh_pygal)

        writer = partial(meshio.exodus.write)
        writer(os.path.join(out_directory,"brain_main.e"), mesh_pygal)

        outfilename = os.path.join(out_directory,
                               f"mask_mesher.nii.gz")
        print(f"Saving mask mesher {outfilename}")
        nibabel.save(nibabel.Nifti1Image(mask_np, affine), outfilename)

    if build and cell_type == "hexahedron":
        mask_np = brain_mask_np.copy()
        mask_np[main_network_np > 0 ] = 1

        # estimate used voxel size
        volume = 100 * np.sum(mask_np > 0) / ( mask_np.shape[0] * mask_np.shape[1] * mask_np.shape[2])
        print(f"Volume fraction of hexa mesh: {volume:.2f}% | new={np.sum(mask_np > 0)} old={mask_np.shape[0] * mask_np.shape[1] * mask_np.shape[2]}")

        print(f" mask shape: {mask_np.shape}, unique: {np.unique(mask_np)}")
        print(f" hx: {hx}, hy: {hy}, hz: {hz}")
        mesh_hexa = export_voxel_to_gmsh(mask_np)
        coordinate = mesh_hexa.points
        print("bounds before scaling:")
        xmin, ymin, zmin = coordinate.min(axis=0)
        xmax, ymax, zmax = coordinate.max(axis=0)
        print(f"{xmin:.10e}<= x <= {xmax:.10e}")
        print(f"{ymin:.10e}<= y <= {ymax:.10e}")
        print(f"{zmin:.10e}<= z <= {zmax:.10e}")



        coordinate *= voxel_size[0]  # assuming isotropic voxel size for simplicity

        print("bounds before recentering:")
        xmin, ymin, zmin = coordinate.min(axis=0)
        xmax, ymax, zmax = coordinate.max(axis=0)
        print(f"{xmin:.10e}<= x <= {xmax:.10e}")
        print(f"{ymin:.10e}<= y <= {ymax:.10e}")
        print(f"{zmin:.10e}<= z <= {zmax:.10e}")


        offset = np.array(affine[:3, 3])
        print("Offset:", offset)
        coordinate[:, 0] += offset[0]
        coordinate[:, 1] += offset[1]
        coordinate[:, 2] += offset[2]


        print("bounds after recentering:")    
        xmin, ymin, zmin = coordinate.min(axis=0)
        xmax, ymax, zmax = coordinate.max(axis=0)
        print(f"{xmin:.10e}<= x <= {xmax:.10e} lx={xmax - xmin:.10e} lengths[0]={lengths[0]:.10e} diff={abs((xmax - xmin) - lengths[0]):.10e}")
        print(f"{ymin:.10e}<= y <= {ymax:.10e} ly={ymax - ymin:.10e} lengths[1]={lengths[1]:.10e} diff={abs((ymax - ymin) - lengths[1]):.10e}")
        print(f"{zmin:.10e}<= z <= {zmax:.10e} lz={zmax - zmin:.10e} lengths[2]={lengths[2]:.10e} diff={abs((zmax - zmin) - lengths[2]):.10e}")

        

        # save mesh as vtu and msh
        mesh_hexa.write(os.path.join(out_directory,"brain_hexa_main.vtu"))
        writer = partial(meshio.gmsh.write, fmt_version="2.2", binary=True)
        writer(os.path.join(out_directory,"brain_hexa_main.msh"), mesh_hexa)

        writer = partial(meshio.exodus.write)
        writer(os.path.join(out_directory,"brain_hexa_main.e"), mesh_hexa)

        
    if build_tof_mesh:
        mesh_tof_pygal, mask_tof_np = build_mesh(voxel_size,
                                         brain_mask_np,
                                         sink_support_np,
                                         tof_clean_np,
                                         main_network_np,
                                            include_sink=False)
        # recenter mesh
        coordinate = mesh_tof_pygal.points

        # move the coordinate where they exceed the domain [0, lengths[0]], [0, lengths[1]], [0, lengths[2]]
        coordinate[:,0] = np.clip(coordinate[:,0], 0.0, lengths[0])
        coordinate[:,1] = np.clip(coordinate[:,1], 0.0, lengths[1])
        coordinate[:,2] = np.clip(coordinate[:,2], 0.0, lengths[2])

        xmin, ymin, zmin = coordinate.min(axis=0)
        xmax, ymax, zmax = coordinate.max(axis=0)
        print(f"lengths: {lengths}")
        print(f"Mesh bounds after recentering: \n x[{xmin:.2f}, {xmax:.2f}],\n y[{ymin:.2f}, {ymax:.2f}],\n z[{zmin:.2f}, {zmax:.2f}]")
        

        offset = affine[:3, 3]
        print("Offset:", offset)
        coordinate[:, 0] += offset[0]
        coordinate[:, 1] += offset[1]
        coordinate[:, 2] += offset[2]
        print("Mesh info:")
        print(f"coordinate_shape: {coordinate.shape}")
        print(f"Mesh has {len(coordinate)} points and {len(mesh_pygal.cells_dict['tetra'])} tetrahedra")

        # save mesh as vtu and msh
        mesh_tof_pygal.write(os.path.join(out_directory,"tof_main.vtu"))
        writer = partial(meshio.gmsh.write, fmt_version="2.2", binary=True)
        writer(os.path.join(out_directory,"tof_main.msh"), mesh_pygal)

        outfilename = os.path.join(out_directory,
                               f"mask_mesher.nii.gz")
        print(f"Saving mask mesher {outfilename}")
        nibabel.save(nibabel.Nifti1Image(mask_tof_np, affine), outfilename)

    
    if firedrake_conversion:
        start = time.time()
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
        start  = time.time()
        PETSc.Sys.Print("intepolate marker for relabeled mesh", end="")
        marker_space = FunctionSpace(mesh, "HDiv Trace", 0)
        main_network_indicator = Function(marker_space, name="main_network_indicator")
        x,y,z = mesh.coordinates
        main_network_indicator.interpolate(main_network_mesh * conditional(abs(z-zmin) < hx,1,0))
        PETSc.Sys.Print(f" - completed in {time.time()-start:.2e} s")
        
        elem = main_network_indicator.topological.function_space().ufl_element()
        PETSc.Sys.Print(elem.family(),elem.degree())
        start  = time.time()
        PETSc.Sys.Print("Relabeled mesh", end="")
        relabeled_mesh = MyRelabeledMesh(mesh, [main_network_indicator], 
                                        [99],
                                        boundary_only=True,
                                        name="relabeled_mesh")
        PETSc.Sys.Print(f" - completed in {time.time()-start:.2e} s")

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

        PETSc.Sys.Print("Interpolation completed")
        
        #VTKFile("labeled_mesh.pvd").write(relabeled_mesh)
        # shift coordinate of the relabeled mesh
        offset = affine[:3, 3]
        #relabeled_mesh.coordinates.dat.data[:, 0] += offset[0]
        #relabeled_mesh.coordinates.dat.data[:, 1] += offset[1]
        #relabeled_mesh.coordinates.dat.data[:, 2] += offset[2]
        #PETSc.Sys.Print("Offset completed")


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

      
            
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mri', type=str)
    parser.add_argument('--out', type=str)
    parser.add_argument('--threshold', type=float, default=250,
                        help="Threshold for Tof. Default is 250.")
    parser.add_argument('--blur_main', type=float, default=0.0, 
                        help="Blur for connected components. If 0, no blur is applied.")
    parser.add_argument('--blur_mesh', type=float, default=0.0, 
                        help="Blur for connected components. If 0, no blur is applied.")
    parser.add_argument('--type', type=str, default="tetrahedron", help="tetrahedron or hexahedron")
    parser.add_argument('--tof_only', action='store_true', help="Build only the tof mesh.")
    parser.add_argument('--read', action='store_true')
    parser.add_argument('--meshonly', action='store_true')

    args = parser.parse_args()

    setup(args.mri, args.out, args.threshold, args.blur_main, args.blur_mesh, not args.read, not args.meshonly, args.tof_only, cell_type=args.type)
    
    
