import nibabel 
from firedrake import CheckpointFile, COMM_WORLD, PETSc,dx, conditional, assemble, ExtrudedMesh
from niot import image2dat as i2d
import argparse
import numpy as np
import gc
from connected_components_tof import connected_components, main_network_equal_one, find_external_network
import os
from scipy.ndimage import gaussian_filter

def save_as_npy(file_nii, file_npy, comm=COMM_WORLD):
    PETSc.Sys.Print(f" {file_nii} to {file_npy}", os.path.exists(file_npy))
    if not os.path.exists(file_npy):
        if comm.rank == 0:
            PETSc.Sys.Print(f"Convertion {file_nii} to {file_npy}",end="")
            data = nibabel.load(file_nii)
            data_np = data.get_fdata()
            np.save(file_npy, data_np)
            PETSc.Sys.Print(f"- Done")
    comm.barrier()

def clean_npy_file(file_npy):
    try:
        os.remove(file_npy)
    except:
        pass


def nii2firedrake(path, cartesian_mesh, name, comm=COMM_WORLD):
    # load tof data
    # tof_np = tof_data.get_fdata()
    file_nii = path
    file_npy = "temp.npy"
    save_as_npy(file_nii, file_npy, comm=comm)
    data_np = np.load(file_npy,mmap_mode='r')
    data = i2d.numpy2firedrake(cartesian_mesh, data_np, name=name)
    data_np = None
    clean_npy_file(file_npy)
    return data

def build_masked_mesh(support_mask, lengths, threshold=1e-10, sigma_blur=1.2):
    nx, ny, nz = support_mask.shape
    print("Image shape:", support_mask.shape)

    mask = np.copy(support_mask)
    mask = gaussian_filter(mask, sigma=sigma_blur)
    mask[mask < threshold] = 0.0
    mask[mask >= threshold] = 1.0
    # convert to binary image
    mask = mask.astype(np.int8)

    # select largst connected component
    labels_np, nlabels = connected_components(mask, 0.1)
    labels_np = main_network_equal_one(labels_np, nlabels, mask)
    # free memory
    mask = None
    gc.collect
    


    support = np.zeros_like(labels_np, dtype=np.uint8)
    support[labels_np == 1] = 1
    labels_np = None
    gc.collect()



    # compute proportion of non-zero voxels
    PETSc.Sys.Print("Proportion of non-zero voxels:", np.count_nonzero(support) / support.size)
    mesh3d = i2d.mesh_from_3d_mask(support, lengths, variable_layer=False, invert_rows_columns=False)
    

    # # create 2D mesh from the projection
    # PETSc.Sys.Print("Creating 2D mesh from the projection")
    # topol, coordinates, _ , _, _ = i2d.topol_coords_edges_from_mask(mask_xy, 
    #                                                             Lx=lengths[0], 
    #                                                             Ly=lengths[1], 
    #                                                             invert_rows_columns=True)
    # PETSc.Sys.Print("Creating 2D mesh ")
    # mesh2d = i2d.mesh_from_topology(topol, coordinates, reorder=False)
    
    # PETSc.Sys.Print("Creating 3D mesh ")
    # mesh3d = ExtrudedMesh(mesh2d, nz, lengths[2]/nz)
    # mesh3d.nx = nx
    # mesh3d.ny = ny
    # mesh3d.nz = nz
    # mesh3d.xmin = 0.0
    # mesh3d.ymin = 0.0
    # mesh3d.zmin = 0.0
    # mesh3d.xmax = lengths[0]
    # mesh3d.ymax = lengths[1]
    # mesh3d.zmax = lengths[2]
    # PETSc.Sys.Print("fire from numpy ")

    brain_mask = i2d.numpy2firedrake(mesh3d, support, "brain_mask")

    support = None
    gc.collect()


    return mesh3d, brain_mask   



def setup_h5(mri_directory, threshold, blur = 0.0, masked_mesh=True, comm=COMM_WORLD):
    n_proc = comm.size

    # load tof data and get basic info
    dir_nii = mri_directory
    tof_data = nibabel.load(f"{dir_nii}/TOF.nii.gz")
    dimensions = tof_data.header.get_data_shape()[:3]

    PETSc.Sys.Print(f"Data shape: {dimensions=}")
    hx, hy, hz = tof_data.header['pixdim'][1:4]
    lengths = np.array([float(dimensions[0]*hx), 
                        float(dimensions[1]*hy), 
                        float(dimensions[2]*hz)])
    

    # get brain mask
    brain_nii_file = f"{dir_nii}/brain_mask.nii.gz"
    brain_npy_file = f"{dir_nii}/brain_mask.npy"
    save_as_npy(brain_nii_file, brain_npy_file, comm=comm)
    brain_mask_np = np.load(brain_npy_file,mmap_mode='r')
    
    # define the mesh based on the brain mask
    PETSc.Sys.Print(f"Mesh")
    if masked_mesh:
        PETSc.Sys.Print(f"Building masked mesh ")
        cartesian_mesh, brain_mask = build_masked_mesh(brain_mask_np, lengths, threshold=1e-10, sigma_blur=1.2)
        
    else:
        PETSc.Sys.Print(f"Building full mesh ")
        cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths,comm=comm)        
        brain_mask = i2d.numpy2firedrake(cartesian_mesh, brain_mask_np, name="brain_mask")
    PETSc.Sys.Print(f"done")
    
    
    brain_volume = assemble(conditional(brain_mask>1e-10,1,0)*dx) / np.prod(lengths)
    PETSc.Sys.Print(f"Brain volume: {brain_volume*100:.2f}%")
    brain_mask_np = None
    PETSc.Sys.Print(f"Brain mask done")
    clean_npy_file(brain_npy_file)
    gc.collect()



    


    # load tof data
    # tof_np = tof_data.get_fdata()
    file_nii = f"{dir_nii}/TOF.nii.gz"
    file_npy = f"{dir_nii}/TOF.npy"   
    save_as_npy(file_nii, file_npy, comm=comm)
    PETSc.Sys.Print(f"TOF ",end="")
    tof_np = np.load(file_npy,mmap_mode='r')
    tof = i2d.numpy2firedrake(cartesian_mesh, tof_np, name='tof')
    tof_np = None
    clean_npy_file(file_npy)
    PETSc.Sys.Print(f" - done")
    
    
    # inlets
    # tof_np = tof_data.get_fdata()
    
    file_nii = f"{dir_nii}/inlets.nii.gz"
    file_npy = f"{dir_nii}/inlets.npy"   
    save_as_npy(file_nii, file_npy, comm=comm)
    PETSc.Sys.Print(f"Inlets",end="")
    inlets_2d_np = np.load(file_npy,mmap_mode='r')
    inlets_3d_np = np.zeros(dimensions)
    inlets_3d_np[:,:,0] = inlets_2d_np[:,:,0]
    inlets = i2d.numpy2firedrake(cartesian_mesh, inlets_3d_np, name="inlets")
    inlets_2d_np = None
    inlets_3d_np = None
    PETSc.Sys.Print(f" - done")
    clean_npy_file(file_npy)
    gc.collect()


    # get brain mask
    brain_nii_file = f"{dir_nii}/brain_mask.nii.gz"
    brain_npy_file = f"{dir_nii}/brain_mask.npy"
    save_as_npy(brain_nii_file, brain_npy_file, comm=comm)
    brain_mask_np = np.load(brain_npy_file,mmap_mode='r')
    brain_mask = i2d.numpy2firedrake(cartesian_mesh, brain_mask_np, name="brain_mask")
    brain_volume = assemble(conditional(brain_mask>1e-10,1,0)*dx) / np.prod(lengths)
    PETSc.Sys.Print(f"Brain volume: {brain_volume*100:.2f}%")
    brain_mask_np = None
    PETSc.Sys.Print(f"Brain mask done")
    clean_npy_file(brain_npy_file)
    gc.collect()


    # load tof data
    # tof_np = tof_data.get_fdata()
    file_nii = f"{dir_nii}/aseg.nii.gz"
    file_npy = f"{dir_nii}/aseg.npy"   
    save_as_npy(file_nii, file_npy, comm=comm)
    PETSc.Sys.Print(f"aseg ",end="")
    aseg_np = np.load(file_npy,mmap_mode='r')
    aseg = i2d.numpy2firedrake(cartesian_mesh, aseg_np, name='aseg')
    aseg_np = None
    clean_npy_file(file_npy)
    PETSc.Sys.Print(f" - done")
    

    # load main network
    file_nii = f"{dir_nii}/T1.nii.gz"
    file_npy = f"{dir_nii}/T1.npy"   
    save_as_npy(file_nii, file_npy, comm=comm)
    t1_np = np.load(file_npy,mmap_mode='r')
    t1 = i2d.numpy2firedrake(cartesian_mesh, t1_np, name="t1")
    t1_np = None
    PETSc.Sys.Print(f"T1 done")
    clean_npy_file(file_npy)
    gc.collect()
    
    # 
    # threshold dependent data
    # 
    
    
    # main and external network
    if blur > 0:
        PETSc.Sys.Print(f"Connected components with threshold {threshold:.2e} and blur {blur:.2e}")
        label = f"t{threshold:.2e}_blur{blur:.2e}"  
    else: 
        label = f"t{threshold:.2e}"
    file_nii = f"{dir_nii}/main_network_{label}.nii.gz"
    file_npy = f"{dir_nii}/main_network_{label}.npy"   
    save_as_npy(file_nii, file_npy, comm=comm)
    main_network_np = np.load(file_npy,mmap_mode='r')
    main_network = i2d.numpy2firedrake(cartesian_mesh, main_network_np, name="main_network")
    PETSc.Sys.Print(f"Main network done")
    main_network_np = None
    gc.collect()

    # read skeleton mask of main network
    file_nii = f"{dir_nii}/skeleton_{label}.nii.gz"
    file_npy = f"{dir_nii}/skeleton_{label}.npy"   
    save_as_npy(file_nii, file_npy, comm=comm)
    skeleton_np = np.load(file_npy,mmap_mode='r')
    skeleton = i2d.numpy2firedrake(cartesian_mesh, skeleton_np, name="skeleton")
    PETSc.Sys.Print(f"Skeleton done")
    skeleton_np = None
    gc.collect()

    # read local thickness of main network
    file_nii = f"{dir_nii}/thickness_{label}.nii.gz"
    file_npy = f"{dir_nii}/thickness_{label}.npy"   
    save_as_npy(file_nii, file_npy, comm=comm)
    thickness_np = np.load(file_npy,mmap_mode='r')
    thickness = i2d.numpy2firedrake(cartesian_mesh, thickness_np, name="thickness")
    PETSc.Sys.Print(f"Local thickness done")
    thickness_np = None
    gc.collect()

    

    # external network
    file_nii = f"{dir_nii}/external_network_{label}.nii.gz"
    file_npy = f"{dir_nii}/external_network_{label}.npy"   
    save_as_npy(file_nii, file_npy, comm=comm)
    external_network_np = np.load(file_npy,mmap_mode='r')
    external_network = i2d.numpy2firedrake(cartesian_mesh, external_network_np, name="external_network")
    PETSc.Sys.Print(f"External network done")
    external_network_np = None
    gc.collect()

    return cartesian_mesh, tof, aseg, t1, brain_mask, main_network, external_network, inlets, skeleton, thickness


def write_h5(mri_directory, threshold, blur, comm, n_proc, data):
    # unpack data
    cartesian_mesh, tof, aseg, t1, brain_mask, main_network, external_network, inlets, skeleton, thickness = data

    #
    # save to h5
    #
    cartesian_mesh.name = "mesh"
    if blur > 0:
        PETSc.Sys.Print(f"Connected components with threshold {threshold:.2e} and blur {blur:.2e}")
        label = f"t{threshold:.2e}_blur{blur:.2e}"  
    else: 
        label = f"t{threshold:.2e}"
    h5_filename = f"{mri_directory}/inputs_{label}_nproc{n_proc}.h5"
    
    # removing file if it exists
    if os.path.exists(h5_filename):
        try:
            os.remove(h5_filename)
        except:
            pass
    PETSc.Sys.Print(f"Saving to {h5_filename}", end="")
    with CheckpointFile(h5_filename, 'w', comm=comm) as afile:
        afile.save_mesh(cartesian_mesh)
        PETSc.Sys.Print(f" mesh ", end="")
        afile.save_function(tof)
        PETSc.Sys.Print(f" tof ", end="")
        afile.save_function(aseg)
        PETSc.Sys.Print(f" aseg ", end="")
        afile.save_function(t1)
        PETSc.Sys.Print(f" t1 ", end="")
        afile.save_function(brain_mask)
        PETSc.Sys.Print(f" brain_mask ", end="")
        afile.save_function(main_network)
        PETSc.Sys.Print(f" main_network ", end="")
        afile.save_function(external_network)
        PETSc.Sys.Print(f" external_network ", end="")
        afile.save_function(inlets)
        PETSc.Sys.Print(f" inlets ", end="")
        afile.save_function(skeleton)
        PETSc.Sys.Print(f" skeleton ", end="")
        afile.save_function(thickness)
        PETSc.Sys.Print(f" thickness ", end="")
    PETSc.Sys.Print(f" - done")
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mri', type=str)
    parser.add_argument('--threshold', type=float, default=250,
                        help="Threshold for Tof. Default is 250.")
    parser.add_argument('--blur', type=float, default=0.0, 
                        help="Blur for connected components. If 0, no blur is applied.")
    args = parser.parse_args()

    data = setup_h5(args.mri, args.threshold, args.blur, COMM_WORLD)
    n_proc = COMM_WORLD.size
    write_h5(args.mri, args.threshold, args.blur, COMM_WORLD, n_proc, data)
    
    
