import nibabel 
from firedrake import CheckpointFile, COMM_WORLD, PETSc
from niot import image2dat as i2d
import argparse
import numpy as np
import gc
from connected_components_tof import connected_components, main_network_equal_one, find_external_network
import os

def save_as_npy(file_nii, file_npy, comm=COMM_WORLD):
    PETSc.Sys.Print(f" {file_nii} to {file_npy}", os.path.exists(file_npy))
    if not os.path.exists(file_npy):
        if comm.rank == 0:
            PETSc.Sys.Print(f"Convert {file_nii} to {file_npy}")
            data = nibabel.load(file_nii)
            data_np = data.get_fdata()
            np.save(file_npy, data_np)
    comm.barrier()

def clean_npy_file(file_npy):
    try:
        os.remove(file_npy)
    except:
        pass


def setup_h5(mri_directory, threshold, comm=COMM_WORLD):
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

    # define the mesh
    PETSc.Sys.Print(f"Mesh start")
    cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths,comm=comm)
    PETSc.Sys.Print(f"Mesh done")


    # load tof data
    # tof_np = tof_data.get_fdata()
    file_nii = f"{dir_nii}/TOF.nii.gz"
    file_npy = f"{dir_nii}/TOF.npy"   
    save_as_npy(file_nii, file_npy, comm=comm)
    tof_np = np.load(file_npy,mmap_mode='r')
    tof = i2d.numpy2firedrake(cartesian_mesh, tof_np, name='tof')
    clean_npy_file(file_npy)


    PETSc.Sys.Print(f"TOF done")
    labels_np, n_labels = connected_components(tof_np,  threshold=threshold)
    labels_np = main_network_equal_one(labels_np, n_labels, tof_np)
    tof_np = None
    gc.collect()

    # main and external network
    main_network_np = np.zeros_like(labels_np, dtype=np.uint8)
    main_network_np[labels_np == 1] = 1
    external_network_np = find_external_network(labels_np)
    labels_np = None
    gc.collect()

    nibabel.save(nibabel.Nifti1Image(external_network_np, tof_data.affine),
                        f"{dir_nii}external_network_t{threshold:.2e}.nii.gz")
    external_network = i2d.numpy2firedrake(cartesian_mesh, external_network_np, name="external_network")
    PETSc.Sys.Print(f"External network done")
    external_network_np = None
    gc.collect()


    nibabel.save(nibabel.Nifti1Image(main_network_np, tof_data.affine), 
                    f"{dir_nii}main_network_t{threshold:.2e}.nii.gz")
    main_network = i2d.numpy2firedrake(cartesian_mesh, main_network_np, name="main_network")
    PETSc.Sys.Print(f"Main network done")
    # copy main network before cleaning it
    inlets_np = np.copy(main_network_np)
    main_network_np = None
    gc.collect()

    # inlets
    inlets_np[:,:,1:] = 0
    inlets = i2d.numpy2firedrake(cartesian_mesh, inlets_np, name="inlets")
    nibabel.save(nibabel.Nifti1Image(inlets_np, tof_data.affine), 
                    f"{dir_nii}inlets_t{threshold:.2e}.nii.gz")
    inlets_np = None
    gc.collect()


    # get brain mask
    brain_nii_file = f"{dir_nii}/brain_mask.nii.gz"
    brain_npy_file = f"{dir_nii}/brain_mask.npy"
    save_as_npy(brain_nii_file, brain_npy_file, comm=comm)
    brain_mask_np = np.load(brain_npy_file,mmap_mode='r')
    brain_mask = i2d.numpy2firedrake(cartesian_mesh, brain_mask_np, name="brain_mask")
    brain_mask_np = None
    PETSc.Sys.Print(f"Brain mask done")
    clean_npy_file(brain_npy_file)
    gc.collect()

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

    cartesian_mesh.name = "mesh"
    h5_filename = f"{mri_directory}/inputs_t{threshold:.2e}_nproc{n_proc}.h5"
    
    # removing file if it exists
    if os.path.exists(h5_filename):
        try:
            os.remove(h5_filename)
        except:
            pass
    PETSc.Sys.Print(f"Saving to {h5_filename}")
    with CheckpointFile(h5_filename, 'w', comm=comm) as afile:
        afile.save_mesh(cartesian_mesh)
        PETSc.Sys.Print(f"mesh done")
        afile.save_function(tof)
        PETSc.Sys.Print(f"tof done")
        afile.save_function(t1)
        PETSc.Sys.Print(f"t1 done")
        afile.save_function(brain_mask)
        PETSc.Sys.Print(f"brain mask done")
        afile.save_function(main_network)
        PETSc.Sys.Print(f"main network done")
        afile.save_function(external_network)
        PETSc.Sys.Print(f"external network done")
        afile.save_function(inlets)
        PETSc.Sys.Print(f"inlets done")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mri', type=str)
    parser.add_argument('--threshold', type=float, default=250)
    args = parser.parse_args()

    setup_h5(args.mri, args.threshold)
