import nibabel 
from firedrake import CheckpointFile, COMM_WORLD, PETSc,dx, conditional, assemble
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
    PETSc.Sys.Print(f"Mesh", end="")
    cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths,comm=comm)
    #cartesian_mesh = i2d.build_mesh_from_numpy(
    #    dimensions, 
    #    mesh_type='cartesian',
    #    lengths=lengths,
    #    comm=comm,
    #    extrude=False)
    PETSc.Sys.Print(f"done")


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
    file_nii = f"{dir_nii}/main_network_t{threshold:.2e}.nii.gz"
    file_npy = f"{dir_nii}/main_network_t{threshold:.2e}.npy"   
    save_as_npy(file_nii, file_npy, comm=comm)
    main_network_np = np.load(file_npy,mmap_mode='r')
    main_network = i2d.numpy2firedrake(cartesian_mesh, main_network_np, name="main_network")
    PETSc.Sys.Print(f"Main network done")
    main_network_np = None
    gc.collect()

    # external network
    file_nii = f"{dir_nii}/external_network_t{threshold:.2e}.nii.gz"
    file_npy = f"{dir_nii}/external_network_t{threshold:.2e}.npy"   
    save_as_npy(file_nii, file_npy, comm=comm)
    external_network_np = np.load(file_npy,mmap_mode='r')
    external_network = i2d.numpy2firedrake(cartesian_mesh, external_network_np, name="external_network")
    PETSc.Sys.Print(f"External network done")
    external_network_np = None
    gc.collect()

    #
    # save to h5
    #
    cartesian_mesh.name = "mesh"
    h5_filename = f"{mri_directory}/inputs_t{threshold:.2e}_nproc{n_proc}.h5"
    
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
    PETSc.Sys.Print(f" - done")
    
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mri', type=str)
    parser.add_argument('--threshold', type=float, default=250)
    args = parser.parse_args()

    setup_h5(args.mri, args.threshold)
