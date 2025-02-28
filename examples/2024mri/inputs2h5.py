import nibabel 
from firedrake import CheckpointFile, COMM_WORLD, PETSc
from niot import image2dat as i2d
import argparse
import numpy as np
import gc
from connected_components_tof import connected_components, main_network_equal_one, find_external_network

parser = argparse.ArgumentParser()
parser.add_argument('--mri', type=str)
parser.add_argument('--threshold', type=float, default=250)
args = parser.parse_args()

n_proc = COMM_WORLD.size

# load tof data and get basic info
dir_nii = args.mri
tof_data = nibabel.load(f"{dir_nii}/TOF.nii.gz")
dimensions = tof_data.header.get_data_shape()[:3]

PETSc.Sys.Print(f"Data shape: {dimensions=}")
hx, hy, hz = tof_data.header['pixdim'][1:4]
lengths = np.array([float(dimensions[0]*hx), 
                    float(dimensions[1]*hy), 
                    float(dimensions[2]*hz)])

# define the mesh
PETSc.Sys.Print(f"Mesh start")
cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths)
PETSc.Sys.Print(f"Mesh done")


# load tof data
# tof_np = tof_data.get_fdata()
tof_np = np.load(f"{dir_nii}/TOF.npy",mmap_mode='r')
tof = i2d.numpy2firedrake(cartesian_mesh, tof_np, name='tof')

PETSc.Sys.Print(f"TOF done")
labels_np, n_labels = connected_components(tof_np,  threshold=args.threshold)
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
                    f"{dir_nii}external_network_t{args.threshold:.2e}.nii.gz")
external_network = i2d.numpy2firedrake(cartesian_mesh, external_network_np, name="external_network")
PETSc.Sys.Print(f"External network done")
external_network_np = None
gc.collect()


nibabel.save(nibabel.Nifti1Image(main_network_np, tof_data.affine), 
                 f"{dir_nii}main_network_t{args.threshold:.2e}.nii.gz")
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
                 f"{dir_nii}inlets_t{args.threshold:.2e}.nii.gz")
inlets_np = None
gc.collect()


# get brain mask
brain_mask_data = nibabel.load(f"{dir_nii}/brain_mask.nii.gz")
brain_mask_np = np.load(f"{dir_nii}brain_mask.npy",mmap_mode='r')
brain_mask = i2d.numpy2firedrake(cartesian_mesh, brain_mask_np, name="brain_mask")
brain_mask_np = None
PETSc.Sys.Print(f"Brain mask done")
gc.collect()

# load main network
t1_nii = nibabel.load(f"{dir_nii}/T1.nii.gz")
#t1_np = t1_nii.get_fdata()
t1_np = np.load(f"{dir_nii}/T1.npy",mmap_mode='r')
t1 = i2d.numpy2firedrake(cartesian_mesh, t1_np, name="t1")
t1_np = None
PETSc.Sys.Print(f"T1 done")
gc.collect()

cartesian_mesh.name = "mesh"
with CheckpointFile(f"{args.mri}/inputs_nproc{n_proc}.h5", 'w') as afile:
    afile.save_mesh(cartesian_mesh)
    PETSc.Sys.Print(f"mesh done")
    afile.save_function(tof)
    PETSc.Sys.Print(f"tof done")
    afile.save_function(t1)
    PETSc.SYS.Print(f"t1 done")
    afile.save_function(brain_mask)
    PETSc.Sys.Print(f"brain mask done")
    afile.save_function(main_network)
    PETSc.Sys.Print(f"main network done")
    afile.save_function(external_network)
    PETSc.Sys.Print(f"external network done")
    afile.save_function(inlets)
    PETSc.Sys.Print(f"inlets done")