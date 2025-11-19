import nibabel 
import argparse
import numpy as np
from connected_components_tof import connected_components, main_network_equal_one, find_external_network
import os
from scipy.ndimage import gaussian_filter
import pygalmesh
from firedrake import *
from niot import image2dat as i2d

def setup(mri_directory, threshold, blur = 0.0, blur_tof = 0.0 ):
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

    

    mask = brain_mask_np.copy()
    label_t1 = 4
    mask[t1_np > 500 ] = label_t1
    
    
    label_tof = 3
    t = 0.175 * tof_smooth_np.max()
    mask[tof_smooth_np > t ] = label_tof
    
    
    label_main = 2
    mask[main_network > 0 ] = label_main 

    mask[brain_mask_np < 1 ] = 0
    mask = mask.astype(np.uint8)


    voxel_size = (hx, hy, hz)

    # mesh = pygalmesh.generate_from_array(
    #     mask,
    #     voxel_size, 
    #     max_facet_distance=0.2,
    #     max_cell_circumradius={
    #         "default": 2.0, 
    #         label_main: 0.5,
    #         label_tof: 0.5,
    #         label_t1: 2.0
    #         },
    # )
    # mesh.write("brain.vtu")
    
    mesh = Mesh("brain.msh")


    cartesian_mesh =  i2d.cartesian_grid_3d(dimensions,lengths)
    tof_cartesian = i2d.numpy2firedrake(cartesian_mesh, tof_np, name='tof')
    t1_cartesian = i2d.numpy2firedrake(cartesian_mesh, t1_np, name='t1')

    DG0 = FunctionSpace(mesh, "DG", 0)
    tof_mesh = Function(DG0, name="tof_mesh")
    t1_mesh = Function(DG0, name="t1_mesh")
    tof_mesh.interpolate(tof_cartesian)
    t1_mesh.interpolate(t1_cartesian)

    # save as pvd
    VTKFile("tof_mesh.pvd").write(tof_mesh)


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
    args = parser.parse_args()

    setup(args.mri, args.threshold, args.blur_main, args.blur_tof)
    
    
