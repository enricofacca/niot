import nibabel
import os
import numpy as np
import argparse
from scipy.ndimage import gaussian_filter
from skimage.morphology import skeletonize
import localthickness as lt
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt
from skimage.morphology import medial_axis
import cc3d


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


def connected_components(np_data, threshold, connectivity=26):
    network = np.zeros_like(np_data)
    network [np_data >= threshold] = 1
    labels, n_labels = cc3d.connected_components(network, 
                                                 connectivity=connectivity, 
                                                 binary_image=True, 
                                                 return_N=True)
    return labels, n_labels



def set_sink_support(aseg_np, main_network_np):
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


def find_external_network(labels_np):
    """
    Find the connected components different from the 
    main network (labels=1) that are connected to the bottom of the domain.
    """
    # find inlets of external network
    indices_bottom = labels_np[:,:,0]
    list_indices_bottom = np.unique(indices_bottom)
    
    # remove 0 (background) and 1(main network) from the list
    list_external = list_indices_bottom[2:]
    external_np = np.zeros_like(labels_np, dtype=np.uint8)
    for index in list_external:
        location = np.where(labels_np == index)
        external_np[location] = 1
    return external_np



def setup_nifti(mri_directory, 
                out_directory, 
                threshold_tof_4_main_network=400, 
                blur_tof_4_main_network=0.0,
                threshold_tof_4_fitting=175,
                blur_tof_4_fitting=1.5,
                expansion_mm=10
                ):
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)

    voxel_size, affine, tof_np, brain_mask_np, t1_np, aseg_np = load_data(mri_directory)
    hx, hy, hz = voxel_size

    dimensions = tof_np.shape
    lengths = np.array([hx,hy,hz]) * np.array(dimensions)
    
    print(f"{hx=:.10e} {hy=:.10e} {hz=:.10e}")
    print(f"voxel_size: {voxel_size}, lengths: {lengths}")
    h_new = lengths/np.array(dimensions)
    print(f"New voxel size: {h_new[0]=:.10e}, {h_new[1]=:.10e}, {h_new[2]=:.10e}")
        
    # process parameters
     # blur tof, separate main network, and keep largest connected component
    if blur_tof_4_main_network> 0:
        print(f" - applying gaussian blur {blur_tof_4_main_network:.2e}")
        tof_main_network_np = gaussian_filter(tof_np, sigma=blur_tof_4_main_network * hx)
    else:
        tof_main_network_np = tof_np.copy()
    labels_np, nlabels = connected_components(tof_main_network_np, threshold_tof_4_main_network)
    

    # We count the occurences of each label in the labels_np array
    #
    counts = np.bincount(labels_np.flatten())
    print("Label counts (label: count):")
    for label, count in enumerate(counts):
        print(f"  {label}: {count}")

    # sort labels by size
    sorted_labels = np.argsort(counts).astype(np.int32)
    # swap labels so that the largest connected component is labeled as 0, the second largest as 1, etc.
    label_mapping = np.zeros_like(sorted_labels)
    for new_label, old_label in enumerate(reversed(sorted_labels)):
        label_mapping[old_label] = new_label
    # apply the mapping to labels_np
    labels_np = label_mapping[labels_np]
    

    
    #labels_np = main_network_equal_one(labels_np, nlabels, tof_np)
    
    # define main network 
    main_network_np = np.zeros_like(labels_np, dtype=np.uint8)
    main_network_np[labels_np == 1] = 1
    
    # get the skeleton of main network
    skeleton_np = skeletonize(main_network_np).astype(np.uint8)

    # compute local thickness of the main network
    print("Local thickness computation",end="")
    thickness_np = lt.local_thickness(main_network_np)
    # scale by thickness 
    thickness_np *= hx
    print(" -done")

    # find external network
    print(" Finding external network",end="")
    external_network_np = find_external_network(labels_np)
    print(" -done")

    # set sink support
    print(" Setting sink support",end="")
    sink_support_np = set_sink_support(aseg_np, main_network_np)
    print(" -done")
    # set inlet marker. Bottom slice of main network
    inlets_np = main_network_np.copy()
    inlets_np[:,:,1:] = 0
    

    # in tof there are small isolated components that we do not want to fit
    # so we apply a slight gaussian blur to remove them
    print(" Preparing TOF for fitting",end="")
    if blur_tof_4_fitting>0.0:
        tof_clean_np = gaussian_filter(tof_np, sigma = hx * blur_tof_4_fitting)
    else:
        tof_clean_np = tof_np
    tof_4_fit = tof_clean_np.copy()
    tof_4_fit[brain_mask_np == 0] = 0.0
    tof_4_fit[main_network_np > 0] = tof_np[main_network_np > 0]

    # remove external network from fitting tof
    tof_4_fit[external_network_np > 0] = 0.0
    tof_4_fit[tof_4_fit < threshold_tof_4_fitting] = 0.0
    print(" -done")

    # compute distance map from the boundary
    support_corrupted = np.zeros_like(labels_np, dtype=np.uint8)
    support_corrupted[tof_4_fit > 0] = 1
    print(" Filling holes in support for distance map",end="")
    support_corrupted = binary_fill_holes(support_corrupted).astype(np.uint8)
    print(" -done")
    print(" Computing distance map",end="")
    distance_map = distance_transform_edt(support_corrupted)
    print(" -done")
    # compute mu based on distance map and thickness
    distance_map *= hx

    skeleton_corrupted = skeletonize(support_corrupted).astype(np.uint8)


    thickness_support =  lt.local_thickness(support_corrupted)
    # scale by thickness 
    thickness_support *= hx

    #
    mu = (thickness_support/2)**4 * main_network_np

    
    print(" Dilating TOF and main network",end="")
    iterations = int(np.round(expansion_mm / hx))
    tof_neighborhood_np = binary_dilation(tof_clean_np,
                                    iterations=iterations,
                                    mask=brain_mask_np)
    print(" -done")

    print(" Dilating main network",end="")
    main_network_neighborhood_np = binary_dilation(main_network_np,
                                    iterations=iterations,
                                    mask=brain_mask_np).astype(np.uint8)
    print(" -done")
    
    main_network_filled_np = binary_fill_holes(main_network_np).astype(np.uint8)
    skeleton_filled_np = skeletonize(main_network_filled_np).astype(np.uint8)

    domain_np = binary_dilation(main_network_filled_np,
                                iterations=1).astype(np.uint8)
    
    # create mask for the extension only
    boundary_np = domain_np - main_network_filled_np
    boundary_np[boundary_np < 0] = 0


    # save as nifti
    data = [
        (main_network_np, "main_network"),
        (main_network_neighborhood_np, "main_network_neighborhood"),
        (main_network_filled_np, "main_network_filled"),
        (skeleton_np, "skeleton"),
        (skeleton_filled_np, "skeleton_filled"),
        (thickness_np, "thickness"),
        (sink_support_np, "sink_support"),
        (tof_clean_np, "tof_clean"),
        (external_network_np, "external_network"),
        (inlets_np, "inlets"),
        (labels_np, "connected_components"),
        (tof_4_fit, "tof_4_fitting"),
        (mu, "mu"),
        (distance_map, "distance_map"),
        (thickness_support, "thickness_support"),
        (skeleton_corrupted, "skeleton_corrupted"),
        (domain_np,"domain_np"),
        (boundary_np,"boundary_np")
    ]
    for var, name in data:
        outfilename = os.path.join(out_directory,f"{name}.nii.gz")
        print(f"Saving {outfilename}")
        nibabel.save(nibabel.Nifti1Image(var, affine), outfilename)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess MRI NIfTI files for mesh generation.")
    parser.add_argument("-i","--input", type=str, help="Directory containing the input NIfTI files.")
    parser.add_argument("-o","--out", type=str, help="Directory to save the processed files.")
    parser.add_argument("-t","--threshold", type=float, help="Threshold for TOF main network extraction.", default=350)
    parser.add_argument("-b","--blur", type=float, help="Blur for TOF main network extraction.", default=1.5)
    parser.add_argument("--t_fitting", type=float, help="Threshold for TOF fitting extraction.", default=175)
    parser.add_argument("--b_fitting", type=float, help="Blur for TOF fitting extraction.", default=1.5)
    parser.add_argument("-e","--expansion_mm", type=float, help="Number of iterations for expanding TOF and main network.", default=2)
    args = parser.parse_args()

    mri_directory = args.input
    out_directory = args.out

    setup_nifti(mri_directory, 
                out_directory, 
                threshold_tof_4_main_network=args.threshold, 
                blur_tof_4_main_network=args.blur,
                threshold_tof_4_fitting=args.t_fitting,
                blur_tof_4_fitting=args.b_fitting,
                expansion_mm=args.expansion_mm
                )
                

