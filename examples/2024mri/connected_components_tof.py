import nibabel
import numpy as np
import argparse
import cc3d
from skimage.morphology import skeletonize
import localthickness as lt
import copy as cp



def connected_components(np_data, threshold, connectivity=26):
    network = np.zeros_like(np_data)
    network [np_data >= threshold] = 1
    labels, n_labels = cc3d.connected_components(network, 
                                                 connectivity=connectivity, 
                                                 binary_image=True, 
                                                 return_N=True)
    return labels, n_labels

def main_network_equal_one(labels_np, n_labels, tof_np):
    """
    Mark the main network as label 1.
    """
    
    # find the index of the main network
    index_max_tof = np.unravel_index(tof_np.argmax(), tof_np.shape)
    label_largest_tof = labels_np[index_max_tof]
    
    #
    # We count the occurences of each label in the labels_np array
    #
    counts = np.bincount(labels_np.flatten())
    # 0 is background, 
    label_largest_component = np.argmax(counts[1:])+1
    
    candidates = [label_largest_tof, label_largest_component]
    if not all(candidates):
        for c in candidates:
            print("Candidate", c)
        raise ValueError(f"Unique identification of the main network failed")

    label_main_network = candidates[0]


    # set label of the main network to 1 (swapping with label 1)
    temp = np.max(labels_np) + 1
    labels_np[labels_np == 1] = temp
    labels_np[labels_np == label_main_network] = 1
    labels_np[labels_np == temp] = label_main_network

    return labels_np


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

def save_main_and_external_network_as_nifti(dir_nii, threshold, blur=0.0):
    
    # load tof data and get basic info
    tof_data = nibabel.load(f"{dir_nii}TOF.nii.gz")
    tof_np = tof_data.get_fdata()

    # set common label
    label = f"t{threshold:.2e}"
    if blur > 0:
        label += f"_blur{blur:.2e}"
    print(f"Label: {label}")

    # blur 
    if blur > 0:
        from scipy.ndimage import gaussian_filter
        # get pixel size
        hx, hy, hz = tof_data.header['pixdim'][1:4]
        tof_np = gaussian_filter(tof_np, sigma=blur*hx)


    # separe connected components
    labels_np, nlabels = connected_components(tof_np, threshold)
    print(f"Found {nlabels=} with tof>={threshold:.2e}")
    labels_np = main_network_equal_one(labels_np, nlabels, tof_np)
    nibabel.save(nibabel.Nifti1Image(labels_np, tof_data.affine), 
                 f"{dir_nii}connected_components_{label}.nii.gz")

    # save as nifti 
    main_network = np.zeros_like(labels_np, dtype=np.uint8)
    main_network[labels_np == 1] = 1
    outfilename = f"{dir_nii}main_network_{label}.nii.gz"
    print(f"Saving main network {outfilename}")
    nibabel.save(nibabel.Nifti1Image(main_network, tof_data.affine), 
                 outfilename)


    # get the skeleton of main network
    skeleton_np = skeletonize(main_network)

    # compute local thickness of the main network
    thickness_np = lt.local_thickness(main_network)


    # convert to integer
    skeleton_np = skeleton_np.astype(np.uint8)

    # save as nifti 
    outfilename = f"{dir_nii}skeleton_{label}.nii.gz"
    print(f"Saving skeleton to {outfilename}")
    nibabel.save(nibabel.Nifti1Image(skeleton_np, tof_data.affine), 
                 outfilename)
    
    # save local thickness
    outfilename = f"{dir_nii}thickness_{label}.nii.gz"
    print(f"Saving local thickness to {outfilename}")
    nibabel.save(nibabel.Nifti1Image(thickness_np, tof_data.affine), 
                 outfilename)
    
    def indices_surronding_box(array):
        """
        Given a nd-array, find the indices that contains all 
        nonzeros values.
        """
        indices = np.where(array > 0)
        min_indices = np.min(indices, axis=1)
        max_indices = np.max(indices, axis=1)
        return min_indices, max_indices

    
    # save skeleton with local thickness
    skeleton_thickness_np = skeleton_np * thickness_np
    # restrict data to top surrounding box
    min_indices, max_indices = indices_surronding_box(skeleton_thickness_np)
    skeleton_thickness_np = skeleton_thickness_np[min_indices[0]:max_indices[0]+1,
                              min_indices[1]:max_indices[1]+1,
                              min_indices[2]:max_indices[2]+1]
    
    offsets = [hx*min_indices[0],
               hy*min_indices[1],
               hz*min_indices[2]]
    
    new_affine = cp.copy(tof_data.affine)
    new_affine[0,0] = hx
    new_affine[1,1] = hy
    new_affine[2,2] = hz

    new_affine[0,3] += offsets[0]
    new_affine[1,3] += offsets[1]
    new_affine[2,3] += offsets[2]
    
    new_header = cp.copy(tof_data.header)
    new_header['pixdim'][1:4] = skeleton_thickness_np.shape
    
    out = nibabel.Nifti1Image(skeleton_thickness_np, new_affine, header=new_header)
    out.header["qoffset_x"] = 0.0
    out.header["qoffset_y"] = 0.0
    out.header["qoffset_z"] = 0.0
    
    outfilename = f"{dir_nii}skeleton_thickness_{label}.nii.gz"
    print(f"Saving skeleton with local thickness to {outfilename}")
    nibabel.save(out, outfilename)



    # find inlets of external network
    outfilename = f"{dir_nii}external_network_{label}.nii.gz"
    print(f"Saving external network to {outfilename}")
    external_np = find_external_network(labels_np)
    nibabel.save(nibabel.Nifti1Image(external_np, tof_data.affine), 
                 outfilename)
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mri', type=str)
    parser.add_argument('--threshold', type=float)
    parser.add_argument('--blur', type=float, default=0.0)
    args = parser.parse_args()

    save_main_and_external_network_as_nifti(args.mri, args.threshold, args.blur)

    

    
    