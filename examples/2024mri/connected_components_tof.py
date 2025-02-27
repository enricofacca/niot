import nibabel
import numpy as np
import argparse
import cc3d


def connected_components(np_data, threshold, connectivity=26):
    network = np.zeros_like(np_data)
    network [np_data >= threshold] = 1
    labels, n_labels = cc3d.connected_components(network, 
                                                 connectivity=connectivity, 
                                                 binary_image=True, 
                                                 return_N=True)
    return labels, n_labels

def detect_main_network(labels_np, n_labels, tof_np):
    # find the index of the main network
    index_max_tof = np.unravel_index(tof_np.argmax(), tof_np.shape)
    label_largest_tof = labels_np[index_max_tof]
    print(f"Label: Largest tof: {label_largest_tof=}")

    #
    # We count the occurences of each label in the labels_np array
    #
    counts = np.bincount(labels_np.flatten())
    # 0 is background, 
    label_largest_component = np.argmax(counts[1:])+1
    print(f"Label: Largest component: {label_largest_component=}")

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mri', type=str)
    parser.add_argument('--coarseness', type=int)
    parser.add_argument('--threshold', type=float)
    args = parser.parse_args()

    # load tof data and get basic info
    dir_nii = args.mri+'/c'+f"{args.coarseness:02}/"
    tof_data = nibabel.load(f"{dir_nii}TOF.nii.gz")
    dimensions = tof_data.header.get_data_shape()[:3]
    tof_np = tof_data.get_fdata()

    # separe connected components
    labels_np, nlabels = connected_components(tof_np, args.threshold)
    print(f"Found {nlabels=} with tof>={args.threshold:.2e}")
    labels_np = detect_main_network(labels_np, nlabels, tof_np)
    nibabel.save(nibabel.Nifti1Image(labels_np, tof_data.affine), 
                 f"{dir_nii}connected_components_t{args.threshold:.2e}.nii.gz")

    # save as nifti 
    main_network = np.zeros_like(labels_np, dtype=np.uint8)
    main_network[labels_np == 1] = 1
    nibabel.save(nibabel.Nifti1Image(main_network, tof_data.affine), 
                 f"{dir_nii}main_network_t{args.threshold:.2e}.nii.gz")

    # inlets
    inlets_np = np.copy(labels_np)
    inlets_np[:,:,1:] = 0
    nibabel.save(nibabel.Nifti1Image(inlets_np, tof_data.affine), 
                 f"{dir_nii}inlets_t{args.threshold:.2e}.nii.gz")



    # find inlets of external network
    indices_bottom = labels_np[:,:,0]
    list_indices_bottom = np.unique(indices_bottom)
    # remove 0 (background) and 1(main network) from the list
    list_external = list_indices_bottom[2:]
    external_np = np.zeros_like(labels_np, dtype=np.uint8)
    for index in list_external:
        external_np[indices_bottom == index] = 1
    nibabel.save(nibabel.Nifti1Image(external_np, tof_data.affine), 
                 f"{dir_nii}external_network_t{args.threshold:.2e}.nii.gz")
    
    