
from copy import deepcopy as cp

import numpy as np
import cc3d


import os
import sys

import argparse
import nibabel

from niot import image2dat as i2d

from firedrake import PETSc


import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(formatter={'float': '{:0.2e}'.format})



def experiment(args):

    field = "TOF"
    results = args.out


    # make directories
    if not  os.path.exists(results):
        os.mkdir(results)

    out_directory = results
    if not os.path.exists(out_directory):
        os.mkdir(out_directory)



    # load tof data
    tof_data = nibabel.load(args.mri+'TOF.nii.gz')
    tof_np = tof_data.get_fdata()
    hx, hy, hz = tof_data.header['pixdim'][1:4]
    lengths = np.array([float(tof_np.shape[0]*hx), float(tof_np.shape[1]*hy), float(tof_np.shape[2]*hz)])
    
    # load inlet data
    tof_inlet_np = i2d.image2numpy(f"{args.mri}/tof_inlets.png")
    tof_inlet_np = np.flipud(tof_inlet_np)
    tof_inlet_np = tof_inlet_np.reshape((tof_inlet_np.shape[0],tof_inlet_np.shape[1],1),order='F', copy=True)
    

    #
    # connected components
    # 

    connectivity = 26 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    inlets, n_inlets = cc3d.connected_components(tof_inlet_np, 
                                                 connectivity=connectivity, 
                                                 binary_image=True, 
                                                 return_N=True)
    print(f"{n_inlets=}")
    
    nibabel.save(nibabel.Nifti1Image(inlets, tof_data.affine), f"{args.out}/bottom_inlets.nii.gz")


    # the next vessels are connected due to the
    # presence of the Willis cycle
    internal_carotide_right = 12#5
    vertebral_right = 7#6

    internal_carotide_left = 9#16
    vertebral_left = 2#13

    main_inlets = np.zeros_like(tof_inlet_np, dtype=np.uint8)
    for l in [internal_carotide_right, vertebral_right, internal_carotide_left, vertebral_left]:
        main_inlets[inlets==l] = 1
    nibabel.save(nibabel.Nifti1Image(main_inlets, tof_data.affine), f"{args.out}/main_inlets.nii.gz")

    main_inlets_2d = main_inlets.reshape((main_inlets.shape[0],main_inlets.shape[1]), order='F', copy=True)
    main_inlets_2d = np.flipud(main_inlets_2d)
    i2d.numpy2image(main_inlets_2d, f"{args.out}/main_inlets.png",normalized=False)
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Reconstruct network')
    parser.add_argument("--mri", type=str, default="./data/", help="directory with mri data")
    parser.add_argument("--out", type=str, default="./data/", help="output directory")
    parser.add_argument("--threshold", type=float, default=250.0, help="tof threshold")
    args, unknown = parser.parse_known_args()

    

    experiment(args)
