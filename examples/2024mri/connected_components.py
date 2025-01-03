
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


def downsample(data,coarseness,mode="zoom"):
    """
    Downsample data by a factor of coarseness
    Args:
    data: numpy 2d or 3d array
    coarseness: int
    mode: str
        "zoom": uses scipy.ndimage.zoom
        "max" : uses max of neighbours cells
    Returns:
        data: numpy 2d or 3d array
    """
    if coarseness <= 1:
        return data
      
    PETSc.Sys.Print('coarsening image')
    if mode == "zoom":
        factors = tuple([1 if n==1 else 1/coarseness for n in data.shape])
        data = zoom(data, factors, order=0)
        PETSc.Sys.Print(data.shape)
        return data

    elif mode == "max":
        # the extra boundary required
        pad_shape = tuple([(0, int(coarseness*np.ceil(n/coarseness)-n)) for n in data.shape])

        # pad with -inf
        data = np.pad(data, pad_shape, mode='constant', constant_values=-np.inf)
        

        # coarseness = 2
        # (nx//2,2,ny//2,2,nz//2,2)
        dim = data.ndim
        reshaped_shape = sum([[n//coarseness, coarseness] for n in data.shape],[])
        
        
        # reshape gathering neighbours cells and get the max
        extra_dim = tuple([2*i+1 for i in range(data.ndim)])
        
        data = data.reshape(reshaped_shape).max(axis=extra_dim)
        
        return data

def indices_restrict(data, lengths, xyz_bounds):
    """ 
    Assuming that LX=1
    """
    indices_bounds = []
    for axis_index in range(len(data.shape)):
        n_axis = data.shape[axis_index]
        len_axis = lengths[axis_index]
        if xyz_bounds[axis_index] is None:
            indices_bounds.append([0,n_axis])
        else:
            lower, upper = xyz_bounds[axis_index]
            indices_bound = [max(0,int(lower/len_axis*n_axis)),min(int(upper/len_axis*n_axis),n_axis)]
            indices_bounds.append( indices_bound)
                

    return np.array(indices_bounds)

def restrict(data, indices_bounds):
    """ 
    Assuming that LX=1
    """
    data = data[indices_bounds[0][0]:indices_bounds[0][1],
                indices_bounds[1][0]:indices_bounds[1][1],
                indices_bounds[2][0]:indices_bounds[2][1]]
    data = np.ascontiguousarray(data)
    return data




def select_slice(tof_np, out_directory):
    # get bottom slice of tof
    tof_bottom_np = np.flipud(tof_np[:,:,0])
    tof_bottom_np /= tof_bottom_np.max()
    tof_bottom_np[tof_bottom_np<0.25] = 0
    # save as vtr and png
    #i2d.numpy2vtr(tof_bottom_np, lengths[0:2], f"{out_directory}/tof", name='tof')
    i2d.numpy2image(tof_bottom_np, f"{out_directory}/tof_bottom.png") 



def experiment(args):

    field = "TOF"
    coarseness = args.c
    results = args.out


    # make directories
    if not  os.path.exists(results):
        os.mkdir(results)

    test_case = f"{field}_{coarseness:02}"
    out_directory = results+test_case
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
    
    
    # restrict data
    restrict_domain = True
    if restrict_domain:
        indices_bounds = indices_restrict(tof_np,
                                        lengths=lengths,
                                        xyz_bounds=[
                                            [args.xmin,args.xmax],
                                            [args.ymin,args.ymax],
                                            [args.zmin,args.zmax]
                                        ])    
        tof_np = restrict(tof_np,indices_bounds)
        tof_inlet_np = restrict(tof_inlet_np,indices_bounds)
        lengths = np.array(tof_np.shape)*np.array([hx,hy,hz])
        
        PETSc.Sys.Print(f"Data shape: {tof_np.shape} Lengths: {lengths} After restriction ")

    # coarsen data
    mode = "max"
    if coarseness > 1:
        tof_np = downsample(tof_np,coarseness,mode)
        tof_inlet_np = downsample(tof_inlet_np,coarseness,mode)
        PETSc.Sys.Print(f"Coarse Data shape: {tof_np.shape}")



    #
    # connected components
    # 

    connectivity = 26 # only 4,8 (2D) and 26, 18, and 6 (3D) are allowed
    inlets, n_inlets = cc3d.connected_components(tof_inlet_np, 
                                                 connectivity=connectivity, 
                                                 binary_image=True, 
                                                 return_N=True)
    print(f"{n_inlets=}")


    network = tof_np.copy()
    threshold_network = args.threshold
    network[ tof_np >= threshold_network ] = 1.0
    network[ tof_np < threshold_network ] = 0.0

    
    labels, n_labels = cc3d.connected_components(network, 
                                                 connectivity=connectivity, 
                                                 binary_image=True, 
                                                 return_N=True)
    print(f"{n_labels=}")
    

    external_carotide_right = 19#18
    external_carotide_left = 2 

    external_back_right = 18# 17 
    external_back_left = 3

    # the next vessels are connected due to the
    # presence of the Willis cycle
    internal_carotide_right = 5#5
    vertebral_right = 7#6

    internal_carotide_left = 17#16
    vertebral_left = 14#13


    bottom_labels = labels[:,:,0:1]

    def inlets2network_labels(inlet_label, bottom_inlets, bottom_labels):
        """
        Given the inlet label, the bottom inlets and the bottom labels
        the label of the network is returned
        """
        labels = bottom_labels[bottom_inlets==inlet_label]
        print(f"{labels=}")
        # there may be same zeros
        labels = labels[labels != 0]


        print(f"{labels=}")
        label = np.unique(labels)
        print(f"{label=}")
        return label

    label_external_carotide_right = inlets2network_labels(external_carotide_right, 
                                                          bottom_inlets=inlets, 
                                                          bottom_labels=bottom_labels)
    label_external_carotide_left = inlets2network_labels(external_carotide_left,
                                                            bottom_inlets=inlets, 
                                                            bottom_labels=bottom_labels)
    
    label_external_back_right = inlets2network_labels(external_back_right, 
                                                          bottom_inlets=inlets, 
                                                          bottom_labels=bottom_labels)
    label_external_back_left = inlets2network_labels(external_back_left,
                                                            bottom_inlets=inlets, 
                                                            bottom_labels=bottom_labels)
    
    label_internal= inlets2network_labels( internal_carotide_right,
                                                            bottom_inlets=inlets, 
                                                            bottom_labels=bottom_labels)
    
    labels[labels==label_external_carotide_right] = 20000
    labels[labels==label_external_carotide_left] = 30000
    labels[labels==label_external_back_right] = 40000
    labels[labels==label_external_back_left] = 50000
    labels[labels==label_internal] = 60000




    # saving inputs in vtr
    i2d.numpy2vtr([network, labels], lengths, f"{args.out}/cc", names=['network','labels'])
    i2d.numpy2vtr(inlets, [lengths[0],lengths[1], hz], f"{args.out}/inlets", names='inlets')

    inlets_labels = inlets.reshape((inlets.shape[0],inlets.shape[1]), order='F', copy=True)
    inlets_labels = np.flipud(inlets_labels)
    i2d.numpy2image(inlets_labels, f"{args.out}/labels.png",normalized=False)
    # save as numpy
    np.save(f"{args.out}/inlets.npy",inlets)

    nibabel.save(nibabel.Nifti1Image(labels, tof_data.affine), f"{args.out}/labels.nii.gz")

        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Reconstruct network')
    #parser.add_argument("--field", type=str, default='TOF', help="TOF")
    parser.add_argument("--c", type=int, default=1, help="coarseing factor")
    parser.add_argument("--mri", type=str, default="./data/", help="directory with mri data")
    parser.add_argument("--out", type=str, default="./data/", help="output directory")
    parser.add_argument("--threshold", type=float, default=250.0, help="tof threshold")
    parser.add_argument("--xmin", type=float, default=0.0, help="Lower bound x")
    parser.add_argument("--xmax", type=float, default=1000.0, help="Upper bound x")
    parser.add_argument("--ymin", type=float, default=0.0, help="Lower bound y")
    parser.add_argument("--ymax", type=float, default=1000.0, help="Upper bound y")
    parser.add_argument("--zmin", type=float, default=0.0, help="Lower bound z")
    parser.add_argument("--zmax", type=float, default=1000.0, help="Upper bound z")
    
    args, unknown = parser.parse_known_args()

    

    experiment(args)
