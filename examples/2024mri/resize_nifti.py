import os
from copy import deepcopy as cp 
import numpy as np
from scipy.ndimage import zoom
import argparse
import nibabel

import warnings
warnings.filterwarnings("ignore")

np.set_printoptions(formatter={'float': '{:0.2e}'.format})


def downsample(data, coarseness, mode="zoom"):
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
      
    if mode == "zoom":
        factors = tuple([1 if n==1 else 1/coarseness for n in data.shape])
        data = zoom(data, factors, order=0)
        return data

    elif mode == "max":
        # the extra boundary required
        pad_shape = tuple([(0, int(coarseness*np.ceil(n/coarseness)-n)) for n in data.shape])

        # pad with -inf
        data = np.pad(data, pad_shape, mode='constant', constant_values=-np.inf)
        

        # coarseness = 2
        # (nx//2,2,ny//2,2,nz//2,2)
        reshaped_shape = sum([[n//coarseness, coarseness] for n in data.shape],[])
        
        
        # reshape gathering neighbours cells and get the max
        extra_dim = tuple([2*i+1 for i in range(data.ndim)])
        
        data = data.reshape(reshaped_shape).max(axis=extra_dim)
        
        return data

def indices_restrict(data_shape, lengths, xyz_bounds):
    """ 
    Assuming that LX=1
    """
    indices_bounds = []
    for axis_index in range(len(data_shape)):
        n_axis = data_shape[axis_index]
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


def resize(args):
    # load tof data
    tof_data = nibabel.load(args.input)
    tof_np = tof_data.get_fdata()
    
    original_dimensions = tof_data.header.get_data_shape()[:3]
    print(tof_data.header)
    hx, hy, hz = tof_data.header['pixdim'][1:4]
    lengths = np.array([float(original_dimensions[0]*hx), 
                        float(original_dimensions[1]*hy), 
                        float(original_dimensions[2]*hz)])

    
    xyz_bounds=[[args.xmin,args.xmax],
                [args.ymin,args.ymax],
                [args.zmin,args.zmax]]
    indices_bounds = indices_restrict(tof_np.shape,
                                    lengths=lengths,
                                    xyz_bounds=xyz_bounds)
    offsets = [hx*indices_bounds[0][0],
               hy*indices_bounds[1][0],
               hz*indices_bounds[2][0]]
    
    tof_np = restrict(tof_np,indices_bounds)  
    


    # coarsen data
    coarseness = args.c
    if coarseness > 1:
        # check if coarseness is a power of 2
        if not np.log2(coarseness).is_integer():
            raise ValueError("Coarseness must be a power of 2")

        tof_np = downsample(tof_np,coarseness,args.cmode)
        
        hx *= coarseness
        hy *= coarseness
        hz *= coarseness

    new_affine = cp(tof_data.affine)
    new_affine[0,0] = hx
    new_affine[1,1] = hy
    new_affine[2,2] = hz

    new_affine[0,3] += offsets[0]
    new_affine[1,3] += offsets[1]
    new_affine[2,3] += offsets[2]
    
    new_header = cp(tof_data.header)
    new_header['pixdim'][1:4] = tof_np.shape
    
    out = nibabel.Nifti1Image(tof_np, new_affine, header=new_header)
    out.header["qoffset_x"] = 0.0
    out.header["qoffset_y"] = 0.0
    out.header["qoffset_z"] = 0.0
    print(out.header)
    print(out.affine)
    print(out.get_sform())
    print(out.get_qform())
    print(args.output)
    nibabel.save(out, args.output)        
        

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Resize and downsample nii file')
    #parser.add_argument("--field", type=str, default='TOF', help="TOF")
    parser.add_argument("--input", type=str, help="path to the input file")
    parser.add_argument("--output", type=str, help="path to the output file")
    parser.add_argument("--c", type=int, default=1, help="Coarseness")
    parser.add_argument("--cmode", type=str, default="max", help="Coarseness mode")
    parser.add_argument("--xmin", type=float, default=0.0, help="Lower bound x")
    parser.add_argument("--xmax", type=float, default=1000.0, help="Upper bound x")
    parser.add_argument("--ymin", type=float, default=0.0, help="Lower bound y")
    parser.add_argument("--ymax", type=float, default=1000.0, help="Upper bound y")
    parser.add_argument("--zmin", type=float, default=0.0, help="Lower bound z")
    parser.add_argument("--zmax", type=float, default=1000.0, help="Upper bound z")
    
    args, unknown = parser.parse_known_args()

    
    resize(args)
