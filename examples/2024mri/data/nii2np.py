import nibabel
import numpy as np
import argparse
def nii2np(nii_file):
    nii = nibabel.load(nii_file)
    data = nii.get_fdata()
    return data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str)
    parser.add_argument('--output', type=str, default='output.npy')
    args = parser.parse_args()


    data = nii2np(args.input)
    np.save(args.output, data)