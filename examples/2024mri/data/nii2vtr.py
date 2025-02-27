import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pyevtk.hl import gridToVTK, writeParallelVTKGrid
from niot import image2dat as i2d
import sys


nii_files = sys.argv[1:-1]
vtr_file = sys.argv[-1]


names = []
data = []
for i, nii_file in enumerate(nii_files):
    # remove extension nii.gz form the name
    data_name = nii_file.split('/')[-1].split('.')[0]
    names.append(data_name)


            
    epi_img = nib.load(nii_file)
    epi_img_data = epi_img.get_fdata()
    data.append(epi_img_data)
    
    hdr = epi_img.header
    nx, ny, nz = epi_img_data.shape
    dx, dy, dz = epi_img.header['pixdim'][1:4]
    offset = epi_img.affine[:3, 3]

    old_shape = epi_img_data.shape
    old_size = epi_img.header['pixdim'][1:4]
    old_offset = epi_img.affine[:3, 3]

    if i > 0:
        # check if all the images have the same shape
        if not (old_shape == epi_img_data.shape):
            raise ValueError("All images must have the same shape")
        # check if all the images have the same size

        if not (old_size == epi_img.header['pixdim'][1:4]).all():
            raise ValueError("All images must have the same size")
        
        # check if all the images have the same offset
        if not (old_offset == offset).all():
            raise ValueError("All images must have the same offset")

lx, ly, lz = nx * dx, ny * dy, nz * dz

print(names)

# Coordinates
# remove .vtr extension if passed in vtr_file
if vtr_file[-4:] == '.vtr':
    vtr_file = vtr_file[:-4]
i2d.numpy2vtr(data, [lx, ly, lz], vtr_file, names=names, offset=offset)
exit()

#epi_img = nib.load('QSM.nii.gz')
#epi_img_data = epi_img.get_fdata()
#np.save('QSM.npy',epi_img_data)


def show_slices(slices):
   """ Function to display row of image slices """
   fig, axes = plt.subplots(1, len(slices))
   for i, slice in enumerate(slices):
       axes[i].imshow(slice.T, cmap="gray", origin="lower")
       
slice_0 = epi_img_data[26, :, :]
slice_1 = epi_img_data[:, 30, :]
slice_2 = epi_img_data[:, :, 16]
show_slices([slice_0, slice_1, slice_2])
plt.suptitle("Center slices for EPI image")  
plt.show()
