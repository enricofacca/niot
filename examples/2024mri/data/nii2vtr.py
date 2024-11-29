import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pyevtk.hl import gridToVTK, writeParallelVTKGrid
from niot import image2dat as i2d
import sys

#epi_img = nib.load('T1.nii.gz')
#epi_img_data = epi_img.get_fdata()
#print(epi_img_data.shape)
#np.save('T1.npy',epi_img_data)

nii_file = sys.argv[1]
try:
    data_name = sys.argv[2]
except:
    data_name = 'data'
vtr_file = nii_file.replace('.nii.gz', '')

epi_img = nib.load(nii_file)
epi_img_data = epi_img.get_fdata()
hdr = epi_img.header
print(hdr)
print(epi_img.affine)


nx, ny, nz = epi_img_data.shape
dx, dy, dz = epi_img.header['pixdim'][1:4]
lx, ly, lz = nx * dx, ny * dy, nz * dz

# Coordinates
i2d.numpy2vtr([epi_img_data], [lx, ly, lz], vtr_file, names=[data_name])
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
