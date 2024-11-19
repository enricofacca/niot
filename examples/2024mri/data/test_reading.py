import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pyevtk.hl import gridToVTK, writeParallelVTKGrid
from niot import image2dat as i2d

#epi_img = nib.load('T1.nii.gz')
#epi_img_data = epi_img.get_fdata()
#print(epi_img_data.shape)
#np.save('T1.npy',epi_img_data)
epi_img = nib.load('../../../tests/mri/TOF.nii.gz')
epi_img_data = epi_img.get_fdata()
#np.save('TOF.npy',epi_img_data)
hdr = epi_img.header
print(hdr)



nx, ny, nz = epi_img_data.shape
dx, dy, dz = epi_img.header['pixdim'][1:4]
lx, ly, lz = nx * dx, ny * dy, nz * dz

# Coordinates
i2d.numpy2vtr(epi_img_data, [lx, ly, lz], "TOF", name='tof')
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
