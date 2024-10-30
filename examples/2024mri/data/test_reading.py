import nibabel as nib
import matplotlib.pyplot as plt
import numpy as np
from pyevtk.hl import gridToVTK, writeParallelVTKGrid

#epi_img = nib.load('T1.nii.gz')
#epi_img_data = epi_img.get_fdata()
#print(epi_img_data.shape)
#np.save('T1.npy',epi_img_data)
epi_img = nib.load('TOF.nii.gz')
epi_img_data = epi_img.get_fdata()
np.save('TOF.npy',epi_img_data)
hdr = epi_img.header
print(hdr)



nx, ny, nz = epi_img_data.shape
dx, dy, dz = epi_img.header['pixdim'][1:4]
lx, ly, lz = nx * dx, ny * dy, nz * dz

ncells = nx * ny * nz
npoints = (nx + 1) * (ny + 1) * (nz + 1)

# Coordinates
x = np.arange(0, lx + 0.1 * dx, dx, dtype="float64")
y = np.arange(0, ly + 0.1 * dy, dy, dtype="float64")
z = np.arange(0, lz + 0.1 * dz, dz, dtype="float64")

# Variables
pressure = np.random.rand(ncells).reshape((nx, ny, nz))
temp = np.random.rand(npoints).reshape((nx + 1, ny + 1, nz + 1))

gridToVTK(
    "./rectilinear",
    x,
    y,
    z,
    cellData={"tof": epi_img_data },
    #pointData={"temp": temp},
)
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
