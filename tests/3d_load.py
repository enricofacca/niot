import sys
import glob
import os
from copy import deepcopy as cp


from niot import image2dat as i2d
import numpy as np
from niot import utilities
from firedrake import *
from scipy.ndimage import zoom
import time


field = 'T1'
data = np.load(f'mri/{field}.npy')


print(data.shape)

coarseness = 4
if coarseness > 1:
    print('coarsening image')
    data = zoom(data, (1/coarseness,1/coarseness,1/coarseness), order=0)
    print(data.shape)

# create mesh
print('building mesh')
start = time.time()
mesh = i2d.build_mesh_from_numpy(data, mesh_type='cartesian')
mesh.name = 'mesh'
end = time.time()
print(f'mesh built {end-start}s')

# convert to firedrake
print('converting image')
start = time.time()
data_fire = i2d.numpy2firedrake(mesh, data, name=field)
end = time.time()
print(f'image converted {end-start}s')

utilities.save2pvd(data_fire,f'{field}_c{coarseness}.pvd')

print('saving to file')
start = time.time()
fname = f'{field}_c{coarseness}.h5'
with CheckpointFile(fname, 'w') as afile:
    afile.save_mesh(mesh)  # optional
    afile.save_function(data_fire)
end = time.time()
print(f'saved to file {end-start}s')

print ('loading from file')
start = time.time()
with CheckpointFile(fname, 'r') as afile:
    mesh = afile.load_mesh("mesh")
    field_read = afile.load_function(mesh, f"{field}")
end = time.time()
print(f'loaded from file {end-start}s')