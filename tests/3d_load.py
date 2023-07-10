import sys
import glob
import os
from copy import deepcopy as cp
import numpy as np


sys.path.append('../src/niot')
from niot import NiotSolver
from utilities import save2pvd
from firedrake import *
from time import process_time

print('mesh 100')
time = process_time()
mesh = BoxMesh(nx=200,ny=200, 
                        nz=200,  
                        Lx=1, 
                        Ly=1,
                        Lz=1,
                        #hexahedral=True,
                        )
print('mesh 100',process_time()-time)

exit()

nx=5
ny=11
nz=17
example=np.zeros([nx,ny,nz])

x_array = np.linspace(1,nx+1,nx+1)[:-1]
y_array = np.linspace(nx+1,nx+ny+1,ny+1)[:-1]
z_array = np.linspace(nx+ny+1,nx+ny+nz+1,nz+1)[:-1]
print(x_array)
print(y_array)
print(z_array)

example[:,0,0] = x_array[:]
example[0,:,0] = y_array[:]
example[0,0,:] = z_array[:]

print(example[:,:,:])
niot_solver = NiotSolver('DG0DG0',example)
save2pvd(niot_solver.img_observed,'3d_load.pvd')

t1 = np.load('TOF.npy')


t1 = t1[:,:,0:-1]
print(t1.shape)

coarseness = 4
from scipy.ndimage import zoom
t1_vtu = zoom(t1, (1/coarseness,1/coarseness,1/coarseness), order=0)
print(t1_vtu.shape)
niot_solver = NiotSolver('DG0DG0',t1_vtu)
save2pvd(niot_solver.img_observed,'t1.pvd')