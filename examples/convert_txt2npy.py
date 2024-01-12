import numpy as np
import sys
from niot import image2dat as i2d
from niot.utilities import save2pvd

# read file path
file_path = sys.argv[1]
file_skeleton = sys.argv[2]



thickness = np.loadtxt(file_path, dtype=np.float32, delimiter='\t', skiprows=0)
# replace Nan with 0 in data
thickness = np.nan_to_num(thickness)
print(thickness.shape)
pixel_h = 1.0 / thickness.shape[1]
thickness *= pixel_h

# save as npy file
np.save('thickness.npy', thickness)
np_skeleton = i2d.image2numpy(file_skeleton,normalize=True,invert=True)


thickness_skeleton = thickness * np_skeleton

pouseille = thickness_skeleton**4
np.save('thickness_skeleton.npy', thickness_skeleton)
np.save('pouseille.npy', pouseille)

# convert to firedrake function and save
mesh = i2d.build_mesh_from_numpy(thickness, mesh_type='cartesian')


# save inputs    
thickness_fire = i2d.numpy2firedrake(mesh, thickness, name="thickness_fire")
np_skeleton_fire = i2d.numpy2firedrake(mesh, np_skeleton, name="skeleton_fire")
thickness_skeleton_fire = i2d.numpy2firedrake(mesh, thickness_skeleton, name="thickness_skeleton_fire")
pouseille_fire = i2d.numpy2firedrake(mesh, pouseille, name="pouseille_fire")

# save inputs
save2pvd([
    thickness_fire,
    np_skeleton_fire,
    thickness_skeleton_fire,
    pouseille_fire
    ], 
    'thickness.pvd'
    )



exit()