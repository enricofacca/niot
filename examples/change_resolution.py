import sys
from niot import image2dat as i2d
from niot.utilities import save2pvd
# read file path
file_in = sys.argv[1]
nref = float(sys.argv[2])
file_out = sys.argv[3]

in_np = i2d.image2numpy(file_in,normalize=True,invert=True)
out_np = i2d.image2numpy(file_in,normalize=True,invert=True,factor=2**nref)
i2d.numpy2image(out_np,file_out, normalized=True, inverted=True)    

# convert to firedrake function and save
#mesh_old = i2d.build_mesh_from_numpy(in_np, mesh_type='cartesian')
#fire_old = i2d.numpy2firedrake(mesh_old, in_np, name="new")


#mesh_new = i2d.build_mesh_from_numpy(out_np, mesh_type='cartesian')
#fire_new = i2d.numpy2firedrake(mesh_new, out_np, name="new")

# save inputs
#save2pvd([fire_old,], 'old.pvd')
#save2pvd([fire_new,], 'new.pvd')