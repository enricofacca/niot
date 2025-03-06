import os
import pytest
from niot import image2dat as i2d
import numpy as np
from firedrake import Function, FunctionSpace,VTKFile

save_pvd = False
images = [os.path.join(os.path.dirname(__file__),'7.png')]
mesh_types = ['simplicial','cartesian']
normalizeRGB = [True, False]
inverteBW = [True, False]

img_dir = os.path.dirname(__file__)
@pytest.mark.parametrize('img_path', images)
@pytest.mark.parametrize('mesh_type', mesh_types)
@pytest.mark.parametrize('normalizeRGB', normalizeRGB)
@pytest.mark.parametrize('inverteBW', inverteBW)
def test_convert_write(img_path, mesh_type, normalizeRGB, inverteBW):
    """
    Load an image and create a mesh
    image -> numpy -> firedrake function-> numpy -> image 
    Compare the results and assert equality.
    """
    # convert image 2 numpy matrix
    np_img = i2d.image2numpy(img_path,normalize=normalizeRGB,invert=inverteBW)
    i2d.numpy2image(np_img,'test_load_image2.png', normalized=normalizeRGB, inverted=inverteBW)
    np_img2 = i2d.image2numpy('test_load_image2.png', normalize=normalizeRGB, invert=inverteBW)
    assert np.allclose(np_img,np_img2)


    # create mesh
    mesh = i2d.build_mesh_from_numpy(np_img.shape, mesh_type=mesh_type)
    
    # convert to firedrake
    fire_img = i2d.numpy2firedrake(mesh, np_img)
    
    if save_pvd:
        out_file = VTKFile('test_image_firedrake.pvd')
        out_file.write(fire_img)

    # convert back to numpy
    if mesh_type == 'simplicial':
        mesh_cartesian = i2d.build_mesh_from_numpy(np_img.shape, mesh_type='cartesian')
        image_dg0 = Function(FunctionSpace(mesh_cartesian, "DG", 0))
        image_dg0.interpolate(fire_img)
        np_img_converted = i2d.firedrake2numpy(image_dg0)
    else:
        np_img_converted = i2d.firedrake2numpy(fire_img)
    assert np.allclose(np_img,np_img_converted)


    # write to file
    i2d.numpy2image(np_img_converted,'test_load_image.png', normalized=normalizeRGB, inverted=inverteBW)
    


    # read stored image
    np_img2 = i2d.image2numpy('test_load_image.png', normalize=normalizeRGB, invert=inverteBW)

    # compare saved and stored images
    assert np.allclose(np_img,np_img2)    


if __name__ == '__main__':
    test_convert_write(images[0], mesh_types[1], normalizeRGB[0], inverteBW[0])