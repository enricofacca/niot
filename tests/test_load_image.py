import os
import pytest
from niot import image2dat as i2d
import numpy as np

img_dir = os.path.dirname(__file__)
@pytest.mark.parametrize('img_path', [os.path.join(img_dir,'7.png')])
@pytest.mark.parametrize('mesh_type', ['simplicial','cartesian'])
@pytest.mark.parametrize('normalizeRGB', [True, False])
@pytest.mark.parametrize('inverteBW', [True, False])
def test_convert_write(img_path, mesh_type, normalizeRGB, inverteBW):
    """
    Load an image and create a mesh
    image -> numpy -> firedrake function-> numpy -> image 
    Compare the results and assert equality.
    """
    # convert to matrix
    np_img = i2d.image2numpy(img_path,normalize=normalizeRGB,invert=inverteBW)
    
    # create mesh
    mesh = i2d.build_mesh_from_numpy(np_img, mesh_type=mesh_type)

    # convert to firedrake
    fire_img = i2d.numpy2firedrake(mesh, np_img)

    # convert back to numpy
    np_img_converted = i2d.firedrake2numpy(fire_img)

    # write to file
    i2d.numpy2image(np_img_converted,'test_load_image.png', normalized=normalizeRGB, inverted=inverteBW)
    
    # read stored image
    np_img2 = i2d.image2numpy('test_load_image.png', normalize=normalizeRGB, invert=inverteBW)
    
    # compare saved and stored images
    assert np.allclose(np_img,np_img2)    
