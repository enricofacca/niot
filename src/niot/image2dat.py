#!/usr/bin/env python
from PIL import Image
import numpy as np
from copy import deepcopy as cp
import firedrake as fd
import matplotlib.pyplot as plt
from firedrake import COMM_WORLD
from firedrake import mesh as meshtools

from firedrake.petsc import PETSc

def build_mesh_from_numpy(np_image, mesh_type='simplicial', comm=COMM_WORLD): 
   '''
   Create a mesh (first axis size=1) from a numpy array
   '''
   if (np_image.ndim == 2):
      if (mesh_type == 'simplicial'):
         quadrilateral = False
      elif (mesh_type == 'cartesian'):
         quadrilateral = True
      
      # here we swap the axes because the image is 
      # read from left, right, top to bottom
      height, width  = np_image.shape

      # create mesh
      mesh = fd.RectangleMesh(
            width,
            height,
            1,
            height/width, 
            quadrilateral = quadrilateral,
            reorder=False,
            diagonal="right",
            comm=comm
            )
            
   elif (np_image.ndim == 3):
      height, width, depth = np_image.shape
      
      if (mesh_type == 'simplicial'):
            hexahedral = False
      elif (mesh_type == 'cartesian'):
            hexahedral = True
      
      mesh = fd.BoxMesh(nx=height,
                   ny=width, 
                   nz=depth,  
                   Lx=1, 
                   Ly=height/width,
                   Lz=height/depth,
                   hexahedral=hexahedral,
                   reorder=False,
                   #diagonal="default"
                   comm=comm
                  )
   else:
      raise ValueError('Only 2D and 3D images are supported')

   # the following is needed because (from Firedrake documentation)
   """
   Finish the initialisation of the mesh.  Most of the time
   this is carried out automatically, however, in some cases (for
   example accessing a property of the mesh directly after
   constructing it) you need to call this manually.
   """
   mesh.init()

   return mesh

def get_box_division(mesh):
   """ 
   Given a mesh, return the number of divisions in each direction.
   These informations are lost when the mesh is created.
   Args:
      mesh: firedrake mesh
   Returns:
      list with the number of divisions in the direction x, y, [z]
   """
   # get x and y dimensions   
   if mesh.geometric_dimension() == 2:
      # get x and y dimensions
      nx = mesh.exterior_facets.subset(3).size
      ny = mesh.exterior_facets.subset(1).size
      return [nx, ny]
   elif mesh.geometric_dimension() == 3:
      # get x, y and z dimensions
      xy = mesh.exterior_facets.subset(5).size
      yz = mesh.exterior_facets.subset(1).size
      xz = mesh.exterior_facets.subset(3).size

      print(f'xy {xy} yz {yz} xz {xz}')
      nx = int(np.rint(np.sqrt(xy*xz/yz)))
      ny = int(np.rint(xy/nx))
      nz = int(np.rint(xz/nx))
      return nx, ny, nz
   
def compatible(mesh, value):
   """
   Ceck that mesh and image have the same shape
   """
   np_shape = value.shape
   mesh_shape = get_box_division(mesh)
   print(mesh_shape)
   check = True
   if (len(np_shape) == 2):
      if (mesh_shape[0] != np_shape[1]) or (mesh_shape[1] != np_shape[0]):
         print(f'shape data {np_shape} mesh_shape {mesh_shape}')
         check = False
   elif (len(np_shape) == 3):
      if (mesh_shape[0] != np_shape[0]) or (mesh_shape[1] != np_shape[1]) or (mesh_shape[2] != np_shape[2]):
         print(f'shape data {np_shape} mesh_shape {mesh_shape}')
         check = False
   else:
      raise ValueError('Only 2D and 3D images are supported')
   return check
      
def numpy2firedrake(mesh, value, name=None):
   '''
   Convert np array (2d o 3d) into a function compatible with the mesh solver.
   Args:
   
   value: numpy array (2d or 3d) with images values

   returns: piecewise constant firedake function 
   '''
   #if (not compatible(mesh, value) ):
   #   raise ValueError('Mesh and image are not compatible')
   
   # we flip vertically because images are read from left, right, top to bottom
   if mesh.ufl_cell().is_simplex():
      # Each pixel is splitted in two triangles.
      if mesh.geometric_dimension() == 2:
            value = np.flip(value,0)

            # double the value to copy the pixel value to the triangles
            double_value = np.zeros([2,value.shape[0],value.shape[1]])
            double_value[0,:,:] = value[:,:]
            double_value[1,:,:] = value[:,:]
            triangles_image = double_value.swapaxes(0,2).flatten()
            DG0 = fd.FunctionSpace(mesh,'DG',0)
            
            img_function = fd.Function(DG0)#, value=triangles_image, name=name)
            with img_function.dat.vec as d:
               d.array = triangles_image
            if name is not None:
               img_function.rename(name,name)
      elif mesh.geometric_dimension() == 3:
            value = np.flip(value,0)
            # copy the value to copy the voxel value
            # flatten the numpy matrix (EF: I don't know why swapaxes is needed)
            flat_value = value.swapaxes(0,2).flatten()
            ncopies = mesh.num_cells() // value.size
            copies = np.zeros([ncopies,len(flat_value)])
            for i in range(ncopies):
               copies[i,:] = flat_value[:]
            cells_values = copies.swapaxes(0,1).flatten()
            # define function
            DG0 = fd.FunctionSpace(mesh,'DG',0)
            img_function = fd.Function(DG0, val=cells_values, name=name)

   elif (
      (mesh.ufl_cell().cellname() == 'quadrilateral')
      or (mesh.ufl_cell().cellname() == 'hexahedron')):
      DG0 = fd.FunctionSpace(mesh,'DG',0)
      img_function = fd.Function(DG0)
      value_flipped = np.flip(value,0)
      with img_function.dat.vec as d:
         d.array = value_flipped.flatten('F')
   else:
      raise ValueError('Only simplicial and quadrilateral meshes are supported')
   if (name is not None):
      img_function.rename(name,name)
   return img_function

def image2numpy(img_name, normalize=True, invert=True, factor=1):
   """
   Given a path to an image, return a numpy matrix.
   The image is converted to greyscale, and it can be normalized ([0,255] to [0,1])
   and inverted (black to white and viceversa).
   """

   #open file in fileList:
   img_file = Image.open(img_name)

   if abs(factor-1)>1e-16:
      # resize image
      width, height = img_file.size
      img_file = img_file.resize((int(width*factor),int(height*factor)), resample=Image.NEAREST)
   
   # get original image parameters...
   width, height = img_file.size
   

   # Make image Greyscale
   img_grey = img_file.convert('L')

   # Save Greyscale values
   value = np.asarray(img_grey.getdata(), dtype=float)
   value = value.reshape((img_grey.size[1], img_grey.size[0]))
   if invert:
      value = 255 - value
   
   if normalize:
      value = value/255
   
   return value

def numpy2image(numpy_matrix, image_path, normalized=True, inverted=True):
   """ Given a (numpy) matrix,
   save a grayscale image to file. Grayscale can be inverted.
   """
   # Creates PIL image
   copy = cp(numpy_matrix) 
   if normalized:
      # normalize to [0,255]
      # this can lead to rounding errors
      copy *= 255
   
   # invert black and white (to have a white background when array is zero) 
   if inverted:
      copy = 255 - copy
   img = Image.fromarray(np.uint8(copy),'L')
   img.save(image_path)

def firedrake2numpy(function):
   """
   Convert DG0 firedrake function to numpy array (2d or 3d).
   It works only for meshes genereted with RectangleMesh or BoxMesh.
   If the mesh is simplicial, the function is averaged neighbouring cells.
   If the mesh is cartesian, the results is reshaped to the original image shape.
   TODO: deduced dimensions from mesh. Probably from numbe of boundary facets.
   """
   mesh = function.function_space().mesh()
   
   if mesh.ufl_cell().is_simplex():
      # Each pixel is splitted in two triangles.
      if mesh.geometric_dimension() == 2:
         nx, ny = get_box_division(mesh)
         # get the values of the function
         with function.dat.vec_ro as f:
            value = f.array
            # reshape the values in a matrix of size (2,nx*ny)
            value = value.reshape([-1,2])
            # average the values along the first dimension
            value = np.mean(value,1)
            # reshape the values in a matrix of size ()
            value = value.reshape([ny,nx],order='F')

            return np.flip(value,0)
            
      elif mesh.geometric_dimension() == 3:
         raise NotImplementedError('3D mesh not implemented yet')
   else:
      if (mesh.ufl_cell().cellname() != 'quadrilateral'):
            raise ValueError('Only simplicial and quadrilateral meshes are supported')
      # get the values of the function
      with function.dat.vec_ro as f:
         value = f.array
         # reshape the values in a matrix of size (nx,ny)
         new_shape = get_box_division(mesh)
         nx = new_shape[0]
         new_shape[0] = new_shape[1]
         new_shape[1] = nx
         value = value.reshape(new_shape, order='F')
         return np.flip(value,0)


def function2image(function,image_path,colorbar=True,vmin=None,vmax=None):
   """
   Print a firedrake function to grayscale image (0=white, >0=black)
   using matplotlib tools in fddrake
   """
   fig, axes = plt.subplots()
   if vmin is None:
      with function.dat.vec_ro as d:
         vmin = d.min()[1]
   if vmax is None:
      with function.dat.vec_ro as d:
         vmax = d.max()[1]
   
   colors = fd.tricontourf(function, 
      axes=axes, 
      #cmap='gray_r',
      cmap='Greys',
      #cmap='binary',
      extend="both", vmin=vmin, vmax=vmax)
   
   if colorbar:
      #plt.gca().set_aspect('equal')
      fig.colorbar(colors)
      
      fig.subplots_adjust(bottom = 0)
      fig.subplots_adjust(top = 1)
      fig.subplots_adjust(right = 1)
      fig.subplots_adjust(left = 0)
      
      plt.gca().axis('off')
      plt.gca().axis('tight')
      plt.gca().axis('equal')
   else:
      fig.subplots_adjust(bottom = 0)
      fig.subplots_adjust(top = 1)
      fig.subplots_adjust(right = 1)
      fig.subplots_adjust(left = 0)
      plt.gca().axis('off')
      plt.gca().axis('tight')
   
   
   plt.savefig(image_path)#,bbox_inches='tight',transparent=True, pad_inches=0)
