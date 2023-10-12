#!/usr/bin/env python
from PIL import Image
import numpy as np
import sys
import os
import subprocess
from copy import deepcopy as cp
import firedrake as fd
import matplotlib.pyplot as plt


def build_mesh_from_numpy(np_image, mesh_type='simplicial'): 
   '''
   Create a mesh (first axis size=1) from a numpy array
   '''
   if (np_image.ndim == 2):
      if (mesh_type == 'simplicial'):
            quadrilateral = False
      elif (mesh_type == 'cartesian'):
            quadrilateral = True
      print(f'Mesh type {mesh_type} quadrilateral {quadrilateral}')
      
      height, width  = np_image.shape
      print(f'{width=} {height=}')
      mesh = fd.RectangleMesh(
            width,
            height,
            1,
            height/width, 
            quadrilateral = quadrilateral,
            reorder=False,
            diagonal="right",
            )
            
   elif (np_image.ndim == 3):
      height, width, depth = np_image.shape
      
      if (mesh_type == 'simplicial'):
            hexahedral = False
      elif (mesh_type == 'cartesian'):
            hexahedral = True
      print(f'{mesh_type=} {hexahedral=}')
      mesh = fd.BoxMesh(nx=height,
                  ny=width, 
                  nz=depth,  
                  Lx=1, 
                  Ly=height/width,
                  Lz=height/depth,
                  hexahedral=hexahedral,
                  reorder=False,
                  #diagonal="default"
                  )
   else:
      raise ValueError('Only 2D and 3D images are supported')
   return mesh

def numpy2firedrake(mesh, value, name=None):
   '''
   Convert np array (2d o 3d) into a function compatible with the mesh solver.
   Args:
   
   value: numpy array (2d or 3d) with images values

   returns: piecewise constant firedake function 
   ''' 
   
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
      elif fd.mesh.geometric_dimension() == 3:
            # copy the value to copy the voxel value
            # flatten the numpy matrix (EF: I don't know why swapaxes is needed)
            print('reshaping')
            flat_value = value.swapaxes(0,2).flatten()
            print('reshaping done')
            ncopies = mesh.num_cells() // (value.shape[0]*value.shape[1]*value.shape[2])
            copies = np.zeros([ncopies,len(flat_value)])
            for i in range(ncopies):
               print(f'{i=}')
               copies[i,:] = flat_value[:]
            print('reshaping')
            cells_values = copies.swapaxes(0,1).flatten()
            print('reshaping done')
            # define function
            DG0 = FunctionSpace(mesh,'DG',0)
            img_function = Function(DG0, val=cells_values, name=name)

   elif (mesh.ufl_cell().cellname() == 'quadrilateral'):
      DG0 = FunctionSpace(mesh,'DG',0)
      img_function = Function(DG0)
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

   if (factor <1):
      # resize image
      width, height = img_file.size
      img_file = img_file.resize((int(width*factor),int(height*factor)), Image.ANTIALIAS)
   
   # get original image parameters...
   width, height = img_file.size
   

   # Make image Greyscale
   img_grey = img_file.convert('L')

   # Save Greyscale values
   print ('Pixels : '+str(img_grey.size[0])+' x '+ str(img_grey.size[1]))
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
      copy = copy * 255
   else:
      copy = 255 * copy/np.max(copy)

   # invert black and white (to have a white background when array is zero) 
   if inverted:
      copy = 255 - copy
   img = Image.fromarray(np.uint8(copy),'L')
   img.save(image_path)

def firedrake2numpy(function, dimensions):
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
         # get the values of the function
         with function.dat.vec_ro as f:
            value = f.array
            print(f'{value.shape=}')
            # reshape the values in a matrix of size (2,nx*ny)
            value = value.reshape([-1,2])
            # average the values along the first dimension
            value = np.mean(value,1)
            print(f'{value.shape=}')
            # reshape the values in a matrix of size (nx,ny)
            value = value.reshape([dimensions[0],dimensions[1]],order='F')

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
         new_shape = [dimensions[0],dimensions[1]]
         if mesh.geometric_dimension() == 3:
            new_shape.append(self.nz)
         value = value.reshape(new_shape,order='F')
         return np.flip(value,0)


def matrix2function(value,cartesian_mesh):
   """
   Convert np matrix into a piecewise function 
   defined ad the cartesina triangulation.
   Each pixel is splitted in two triangles.
   """ 
   
   # we flip vertically because images are read from left, right, top to bottom
   value = np.flip(value,0)

   # double the value to copy the pixel value to the triangles
   double_value = np.zeros([2,value.shape[0],value.shape[1]])
   double_value[0,:,:] = value[:,:]
   double_value[1,:,:] = value[:,:]
   triangles_image = double_value.swapaxes(0,2).flatten()
   DG0 = fd.FunctionSpace(cartesian_mesh,'DG',0)
   
   img_function = fd.Function(DG0)
   with img_function.dat.vec as d:
      d.array = triangles_image

   return img_function

def function2image(function,image_path,colorbar=True,vmin=None,vmax=None):
   """
   Print a fddrake function to grayscale image (0=white, >0=black)
   using matplotlib tools in fddrake
   """
   fig, axes = plt.subplots()
   if vmin is None:
      with function.dat.vec_ro as d:
         vmin = d.min()[1]
   if vmax is None:
      with function.dat.vec_ro as d:
         vmax = d.max()[1]
   print(f'{vmin=}{vmax=}')
   
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
   
def corrupt_image(path_original,path_corrupted,path_masks):
   """
   Corrupt an image by applying masks
   """
   np_corrupted = i2d.image2matrix(Image.open(path_orig))
   
   for mask in path_masks:
      np_corrupted -= i2d.image2matrix(Image.open(path_orig))
   


def image2grid(img_name,factor):
   
   img_file = Image.open(img_name)

   if (factor <1):
      # resize image
      width, height = img_file.size
      img_file = img_file.resize((int(width*factor),int(height*factor)), Image.ANTIALIAS)

   # get original image parameters...
   width, height = img_file.size
   min_side = min(width,height)
   mesh = fd.RectangleMesh(width,height,1,height/min_side,reorder=False)

   return mesh
   
      
if __name__ == "__main__":
    if len(sys.argv) > 1:
       source_path = sys.argv[1]
       sink_path = sys.argv[2]

       factor = 100
       mesh = image2grid(sink_path,factor)
       #source = image2function(source_path,mesh,factor)
       sink = image2function(sink_path,mesh,factor)
       #source.rename("source")
       sink.rename("sink")
       out_file = File("source_sink.pvd")
       out_file.write(sink)


       
       
       
    else:
        raise SystemExit("usage:  python  image2dat image data [grid]")
