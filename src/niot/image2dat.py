#!/usr/bin/env python
from PIL import Image
import numpy as np
import scipy
from copy import deepcopy as cp
import firedrake as fd
import matplotlib.pyplot as plt
from firedrake import mesh

from firedrake.petsc import PETSc
from firedrake import COMM_WORLD, COMM_SELF

#import localthickness
#from skimage.morphology import skeletonize
from firedrake.__future__ import interpolate

import time

###############################
# CONVENTIONS
# x: horizontal axis
# y: vertical axis
# z: depth axis
###############################
convention_2d_flipud = True
convention_2d_invert_rows_columns = True



def cartesian_grid_3d(
   shape_xyz,
   lengths=[1.0,1.0,1.0],
   #origin=[0.0,0.0,0.0],
   #units="meters"
   ):
   nx,ny,nz = shape_xyz
   mesh2d = RectangleMesh(nx,ny,lenghts[0],lengths[1],quadrilater=True)
   mesh = ExtrudedMesh(mesh2d,nz,lenghts[2]/nz)
   return mesh

def build_mesh_from_numpy(np_image, 
                          mesh_type='simplicial',
                          lengths=None,
                          comm=COMM_WORLD,
                          label_boundary=False): 
   '''
   Create a mesh (first axis size=1) from a numpy array
   '''
   print("rank",comm.rank)
   if (np_image.ndim == 2):
      if (mesh_type == 'simplicial'):
         quadrilateral = False
      elif (mesh_type == 'cartesian'):
         quadrilateral = True
      
      # here we swap the axes because the image is 
      # read from left, right, top to bottom
      if convention_2d_invert_rows_columns:
         height, width  = np_image.shape
      else:
         width, height = np_image.shape

      # if no lengths are given, we assume that dimensions are proportional
      # to the size of the numpy array
      if lengths is None:
         lengths = (width,height)

      #PETSc.Sys.Print(f'npixel = {width*height} {comm.size=}', comm=comm)
      # create mesh
      mesh = fd.RectangleMesh(
            nx=width,
            ny=height,
            Lx=lengths[0],
            Ly=lengths[1], 
            quadrilateral = quadrilateral,
            reorder=False,
            diagonal="right",
         comm=comm
            )
      #print(f'{comm.size=} {comm.rank=} {mesh.comm.size=} {mesh.comm.rank=}' )
            
   elif (np_image.ndim == 3):
      height, width, depth = np_image.shape
      
      if (mesh_type == 'simplicial'):
         hexahedral = False
      elif (mesh_type == 'cartesian'):
         hexahedral = True
      if lengths is None:
         lengths = (height,width,depth)

      if label_boundary:
         mesh = fd.BoxMesh(
            nx=height,
            ny=width, 
            nz=depth,  
            Lx=lengths[0], 
            Ly=lengths[1],
            Lz=lengths[2],
            hexahedral=hexahedral,
            reorder=False,
            comm=comm
         )
      else:
         xcoords = np.linspace(0, lengths[0], height + 1, dtype=np.double)
         ycoords = np.linspace(0, lengths[1], width + 1, dtype=np.double)
         zcoords = np.linspace(0, lengths[2], depth + 1, dtype=np.double)  

         mesh = TensorBoxMesh(
            xcoords,
               ycoords,
               zcoords,
               reorder=None,
               distribution_parameters=None,
               diagonal="default",
               comm=comm,
               name="mesh",
               distribution_name=None,
               permutation_name=None,
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
   t = time.time()
   #mesh.init()
   dt=time.time()-t
   PETSc.Sys.Print(f"init mesh {dt}")
  
   # we attach this info to the mesh
   if (np_image.ndim == 2):
      mesh.nx = width
      mesh.ny = height
      mesh.xmin = 0
      mesh.xmax = lengths[0]
      mesh.ymin = 0
      mesh.ymax = lengths[1]
   if (np_image.ndim == 3):
      mesh.nx = height
      mesh.ny = width
      mesh.nz = depth
      mesh.xmin = 0
      mesh.xmax = lengths[0]
      mesh.ymin = 0
      mesh.ymax = lengths[1] 
      mesh.zmin = 0
      mesh.zmax = lengths[2]



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
   if mesh.geometric_dimension() == 2:
      # get x and y dimensions
      try:
         nx = mesh.nx
         ny = mesh.ny
      except:
         nx = mesh.exterior_facets.subset(3).size
         ny = mesh.exterior_facets.subset(1).size
      return nx, ny
      
   elif mesh.geometric_dimension() == 3:
      try:
         nx = mesh.nx
         ny = mesh.ny
         nz = mesh.nz
      except:
         if mesh.ufl_cell().is_simplex():
            xy = mesh.exterior_facets.subset(5).size/2
            yz = mesh.exterior_facets.subset(1).size/2
            xz = mesh.exterior_facets.subset(3).size/2
            nx = int(np.rint(np.sqrt(xy*xz/yz)))
            ny = int(np.rint(xy/nx))
            nz = int(np.rint(xz/nx))
         else:
            # get x, y and z dimensions
            xy = mesh.exterior_facets.subset(5).size
            yz = mesh.exterior_facets.subset(1).size
            xz = mesh.exterior_facets.subset(3).size
            nx = int(np.rint(np.sqrt(xy*xz/yz)))
            ny = int(np.rint(xy/nx))
            nz = int(np.rint(xz/nx))
      return nx, ny, nz
   

def get_lengths(mesh):
   """ 
   Given a mesh, return the lengths of the box in each direction.
   These informations are lost when the mesh is created.
   Args:
      mesh: firedrake mesh
   Returns:
      list with the lengths of the box in the direction x, y, [z]
   """  
   if mesh.geometric_dimension() == 2:
      # get x and y dimensions
      try:
         Lx = abs(mesh.xmax-mesh.xmin)
         Ly = abs(mesh.ymax-mesh.ymin)
      except:
         # get from min and max coordinates
         x = mesh.coordinates.dat.data[:,0] 
         y = mesh.coordinates.dat.data[:,1]
         Lx = abs(np.max(x)-np.min(x))
         Ly = abs(np.max(y)-np.min(y))

      return Lx, Ly
      
   elif mesh.geometric_dimension() == 3:
      # get x, y and z dimensions
      try:
         Lx = abs(mesh.xmax-mesh.xmin)
         Ly = abs(mesh.ymax-mesh.ymin)
         Lz = abs(mesh.zmax-mesh.zmin)
      except:
         # get from min and max coordinates
         x = mesh.coordinates.dat.data[:,0] 
         y = mesh.coordinates.dat.data[:,1]
         z = mesh.coordinates.dat.data[:,2]
         Lx = abs(np.max(x)-np.min(x))
         Ly = abs(np.max(y)-np.min(y))
         Lz = abs(np.max(z)-np.min(z))

      return Lx, Ly, Lz
   
def compatible(mesh, value):
   """
   Check that mesh and image have the same shape
   """
   np_shape = value.shape
   mesh_shape = get_box_division(mesh)

   check = True
   if (len(np_shape) == 2):
      if (mesh_shape[0] != np_shape[1]) or (mesh_shape[1] != np_shape[0]):
         print('Mesh and image have different shapes', mesh_shape, np_shape)
         check = False
   elif (len(np_shape) == 3):
      if (mesh_shape[0] != np_shape[0]) or (mesh_shape[1] != np_shape[1]) or (mesh_shape[2] != np_shape[2]):
         print('Mesh and image have different shapes', mesh_shape, np_shape)
         check = False
   else:
      raise ValueError('Only 2D and 3D images are supported')
   return check
      
def numpy2firedrake(mesh, value, name=None, lengths=None):
   '''
   Convert np array (2d o 3d) into a function compatible with the mesh solver.
   Args:
   
   value: numpy array (2d or 3d) with images values

   returns: piecewise constant firedake function 

   The code is based on https://www.firedrakeproject.org/interpolation.html#id6
   '''
   #if (not compatible(mesh, value) ):
   #   raise ValueError('Mesh and image are not compatible')   
   DG0 = fd.FunctionSpace(mesh,'DG',0)
   img_function = fd.Function(DG0)
   if lengths is None:
      lengths = get_lengths(mesh)
      
   nxyz = value.shape#get_box_division(mesh)

   if mesh.geometric_dimension() == 3:    
      hx = lengths[0]/nxyz[0]
      hy = lengths[1]/nxyz[1]
      hz = lengths[2]/nxyz[2]
      def my_data(xyz): 
         x = xyz[:,0]
         y = xyz[:,1]
         z = xyz[:,2]
         i = np.fix(x/hx).astype(int)
         j = np.fix(y/hy).astype(int)
         k = np.fix(z/hz).astype(int)
         return value[i,j,k]
   elif mesh.geometric_dimension() == 2:
      #   
      # NOTE that we are reading the transpose of the value
      # 
      hx = lengths[0]/nxyz[0]
      hy = lengths[1]/nxyz[1]
      def my_data(xyz): 
         x = xyz[:,0]
         y = xyz[:,1]
         i = np.fix(x/hx).astype(int)
         j = np.fix(y/hx).astype(int)
         return value[j,i]
   else:
      raise ValueError('Only 2d and 3d images are supported')
   
   # Get current coordinates
   W = fd.VectorFunctionSpace(DG0.ufl_domain(), DG0.ufl_element())
   coordinates = fd.assemble(interpolate(DG0.ufl_domain().coordinates, W))
   img_function = fd.Function(DG0,name=name)
   img_function.dat.data[:] = my_data(coordinates.dat.data)


   if (name is not None):
      img_function.rename(name,name)
   return img_function


def firedrake2numpy(function):
   """
   Convert DG0 firedrake function to numpy array (2d or 3d).
   It works only for meshes genereted with RectangleMesh or BoxMesh.
   If the mesh is simplicial, the function is averaged neighbouring cells.
   If the mesh is cartesian, the results is reshaped to the original image shape.
   TODO: deduced dimensions from mesh. Probably from numbe of boundary facets.
   """
   mesh = function.function_space().mesh()
   if COMM_WORLD.Get_rank() > 0:
      raise ValueError('Only serial meshes are supported')
   
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

            return value
            
      elif mesh.geometric_dimension() == 3:
         raise NotImplementedError('3D mesh not implemented yet')
   else:
      if (mesh.ufl_cell().cellname() != 'quadrilateral'):
            raise ValueError('Only simplicial and quadrilateral meshes are supported')
      # get the values of the function
      with function.dat.vec_ro as f:
         value = f.getArray(readonly=True)
         
         # reshape the values in a matrix of size (nx,ny)
         new_shape = get_box_division(mesh)
         value = value.reshape((new_shape[1],new_shape[0]), order='F')
         return value


def image2numpy(img_name, normalize=True, invert=True):
   """
   Given a path to an image, return a numpy matrix.
   The image is converted to greyscale, and it can be normalized ([0,255] to [0,1])
   and inverted (black to white and viceversa).
   """

   #open file in fileList:
   img_file = Image.open(img_name)

   # get original image parameters...
   width, height = img_file.size

   # Make image Greyscale
   img_grey = img_file.convert('L')

   # convert to a numpy array
   # preserving orientation
   value = np.asarray(img_grey.getdata(), dtype=float)
   value = value.reshape((height, width))
   if convention_2d_flipud:
      value = np.flipud(value)

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
   copy = numpy_matrix
   if convention_2d_flipud:
      copy = np.flipud(copy)
   #copy = numpy_matrix.transpose()
   
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



def thickness(network, pixel_h=None):
   """
   Given a binary network, return the local thickness.
   """
   fd.Citations().register('dahl2023fast')
   # they define thickness as radius
   np_local_thickness = localthickness.local_thickness(network) * 2 
   if pixel_h is not None:
      np_local_thickness *= pixel_h
   return np_local_thickness 

def skeleton(network):
   """
   Return a skeleton of the network
   """
   fd.Citations().register('van2014scikit')
   skeleton = skeletonize(network)
   return skeleton

def TensorBoxMesh(
   xcoords,
   ycoords,
   zcoords,
   reorder=None,
   distribution_parameters=None,
   diagonal="default",
   comm=COMM_WORLD,
   name="mesh",
   distribution_name=None,
   permutation_name=None,
):
   """Generate a mesh of a 3D box.

   :arg xcoords: Location of nodes in the x direction
   :arg ycoords: Location of nodes in the y direction
   :arg zcoords: Location of nodes in the z direction
   :kwarg distribution_parameters: options controlling mesh
         distribution, see :func:`.Mesh` for details.
   :kwarg diagonal: Two ways of cutting hexadra, should be cut into 6
      tetrahedra (``"default"``), or 5 tetrahedra thus less biased
      (``"crossed"``)
   :kwarg reorder: (optional), should the mesh be reordered?
   :kwarg comm: Optional communicator to build the mesh on.

   The boundary surfaces are numbered as follows:

   * 1: plane x == xcoords[0]
   * 2: plane x == xcoords[-1]
   * 3: plane y == ycoords[0]
   * 4: plane y == ycoords[-1]
   * 5: plane z == zcoords[0]
   * 6: plane z == zcoords[-1]
   """
   xcoords = np.unique(xcoords)
   ycoords = np.unique(ycoords)
   zcoords = np.unique(zcoords)
   nx = np.size(xcoords)-1
   ny = np.size(ycoords)-1
   nz = np.size(zcoords)-1

   for n in (nx, ny, nz):
      if n <= 0 or n % 1:
         raise ValueError("Number of cells must be a postive integer")
   # X moves fastest, then Y, then Z
   coords = (
      np.asarray(np.meshgrid(xcoords, ycoords, zcoords)).swapaxes(0, 3).reshape(-1, 3)
   )
   i, j, k = np.meshgrid(
      np.arange(nx, dtype=np.int32),
      np.arange(ny, dtype=np.int32),
      np.arange(nz, dtype=np.int32),
   )
   if diagonal == "default":
      v0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
      v1 = v0 + 1
      v2 = v0 + (nx + 1)
      v3 = v1 + (nx + 1)
      v4 = v0 + (nx + 1) * (ny + 1)
      v5 = v1 + (nx + 1) * (ny + 1)
      v6 = v2 + (nx + 1) * (ny + 1)
      v7 = v3 + (nx + 1) * (ny + 1)

      cells = [
         [v0, v1, v3, v7],
         [v0, v1, v7, v5],
         [v0, v5, v7, v4],
         [v0, v3, v2, v7],
         [v0, v6, v4, v7],
         [v0, v2, v6, v7],
      ]
      cells = np.asarray(cells).reshape(-1, ny, nx, nz).swapaxes(0, 3).reshape(-1, 4)
   elif diagonal == "crossed":
      v0 = k * (nx + 1) * (ny + 1) + j * (nx + 1) + i
      v1 = v0 + 1
      v2 = v0 + (nx + 1)
      v3 = v1 + (nx + 1)
      v4 = v0 + (nx + 1) * (ny + 1)
      v5 = v1 + (nx + 1) * (ny + 1)
      v6 = v2 + (nx + 1) * (ny + 1)
      v7 = v3 + (nx + 1) * (ny + 1)

      # There are only five tetrahedra in this cutting of hexahedra
      cells = [
         [v0, v1, v2, v4],
         [v1, v7, v5, v4],
         [v1, v2, v3, v7],
         [v2, v4, v6, v7],
         [v1, v2, v7, v4],
      ]
      cells = np.asarray(cells).reshape(-1, ny, nx, nz).swapaxes(0, 3).reshape(-1, 4)
      raise NotImplementedError(
         "The crossed cutting of hexahedra has a broken connectivity issue for Pk (k>1) elements"
      )
   else:
      raise ValueError("Unrecognised value for diagonal '%r'", diagonal)
   plex = mesh.plex_from_cell_list(
      3, cells, coords, comm, mesh._generate_default_mesh_topology_name(name)
   )

   m = mesh.Mesh(
      plex,
      reorder=reorder,
      distribution_parameters=distribution_parameters,
      name=name,
      distribution_name=distribution_name,
      permutation_name=permutation_name,
      comm=comm,
   )
   return m