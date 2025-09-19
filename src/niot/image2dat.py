#!/usr/bin/env python
from PIL import Image
import numpy as np
import scipy
from copy import deepcopy as cp
import firedrake as fd
import matplotlib.pyplot as plt
from firedrake import mesh
import firedrake.cython.dmcommon as dmcommon


from firedrake.petsc import PETSc
from firedrake import COMM_WORLD, COMM_SELF, assemble, dx, DistributedMeshOverlapType  

#import localthickness
#from skimage.morphology import skeletonize
from firedrake.__future__ import interpolate

from firedrake import RectangleMesh, ExtrudedMesh,VTKFile

from mpi4py.MPI import SUM,MAX


from pyevtk.hl import gridToVTK, imageToVTK
import time

import pymetis
import networkx

###############################
# CONVENTIONS
# x: horizontal axis
# y: vertical axis
# z: depth axis
###############################
convention_2d_flipud = True
convention_2d_invert_rows_columns = True

###############################
# CONVENTIONS
# x: left axis
# y: vertical axis
# z: depth axis
###############################
convention_3d_axis_left_right = 0
convention_3d_axis_bottom_top = 1
convention_3d_axis_front_back = 2



def cartesian_grid_3d(shape_xyz, 
                      lengths=[1.0,1.0,1.0], 
                      comm=COMM_WORLD):
   nx,ny,nz = shape_xyz
   mesh2d = RectangleMesh(nx,ny,lengths[0],lengths[1],quadrilateral=True,comm=comm)
   mesh = ExtrudedMesh(mesh2d,nz,lengths[2]/nz)
   mesh.nx = nx
   mesh.ny = ny
   mesh.nz = nz

   
   mesh.xmin = 0
   mesh.xmax = lengths[0]
   mesh.ymin = 0
   mesh.ymax = lengths[1] 
   mesh.zmin = 0
   mesh.zmax = lengths[2]

   mesh.invert_rows_columns = False
   mesh.flip_up_down = False


   return mesh   

def build_mesh_from_numpy(np_shape, 
                          mesh_type='cartesian',
                          lengths=[1.0,1.0,1.0],
                          extrude=True,
                          comm=COMM_WORLD,
                          label_boundary=False,
                          invert_rows_columns=convention_2d_invert_rows_columns, 
                          flip_up_down=convention_2d_flipud
                          ): 
   '''
   Create a mesh (first axis size=1) from a numpy array
   '''
   if not ( (len(np_shape) == 2) or (len(np_shape) == 3)):
      raise ValueError('Only 2D and 3D images are supported')
   
   if not( ( mesh_type == 'simplicial') or (mesh_type == 'cartesian')):
      raise ValueError('Only simplicial and cartesian meshes are supported')


   if (len(np_shape) == 2):
      # here we swap the axes because the image is 
      # read from left, right, top to bottom
      if invert_rows_columns:
         ny, nx  = np_shape
      else:
         nx, ny = np_shape

      #PETSc.Sys.Print(f'npixel = {width*height} {comm.size=}', comm=comm)
      # create mesh
      quadrilateral = True
      if mesh_type == 'simplicial':
         quadrilateral = False

      PETSc.Sys.Print(invert_rows_columns,f'Creating mesh {nx}x{ny} {lengths=}')
      mesh = fd.RectangleMesh(
            nx=nx,
            ny=ny,
            Lx=lengths[0],
            Ly=lengths[1], 
            quadrilateral = quadrilateral,
            reorder=False,
            comm=comm
            )

      mesh.nx = nx
      mesh.ny = ny
      mesh.xmin = 0
      mesh.xmax = lengths[0]
      mesh.ymin = 0
      mesh.ymax = lengths[1]
      mesh.invert_rows_columns = invert_rows_columns
      mesh.flip_up_down = flip_up_down
      return mesh
      #print(f'{comm.size=} {comm.rank=} {mesh.comm.size=} {mesh.comm.rank=}' )
            
   elif (len(np_shape) == 3):
      if invert_rows_columns:
         nx, ny, nz = np_shape[1], np_shape[0], np_shape[2]
      else: 
         nx, ny, nz = np_shape
      
      print(invert_rows_columns, f'Creating mesh {nx}x{ny}x{nz} {lengths=}')

      if mesh_type == 'cartesian':
         if extrude:
            mesh = cartesian_grid_3d([nx,ny,nz],lengths,comm=comm)
            mesh.nx = nx
            mesh.ny = ny
            mesh.nz = nz
            mesh.xmin = 0
            mesh.xmax = lengths[0]
            mesh.ymin = 0
            mesh.ymax = lengths[1]
            mesh.zmin = 0
            mesh.zmax = lengths[2]
            return mesh
         else:
            mesh = fd.BoxMesh(
               nx=nx,
               ny=ny, 
               nz=nz,  
               Lx=lengths[0], 
               Ly=lengths[1],
               Lz=lengths[2],
               hexahedral=True,
               reorder=False,
               comm=comm
               )  
            mesh.nx = nx
            mesh.ny = ny
            mesh.nz = nz
            mesh.xmin = 0
            mesh.xmax = lengths[0]
            mesh.ymin = 0
            mesh.ymax = lengths[1] 
            mesh.zmin = 0
            mesh.zmax = lengths[2]
            return mesh
      
      if (mesh_type == 'simplicial'):        
         if label_boundary:
            mesh = fd.BoxMesh(
               nx=nx,
               ny=ny, 
               nz=nz,  
               Lx=lengths[0], 
               Ly=lengths[1],
               Lz=lengths[2],
               hexahedral=False,
               reorder=False,
               comm=comm
            )  
            mesh.nx = nx
            mesh.ny = ny
            mesh.nz = nz
            mesh.xmin = 0
            mesh.xmax = lengths[0]
            mesh.ymin = 0
            mesh.ymax = lengths[1] 
            mesh.zmin = 0
            mesh.zmax = lengths[2]
            return mesh

         else:
         
            xcoords = np.linspace(0, lengths[0], nx + 1, dtype=np.double)
            ycoords = np.linspace(0, lengths[1], ny + 1, dtype=np.double)
            zcoords = np.linspace(0, lengths[2], nz + 1, dtype=np.double)  

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
               
            mesh.nx = nx
            mesh.ny = ny
            mesh.nz = nz
            mesh.xmin = 0
            mesh.xmax = lengths[0]
            mesh.ymin = 0
            mesh.ymax = lengths[1] 
            mesh.zmin = 0
            mesh.zmax = lengths[2]

            return mesh

def mesh_from_topology(
        cells, 
        coords,
        reorder=None,
        distribution_parameters=None,
        comm=COMM_WORLD,
        name=mesh.DEFAULT_MESH_NAME,
        distribution_name=None,
        permutation_name=None,
    ):
    """
    Procedure taken from firedrake.mesh_utils.
    Passed topology and coordinates. Return a mesh. Only in 3D.
      Args:
         cells: list of node in each cells
         coords: list of coordinates
         reorder: (optional), should the mesh be reordered?
         distribution_parameters: options controlling mesh
               distribution, see :func:`.Mesh` for details.
         comm: Optional communicator to build the mesh on.
         name: (optional) name of the mesh
         distribution_name: (optional) name of the distribution
         permutation_name: (optional) name of the permutation

    """    

    dim = coords.shape[1]
    

    plex = mesh.plex_from_cell_list(
        dim, cells, coords, comm, mesh._generate_default_mesh_topology_name(name)
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
    m.init()
    return m


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
      nx = mesh.nx
      ny = mesh.ny
      return nx, ny
      
   elif mesh.geometric_dimension() == 3:
      nx = mesh.nx
      ny = mesh.ny
      nz = mesh.nz

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
      if convention_2d_invert_rows_columns:
         height, width  = np_shape
      else:
         width, height = np_shape

      if (mesh_shape[0] != width) or (mesh_shape[1] != height):
         PETSc.Sys.Print('Mesh and image have different shapes', mesh_shape, np_shape)
         check = False
   elif (len(np_shape) == 3):
      if (mesh_shape[0] != np_shape[0]) or (mesh_shape[1] != np_shape[1]) or (mesh_shape[2] != np_shape[2]):
         PETSc.Sys.Print('Mesh and image have different shapes', mesh_shape, np_shape)
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

   The code is based on https://www.firedrakeproject.org/interpolation.html#id6
   '''

   # Get information stored during mesh creation
   lengths = get_lengths(mesh)   
   nxyz = get_box_division(mesh)

   
   DG0 = fd.FunctionSpace(mesh,'DG',0)
   img_function = fd.Function(DG0)
   
   PETSc.Sys.Print(f'Converting numpy array to firedrake function {value.shape=} {nxyz=} {lengths=}')

   if mesh.geometric_dimension() == 3:    
      hx = lengths[0]/mesh.nx
      hy = lengths[1]/mesh.ny
      hz = lengths[2]/mesh.nz
      print(hx,hy,hz)
      #print(lengths)
      def my_data(xyz): 
         x = xyz[:,0]
         y = xyz[:,1]
         z = xyz[:,2]
         i = np.fix(x/hx).astype(int)
         j = np.fix(y/hy).astype(int)
         k = np.fix(z/hz).astype(int)
         #print(i)
         return value[j,i,k]
   elif mesh.geometric_dimension() == 2:
      invert_rows_columns = mesh.invert_rows_columns
      flip_up_down = mesh.flip_up_down
   

      print(mesh.invert_rows_columns, mesh.flip_up_down)
      #   
      # NOTE that we are reading the transpose of the value
      #
      if invert_rows_columns: 
         hx = lengths[0]/nxyz[0]
         hy = lengths[1]/nxyz[1]
         def my_data(xyz): 
            x = xyz[:,0]
            y = xyz[:,1]
            i = np.fix(x/hx).astype(int)
            j = np.fix(y/hy).astype(int)
            return value[j,i]
      else:
         nx = nxyz[0]
         ny = nxyz[1]
         Lx = lengths[0]
         Ly = lengths[1]
         hx = lengths[0]/nxyz[0]
         hy = lengths[1]/nxyz[1]
         PETSc.Sys.Print(f'Converting numpy array to firedrake function {Lx=} {Ly=}')
         PETSc.Sys.Print(f'Converting numpy array to firedrake function {nx=} {ny=}')
         PETSc.Sys.Print(f'Converting numpy array to firedrake function {hx=} {hy=}')
         def my_data(xyz): 
            x = xyz[:,0]
            y = xyz[:,1]
            if flip_up_down:
               y = Ly - y
            i = np.fix(y/hx).astype(int)
            j = np.fix(x/hy).astype(int)
            return value[i,j]
         
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


def simplex2cartesian(function, cartesian_mesh):
   """
   Return a function defined on a cartesian mesh from a function defined on a simplex mesh.
   """
   DQ0 = fd.FunctionSpace(cartesian_mesh, 'DG', 0)
   cartesian_function = fd.Function(DQ0)
   # interpolate the function
   fd.interpolate(function, cartesian_function)
   return cartesian_function


def firedrake2numpy(function, shape_np=None, invert_rows_columns=convention_2d_invert_rows_columns, fill=0.0):
   """
   Convert DG0firedrake function to numpy array (2d or 3d).
   It works only for meshes genereted with RectangleMesh or BoxMesh.
   If the mesh is simplicial, the function is averaged neighbouring cells.
   If the mesh is cartesian, the results is reshaped to the original image shape.
   TODO: deduced dimensions from mesh. Probably from numbe of boundary facets.
   """
   mesh = function.function_space().mesh()
   #if COMM_WORLD.Get_rank() > 0:
   #   raise ValueError('Only serial meshes are supported')

   if mesh.ufl_cell().is_simplex():
      raise ValueError('Only cartesian meshes are supported. Use simplex2cartesian first')
   
   if shape_np is None:
      shape = get_box_division(mesh)
   else:
      shape = shape_np
   
   
   
   def get_local_to_grid_indices_map(mesh):
      """
      build a map list of indices from the local
      index cell to the correspondence ij(k) index in the numpy array
      """
      # Get the centroid coordinates
      DQ0 = fd.FunctionSpace(mesh, 'DQ', 0)
      W = fd.VectorFunctionSpace(DQ0.ufl_domain(), DQ0.ufl_element())
      centroid_coordinates = fd.assemble(interpolate(DQ0.ufl_domain().coordinates, W))
      # Get the lengths of the box
      lengths = get_lengths(mesh)
      shape = get_box_division(mesh)
      #print('lengths', lengths)
      #print('shape', shape)
      #print(1./(np.array(shape)/np.array(lengths)))

      #print(centroid_coordinates.dat.data)
      
     
      indices = (centroid_coordinates.dat.data/lengths*shape).astype(int)
      #print(indices)
      return indices
   
   # Get current coordinates
   indices = get_local_to_grid_indices_map(mesh)
   
   if mesh.invert_rows_columns:
      out_shape = [shape[1], shape[0]]
   else:
      out_shape = [shape[0], shape[1]]

   if mesh.geometric_dimension() == 3:
      out_shape.append(shape[2])
   
   #print(indices)


   np_data = np.zeros(shape)
   np_data[:] = fill
   if mesh.geometric_dimension() == 3:
      # TODO: check if this is this the most efficient way to do this
      np_data[indices[:,0], indices[:,1], indices[:,2]] = function.dat.data_ro[:]
      if invert_rows_columns:
         np_data = np.transpose(np_data, (1,0,2))

   elif mesh.geometric_dimension() == 2:
      np_data[tuple(np.transpose(indices)[:])] = function.dat.data_ro[:]
      if invert_rows_columns:
         np_data = np.transpose(np_data)
      


   # with the following we create an array in all processes 
   global_data = mesh.comm.allreduce(np_data, op=SUM)

   return global_data
      

def numpy2vtr(np_images, lengths, vtk_file, names, comm=COMM_WORLD, offset=None):
   """
   Given a numpy array, save it to a vtk file.
   """
   # Create a grid
   if comm.rank == 0:
      if not isinstance(np_images, list):
         np_images = [np_images]
      if not isinstance(names, list):
         names = [names]

      if len(np_images) != len(names):
         raise ValueError('The number of images and names must be the same')
   

      if offset is None:
         offset = [0,0,0]
      data_shape = np_images[0].shape
      dim = len(data_shape)
      if ( dim == 2):
         #imageToVTK(vtk_file, cellData={name: reshaped})
         x = np.linspace(offset[0], lengths[0]+offset[0], data_shape[0]+1)
         y = np.linspace(offset[1], lengths[1]+offset[1], data_shape[1]+1)      
         z = np.array([offset[2]])

         cellData = {}
         for i in range(len(np_images)):
            np_image = np_images[i]
            name = names[i]
            reshaped = np_image.reshape((np_image.shape[0],np_image.shape[1],1))
            cellData.update({name: reshaped})
         #imageToVTK(vtk_file, pointData={name: reshaped})
         #reshaped = np_image.reshape((np_image.shape[0],np_image.shape[1],1))
         gridToVTK(vtk_file, x, y, z, cellData={name: reshaped})
   
      elif ( dim == 3):
         x = np.linspace(offset[0], lengths[0] + offset[0], data_shape[0]+1)
         y = np.linspace(offset[1], lengths[1] + offset[1], data_shape[1]+1)
         z = np.linspace(offset[2], lengths[2] + offset[2], data_shape[2]+1)   
         cellData = {}
         for i in range(len(np_images)):
            cellData.update({names[i]: np_images[i]})
         gridToVTK(vtk_file, x, y, z, cellData=cellData)


   


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




def ij_from_index(index, ny):
   "Map a linear index to a 2D index (i,j) assuming row-major order with ny rows"
   return np.array([index % ny, index // ny])

def index_from_ij(i,j,ny):
   "Map a 2D index (i,j) to a linear index assuming row-major order with ny rows"
   return i * ny + j


def coord_from_ij(irow,jcol,hx,hy, 
                  invert_rows_columns=convention_2d_invert_rows_columns):
   "Map a 2D index (i,j) to a coordinate (x,y)"

   if invert_rows_columns:
      return np.array([hx * jcol, hy * irow]).T
   else:
      return np.array([hy * irow, hx * jcol]).T
   
def ij_from_coord(x,y,hx,hy,
                  invert_rows_columns=convention_2d_invert_rows_columns):
   "Map a coordinate (x,y) to a 2D index (i,j) assuming row-major order with ny rows"
   if invert_rows_columns:
      return np.array([x/hx, y/hy]).astype(int).T
   else:
      return np.array([x/hy, y/hx]).astype(int).T
   



def topol_coords_edges_from_mask(mask, Lx=1.0, Ly=1.0, 
                                 invert_rows_columns=convention_2d_invert_rows_columns,
                                 flip_up_down=False#convention_2d_flipud,
                                 ):
   """
   Given and input array of shape (nx, ny) with 0/1 values, return the topology and coordinates a mesh 
   describing the 1 values.
   It returns also the conenectivity of the active cells.
   """

   
   print("invert",invert_rows_columns, mask.shape)   
   if invert_rows_columns:
      input_array = mask.T
      nx, ny = input_array.shape
   else:
      input_array = mask
      # rows are y, columncolss are x
      nx, ny  = input_array.shape
   
   hx, hy = Lx / nx, Ly / ny
      

   
   print(f"Creating mesh from mask")
   print(f"(nx,ny) = ({nx},{ny}) {Lx=} {Ly=} {hx=} {hy=}")
   active_cells = np.where(input_array>0)

   i_cells, j_cells = active_cells   
   ncell = i_cells.size

   # 
   # Define the connectivity list of active cells with the new cell numbering 
   #

   noffset = ny

   #
   # Brute force approach
   #
   edges = []
   for k in range(ncell):
      i, j = i_cells[k], j_cells[k]
      
      if i < nx-1 : 
         # cell below
         if input_array[i+1,j]>0:
               edge = [ index_from_ij(i,j,noffset), index_from_ij(i+1,j,noffset)]
               edges.append(edge)
      if j < ny-1:
         # cell right
         if input_array[i,j+1] > 0:
               edge = [ index_from_ij(i,j,noffset), index_from_ij(i,j+1,noffset)]
               edges.append(edge)
   edges = np.array(edges)

   # Define the new numbering and the inverse of the active cells
   active_cells = np.unique(edges.flatten())
   inverse_cells = np.zeros(nx*ny,dtype=int)
   inverse_cells[:] = -1
   inverse_cells[active_cells] = np.arange(ncell)
   new_edges = inverse_cells[edges]
         

   #
   # topology
   #
   nodes_in_cells = [
      i_cells * (ny + 1) + j_cells,
      i_cells * (ny + 1) + j_cells + 1,
      (i_cells + 1) * (ny + 1) + j_cells + 1,
      (i_cells + 1) * (ny + 1) + j_cells,
      ]
   nodes_in_cells = np.asarray(nodes_in_cells).swapaxes(0,-1).reshape(-1, 4)
   nodes = np.unique(nodes_in_cells.flatten())
   print(nodes)
   
   # Define the new numbering and the inverse of the active nodes
   inverse = np.zeros((nx+1)*(ny+1),dtype=int)
   inverse[:] = -1
   inverse[nodes] = np.arange(nodes.size)
   # Define the new connectivity
   new_nodes_in_cells = inverse[nodes_in_cells]


   # Define the coordinates of the nodes (note the ny+1)
   ij_coord = ij_from_index(nodes, ny+1)
   xy_coord = coord_from_ij(ij_coord[0,:], ij_coord[1,:], 
                            hx, hy, 
                            invert_rows_columns=invert_rows_columns)
   
   xcoords = np.linspace(0, Lx, nx + 1, dtype=np.double)
   ycoords = np.linspace(0, Ly, ny + 1, dtype=np.double)
   coords = np.asarray(np.meshgrid(xcoords, ycoords)).swapaxes(0, 2).reshape(-1, 2)
   
   #print("coords")
   #print(coords)
   #print("xy_coord")
   #print(xy_coord)
   
   #print(xy_coord)
   if flip_up_down:
      xy_coord[:,1] = Ly - xy_coord[:,1]


   return new_nodes_in_cells, xy_coord,  new_edges, active_cells, inverse_cells


def mask_base_height(nonzeros):
   """
   Calculates a 2D mask, base, and height map from a 3D binary array.

   The mask is a 2D array where each value is 1 if there is at least one non-zero
   element in the corresponding z-direction, and 0 otherwise.

   The base is a 2D array where each value is the count of non-zero elements in the

   The height map contains the index of the first non-zero element along the
   z-axis (axis 2) for each (x, y) coordinate.

   Args:
      nonzeros (np.ndarray): A 3D numpy array with binary (0 or 1) values.

   Returns:
      np.ndarray: A 2D numpy array where each value is the index of the first
                  non-zero element in the corresponding z-direction. If a
                  z-column contains all zeros, the value is set to -1.
   """

   def func_bottom(line):
      """
      Get the base and height of a 1D array
      """
      
      # start at the bottom and exist at the first non-zero
      nz = line.shape[0]
      base = 0
      while (base < nz) and (line[base] == 0):
         base += 1
      if base == nz:
         base = -1
      return base

   def func_top(line):
      """
      Get the height of a 1D array
      """
      nz = line.shape[0]      

      # start from the top and exist at the first non-zero
      top = nz-1
      while (top > 0) and (line[top] == 0):
         top -= 1
   
      return top
   
   base = np.apply_along_axis(func_bottom, 2, nonzeros)
   top = np.apply_along_axis(func_top, 2, nonzeros)
   height = top - base + 1
   height[base == -1] = 0
   mask = np.zeros_like(base)
   mask[base >= 0] = 1

   return mask, base, height

   # height = np.sum(nonzeros, axis=2)
   # mask = np.zeros_like(height)
   # mask[height > 0] = 1


   # # np.argmax returns the index of the first occurrence of the maximum value.
   # # Since the array is binary, the first '1' is the maximum value.
   # base = np.argmax(nonzeros, axis=2)

   # # A potential issue with np.argmax is that if a slice along the z-axis
   # # contains all zeros, it will return 0, which is an incorrect height.
   # # We need to identify these cases and mark them.
   # # Where the mask is True, set the height to -1 to indicate no non-zero
   # # element was found.
   # base[mask == 0] = -1

   # return mask, base, height


def mesh_from_2d_mask(mask2d, lengths, 
                      invert_rows_columns=True, 
                      flip_up_down=False,
                      height=None,
                      comm=COMM_WORLD):
   
   
   # 
   if invert_rows_columns:
      ny, nx = mask2d.shape
   else:
      nx, ny = mask2d.shape
   
   
   Lx, Ly = lengths

   # 1. Create topology and coordinates from the 2D mask
   topol, xy_coords, new_edges, active_cells, inverse_cells = topol_coords_edges_from_mask(mask2d, 
                                                                                           Lx, Ly,
                                                                                           invert_rows_columns=invert_rows_columns,
                                                                                           flip_up_down=flip_up_down)
   print(topol)
   print(xy_coords)
   
   
   ncells = len(topol)
   check_numbering = False
   if check_numbering:
      numbering = np.zeros_like(mask2d)
      for k, index in enumerate(inverse_cells):
         ij = ij_from_index(k, ny)
         numbering[ij[0],ij[1]] = index
      print("numbering")
      print(numbering)

   # 2. Set the distribution parameters
   distribution_parameters = None
   # If height is given, use it to balance the partitions
   if not(height is None) and comm.size > 1:
      icell, jcell = np.where(height > 0)
      ncells = len(icell)
      height2 = height[icell, jcell]

      # create the distribution parameters based on height2
      # to balance the number of cells in each partition
      distribution_parameters = set_distribution_parameters(
         new_edges, height2, ncells, comm)
      
   
   # 1. Create the 2D mesh distrubuted according to height-based partition
   selected_mesh2d = mesh_from_topology(topol, 
                                        xy_coords, 
                                        reorder=None, 
                                        distribution_parameters=distribution_parameters,
                                        comm=comm)
   
   selected_mesh2d.nx = nx
   selected_mesh2d.ny = ny
   selected_mesh2d.xmin = 0.0
   selected_mesh2d.ymin = 0.0
   selected_mesh2d.xmax = Lx
   selected_mesh2d.ymax = Ly

   selected_mesh2d.invert_rows_columns = invert_rows_columns
   selected_mesh2d.flip_up_down = flip_up_down
   selected_mesh2d.ncells = ncells
   selected_mesh2d.new_edges = new_edges
   selected_mesh2d.active_cells = active_cells
   selected_mesh2d.inverse_cells = inverse_cells
   
   return selected_mesh2d


def set_distribution_parameters(edges, height2, ncells, comm):
   if comm.size <= 1:
      return None
   
   # 3. Create the adjacency list for the graph
   def convert_to_adjacency_list(edges, num_vertices):
      # Initialize an empty list of size "num_vertices"
      # To store the adjacency list of each vertex
      adjacency_list = [[] for _ in range(num_vertices)]
      
      # Iterate through the edges
      for edge in edges:
         # Get the source and destination nodes
         source = edge[0]
         destination = edge[1]
         
         # Add the destination node to the source node's list
         adjacency_list[source].append(destination)
         
         # Add the source node to the destination node's list
         # (since the graph is bidirectional)
         adjacency_list[destination].append(source)
      
      # Return the adjacency list
      return adjacency_list

   
   Adj_list = convert_to_adjacency_list(edges,ncells)

   # 4. Run the partitioning algorithm
   _, partition_for_node = pymetis.part_graph(
      comm.size,
      adjacency=Adj_list,
      vweights=list(height2),
      options = pymetis.Options(contig=True)
   )

   # 5. Process the output, mapping indices back to node tuples
   partitions = [[] for _ in range(comm.size)]
   for i, part_num in enumerate(partition_for_node):
      partitions[part_num].append(i)

   size_partitions = [len(part) for part in partitions]
   weight_partitions = [int(sum(height2[part])) for part in partitions]
   print("Partitions sizes:", size_partitions)
   print("Partitions weights:", weight_partitions)
   flat_partions = [x for part in partitions for x in part]
   

   if comm.rank == 0:
      partitions = (size_partitions, flat_partions)#, list(range(ncells)))
   else:
      partitions = (None, None)  
   
   # set the distribution parameters
   distribution_parameters = {
      "partition": partitions,
      #"overlap_type": (DistributedMeshOverlapType.VERTEX, 1)
      "overlap_type": (DistributedMeshOverlapType.NONE, 0)
      }
   return distribution_parameters

      # if comm.size > 1:
      #    # Varaible layers need to be consistent with the distribution
      #    # of the 2D mesh. We use pymetis to partition the selected 2D mesh
      #    # weighted by the height, so that the global partition is balanced

      #    # 3. Create the adjacency list for the graph
      #    def convert_to_adjacency_list(edges, num_vertices):
      #       # Initialize an empty list of size "num_vertices"
      #       # To store the adjacency list of each vertex
      #       adjacency_list = [[] for _ in range(num_vertices)]
            
      #       # Iterate through the edges
      #       for edge in edges:
      #          # Get the source and destination nodes
      #          source = edge[0]
      #          destination = edge[1]
               
      #          # Add the destination node to the source node's list
      #          adjacency_list[source].append(destination)
               
      #          # Add the source node to the destination node's list
      #          # (since the graph is bidirectional)
      #          adjacency_list[destination].append(source)
            
      #       # Return the adjacency list
      #       return adjacency_list
      
         
      #    Adj_list = convert_to_adjacency_list(selected_mesh2d.new_edges, nvertices=ncells)

      #    # 4. Run the partitioning algorithm
      #    _, partition_for_node = pymetis.part_graph(
      #       comm.size,
      #       adjacency=Adj_list,
      #       vweights=list(height2),
      #       options = pymetis.Options(contig=True)
      #    )

      #    # 5. Process the output, mapping indices back to node tuples
      #    partitions = [[] for _ in range(n_parts)]
      #    for i, part_num in enumerate(partition_for_node):
      #       partitions[part_num].append(i)

      #    size_partitions = [len(part) for part in partitions]
      #    weight_partitions = [int(sum(height2[part])) for part in partitions]
      #    #print("Partitions sizes:", size_partitions)
      #    print("Partitions weights:", weight_partitions)
      #    flat_partions = [x for part in partitions for x in part]
         

      #    if comm.rank == 0:
      #       partitions = (size_partitions, flat_partions)#, list(range(ncells)))
      #    else:
      #       partitions = (None, None)  
         
      #    # set the distribution parameters
      #    distribution_parameters = {
      #       "partition": partitions,
      #       "overlap_type": (DistributedMeshOverlapType.NONE, 0)}
      # else:
      #    # no partitioning
      #    distribution_parameters = None



def mesh_from_3d_mask(mask3d, lengths, variable_layer=False, 
                      invert_rows_columns=True,
                      comm=COMM_WORLD):
   
   if invert_rows_columns:
      ny, nx, nz = mask3d.shape
   else:
      nx, ny, nz = mask3d.shape 
   Lx, Ly, Lz = lengths
   

   mask, base, height = mask_base_height(mask3d)
   #print("mask")
   #print(mask)
   #print("base")
   #print(base)
   #print("height")
   #print(height)

   #print(np.sum(mask>0),np.sum(base>0), np.sum(height>0))

   # if a varaible layer is requested, 
   # we need to partition the 2D mesh
   # according to the height map
   # Otherwise we just extrude standard distribution

   invert_rows_columns = True
   if variable_layer:
      # 3. Create the 2D mesh distrubuted according to height-based partition
      selected_mesh2d = mesh_from_2d_mask(
                     mask, lengths[0:2], 
                     invert_rows_columns = invert_rows_columns, 
                     flip_up_down = False,
                     height=height,
                     comm=COMM_WORLD,
                     )
      
      check_numbering = False
      if check_numbering:
         numbering = np.zeros_like(mask)
         for k, index in enumerate(selected_mesh2d.inverse_cells):
            ij = ij_from_index(k, ny)
            numbering[ij[0],ij[1]] = index
         print("numbering")
         print(numbering)
      



      # 1. Create the 2D mesh distrubuted according to height-based partition
      #distribution_parameters={"partition": (size_partions, flat_partion)}
      #distribution_parameters = None
      # selected_mesh2d = mesh_from_topology(topol, xy_coords, reorder=False, distribution_parameters=distribution_parameters)
      # selected_mesh2d.nx = nx
      # selected_mesh2d.ny = ny
      # selected_mesh2d.xmin = 0.0
      # selected_mesh2d.ymin = 0.0
      # selected_mesh2d.xmax = Lx
      # selected_mesh2d.ymax = Ly

      VTKFile("inside_selected_mesh2d.pvd").write(selected_mesh2d)
      base_fire = numpy2firedrake(selected_mesh2d, base, name="base")
      height_fire = numpy2firedrake(selected_mesh2d, height, name="height")
      mask_fire = numpy2firedrake(selected_mesh2d, mask, name="mask")
      partions = fd.Function(mask_fire.function_space(), name="partitions")
      with partions.dat.vec as d:
         d += comm.rank

      VTKFile("base_height.pvd").write(base_fire, height_fire,mask_fire,partions)
      

      # The main problem is to set the varaible layers given by
      # [ [base, height]] for each cell
      # since it must match the indexing of the distributed 2D mesh

      # With no overlap, I was able to make it work, but I guess there is
      # a more elegant way to do it.


      # pair into a list base2 and height2
      def get_local_to_grid_indices_map(mesh):
         """
         build a map list of indices from the local
         index cell to the correspondence ij(k) index in the numpy array
         """
         # Get the centroid coordinates
         #print(dir(mesh))
         #print(mesh.make_cell_node_list())

         # print(dir(mesh.cell_set))
         # print(dir(mesh.topology_dm))
         # plex = mesh.topology_dm
         # cStart, cEnd = plex.getHeightStratum(0) 
         # print(f"Cells {cStart} to {cEnd-1}")
         # print(plex.getCone(cStart))
         # print(mesh.cell_set.size)
         #
         # print(mesh.cell_set.partition_size)
         # print(mesh.cell_set.sizes)
         # print(mesh.cell_set[mesh.cell_set.indices[0]])
         # print(dir(mesh.cell_set.owned_part))
         # print(mesh.cell_set.owned_part.size)
         # print(mesh.cell_set.owned_part.offset)
         # print(mesh.cell_set.halo)

         # Derive a cell numbering from the Plex renumbering
         # tdim = dmcommon.get_topological_dimension(plex)
         # from firedrake.utils import as_cstr, IntType, RealType
         # entity_dofs = np.zeros(tdim+1, dtype=IntType)
         # entity_dofs[-1] = 1
         # cell_numbering, _ = mesh.create_section(entity_dofs)
         # #print(dir(cell_numbering))
         #print("offset",cell_numbering.getOffset(2))
         #print(cell_numbering.view)



         DQ0 = fd.FunctionSpace(mesh, 'DQ', 0)
         
         
         W = fd.VectorFunctionSpace(DQ0.ufl_domain(), DQ0.ufl_element())
         centroid_coordinates = fd.assemble(interpolate(DQ0.ufl_domain().coordinates, W))
         #print(dir(DQ0.ufl_domain()))
         #with DQ0.ufl_domain().coordinates.dat.vec_ro as d:
         #   print(d.array)
         #   print(d.array.shape)

         
         
         # Get the lengths of the box
         lengths = get_lengths(mesh)
         shape = get_box_division(mesh)
         #with centroid_coordinates.dat.vec_ro as d:
            #print("centroid_coordinates")
            #print(d.array)
         #print("shape", shape)
         

         hx = lengths[0]/shape[0]
         hy = lengths[1]/shape[1]
         #print("hx, hy", hx, hy)


         #indices = (centroid_coordinates.dat.data*factor).astype(int)
         coord = centroid_coordinates.dat.data_ro_with_halos
         print("coord", coord.shape)
         print(coord)
         indices = ij_from_coord(coord[:,0], coord[:,1], # there is a flip 
                                 hx, 
                                 hy, 
                                 invert_rows_columns=invert_rows_columns)
         print(indices)

         return indices
   
      # Varaible layers need to be consistent with the distribution
      # of the 2D mesh
      indices = get_local_to_grid_indices_map(selected_mesh2d)
      #print("indices", indices.shape)
      #print(indices)
      variable_layers = np.vstack(
         (base[indices[:,1],indices[:,0]],
          height[indices[:,1],indices[:,0]])
          ).T.tolist()
      #print("variable_layers", variable_layers)

      # create the 3D extruded mesh
      selected_mesh3d = ExtrudedMesh(selected_mesh2d, 
                                    layers=variable_layers, 
                                    layer_height=lengths[2]/mask3d.shape[2])
   
   else:
      selected_mesh2d = mesh_from_2d_mask(
                     mask, lengths[0:2], 
                     invert_rows_columns=True, 
                     flip_up_down=False,
                     comm=COMM_WORLD,
                     )
      
      print(dir(selected_mesh2d))
      print(selected_mesh2d.comm.rank, selected_mesh2d.cell_set.total_size, selected_mesh2d.cell_set.size)

      VTKFile("inside_selected_mesh2d.pvd").write(selected_mesh2d)

      selected_mesh3d = ExtrudedMesh(selected_mesh2d, 
                                    layers=nz, 
                                    layer_height=lengths[2]/mask3d.shape[2])

   ncells = selected_mesh2d.ncells
   print(f"{ncells} active cells from {nx*ny} total cells")

  


   
   selected_mesh3d.nx = nx
   selected_mesh3d.ny = ny
   selected_mesh3d.nz = nz
   selected_mesh3d.xmin = 0.0
   selected_mesh3d.ymin = 0.0
   selected_mesh3d.zmin = 0.0
   selected_mesh3d.xmax = lengths[0]
   selected_mesh3d.ymax = lengths[1]
   selected_mesh3d.zmax = lengths[2]

   selected_mesh3d.invert_rows_columns = invert_rows_columns
   
   return selected_mesh3d

   
