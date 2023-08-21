#!/usr/bin/env python
from PIL import Image
import numpy as np
import sys
import os
import subprocess
from copy import deepcopy as cp
import firedrake as fire
import matplotlib.pyplot as plt

def image2matrix(img_name, normalize=True, invert=True, factor=1):
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

def matrix2image(numpy_matrix, image_path, normalized=True, inverted=True):
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
   DG0 = fire.FunctionSpace(cartesian_mesh,'DG',0)
   
   img_function = fire.Function(DG0)
   with img_function.dat.vec as d:
      d.array = triangles_image

   return img_function

def function2image(function,image_path,colorbar=True,vmin=None,vmax=None):
   """
   Print a firedrake function to grayscale image (0=white, >0=black)
   using matplotlib tools in firedrake
   """
   fig, axes = plt.subplots()
   if vmin is None:
      with function.dat.vec_ro as d:
         vmin = d.min()[1]
   if vmax is None:
      with function.dat.vec_ro as d:
         vmax = d.max()[1]
   print(f'{vmin=}{vmax=}')
   
   colors = fire.tricontourf(function, 
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
   mesh = fire.RectangleMesh(width,height,1,height/min_side,reorder=False)

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
