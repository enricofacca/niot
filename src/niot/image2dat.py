#!/usr/bin/env python
from PIL import Image
import numpy as np
import sys
import os
import subprocess
from copy import deepcopy as cp
import firedrake as fire
import matplotlib.pyplot as plt

def image2matrix(img_name,factor=1):
   #open file in fileList:
   img_file = Image.open(img_name)

   if (factor <1):
      # resize image
      width, height = img_file.size
      img_file = img_file.resize((int(width*factor),int(height*factor)), Image.ANTIALIAS)
   
   # get original image parameters...
   width, height = img_file.size
   

   format = img_file.format
   mode = img_file.mode

   # Make image Greyscale
   img_grey = img_file.convert('L')

   # Save Greyscale values
   print ('Pixels : '+str(img_grey.size[0])+' x '+ str(img_grey.size[1]))
   value = np.asarray(img_grey.getdata(), dtype=float)
   value = value.reshape((img_grey.size[1], img_grey.size[0]))
   value = 1-value/255

   return value

def matrix2image(numpy_matrix, image_path, lenght_unit=100):
   """ Given a (numpy) matrix with values between 0 and 1
   save a grayscale image to file. Grayscale is flipped,
   which means that 0 is white and 1 is white
   """
   # Creates PIL image
   img = Image.fromarray(np.uint8((1-numpy_matrix) * 255) , 'L')
   img.save(image_path)

   # alternative
   #from scipy.misc import toimage
   #im = toimage(numpy_matrix)
   #im.save(image_path)


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

def function2image(function,image_path,vmin=0):
   """
   Print a firedrake function to grayscale image (0=white, >0=black)
   using matplotlib tools in firedrake
   """
   fig, axes = plt.subplots()
   colors = fire.tricontourf(function, 
      axes=axes, cmap='gray_r', vmin=vmin)
   fig.colorbar(colors)
   plt.gca().set_aspect('equal')
   plt.savefig(image_path)
   
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
