#!/usr/bin/env python
from PIL import Image
import numpy as np
import sys
import os
import subprocess
from firedrake import *

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
   value = value/255

   return value

def matrix2function(value,cartesian_mesh):
   """
   Convert np matrix into a piecewise function defined ad the cartesina grid
   """ 

   
   # we flip vertically because images are read from left, right, top to bottom
   value = np.flip(value,0)

   # double the value to copy the pixel value to the triangles
   double_value = np.zeros([2,value.shape[0],value.shape[1]])
   double_value[0,:,:] = value[:,:]
   double_value[1,:,:] = value[:,:]
   triangles_image = double_value.swapaxes(0,2).flatten()
   DG0 = FunctionSpace(cartesian_mesh,'DG',0)
   
   img_function = Function(DG0)
   with img_function.dat.vec as d:
      d.array = triangles_image

   return img_function
   

def image2grid(img_name,factor):
   img_file = Image.open(img_name)

   if (factor <1):
      # resize image
      width, height = img_file.size
      img_file = img_file.resize((int(width*factor),int(height*factor)), Image.ANTIALIAS)

   # get original image parameters...
   width, height = img_file.size
   min_side = min(width,height)
   mesh = RectangleMesh(width,height,1,height/min_side,reorder=False)

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
