import multiprocessing as mp
import itertools
import os
from corrupt_and_reconstruct import corrupt_and_reconstruct,labels,imagebtp2np
import sys
import numpy as np
from niot import image2dat as i2d
from scipy.ndimage import zoom



try:
    overwrite=(int(sys.argv[1])==1)
    print('overwritting')
except:
    overwrite=False
    print('skipping')
    


examples = ['frog_coarse_thk']#[f'y_net_hand_drawing/nref{i}' for i in [0]]#'frog_tongue'] 
#examples.append('y_net_hand_drawing/nref3')
#examples.append('y_net/')
#examples.append('y_net_hand_drawing/nref2')
#examples.append('y_net_hand_drawing/nref1')
#mask=['mask_blur.png','mask_large.png','mask_medium.png']
mask=['mask02.png']

nref=[0,1]
fems = ['DG0DG0']
gamma = [0.5]#, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
wd = [0]#,1e-3,1e-2,5e-2,1e-1,1e0]
wr = [0,1e-6,1e-4]
ini = [0]
conf = ['ONE']#,'CORRUPTED','MASK']#,'MASK','CORRUPTED']
maps = [
   {'type':'identity'}, 
#    {'type':'heat', 'sigma': 1e-4},
#    {'type':'heat', 'sigma': 1e-3},
#    {'type':'pm', 'sigma': 1e-3, 'exponent_m': 2.0},
#    {'type':'pm', 'sigma': 1e-2, 'exponent_m': 2.0},
#    {'type':'pm', 'sigma': 1e-1, 'exponent_m': 2.0},
#     {'type':'pm', 'sigma': 5e-1, 'exponent_m': 2.0},
#     {'type':'pm', 'sigma': 1e0, 'exponent_m': 2.0},
]
tdens2image_scaling = [25]
method = [
    #'tdens_mirror_descent_explicit',
    #'tdens_mirror_descent_semi_implicit',
    #'gfvar_gradient_descent_explicit',
    'gfvar_gradient_descent_semi_implicit',
]

parameters=[examples,nref,fems,mask,gamma,wd,wr,ini,conf,maps,tdens2image_scaling,method]
combinations = list(itertools.product(*parameters))


def load_input(example, nref, mask):
    # btp inputs
    img_sources = f'{example}/source.png'
    img_sinks = f'{example}/sink.png'
    
    np_source = i2d.image2numpy(img_sources,normalize=True,invert=True,factor=2**nref)
    np_sink = i2d.image2numpy(img_sinks,normalize=True,invert=True,factor=2**nref)
    
    # taking just the support of the sources and sinks
    np_source[np.where(np_source>0.0)] = 1.0
    np_sink[np.where(np_sink>0.0)] = 1.0

    # balancing the mass
    nx, ny = np_source.shape
    print('npixel=',nx*ny)
    hx, hy = 1.0/nx, 1.0/ny
    mass_source = np.sum(np_source)*hx*hy
    mass_sink = np.sum(np_sink)*hx*hy
    if abs(mass_source-mass_sink)>1e-16:
        np_sink *= mass_source/mass_sink 

    # load image or numpy array
    try:
        img_networks = f'{example}/network.png'
        np_network = i2d.image2numpy(img_networks,normalize=True,invert=True,factor=2**nref)
    except: 
        np_network = np.load(f'{example}/network.npy')
        if nref != 0:
            np_network = zoom(np_network, 2**nref, order=0, mode='nearest')
    
    np_mask = i2d.image2numpy(f'{example}/{mask}',normalize=True,invert=True,factor=2**nref)  


    return np_source, np_sink, np_network, np_mask
    
    

def fun(example,nref,fem,mask,gamma,wd,wr,ini,conf,tdens2image,tdens2image_scaling, method):
    # load input
    np_source, np_sink, np_network, np_mask = load_input(example, nref, mask)
    

    # create directory
    mask_name = os.path.splitext(mask)[0]
    if not os.path.exists(f'{example}/{mask_name}/'):
        os.makedirs(f'{example}/{mask_name}/')

    labels_problem = labels(nref,fem,gamma,wd,wr,
                            ini,
                conf,
                tdens2image,
                tdens2image_scaling,
                method)
    
    label = '_'.join(labels_problem)
    directory=f'{example}/{mask_name}/'
    filename = os.path.join(directory, f'{label}.pvd')
    
    run = True
    if os.path.exists(filename):
        if not overwrite:
            run = False
    print(f'{filename} {os.path.exists(filename)=} {run=}')
    if run:
        ierr = corrupt_and_reconstruct(np_source,np_sink,np_network,np_mask, 
                                       nref=nref,
                            fem=fem,
                            gamma=gamma,
                            wd=wd,wr=wr,
                            corrupted_as_initial_guess=ini,
                            confidence=conf,
                            tdens2image=tdens2image,
                            tdens2image_scaling = tdens2image_scaling,
                            method=method,
                            directory=f'{example}/{mask_name}/',
                            label=label,
                            )
    #print(ierr)
    #print(example,fem,mask,gamma,wd,wr,ini,conf)
    
with mp.Pool(processes = mp.cpu_count()) as p:
    p.starmap(fun, combinations)

#for combination in combinations:
#    fun(*combination)
