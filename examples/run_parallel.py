import multiprocessing as mp
import itertools
import os
from corrupt_and_reconstruct import corrupt_and_reconstruct,labels
import sys
import numpy as np
from niot import image2dat as i2d
from scipy.ndimage import zoom
from niot.conductivity2image import Barenblatt
from firedrake import COMM_WORLD, Ensemble, RectangleMesh

try:
    overwrite=(int(sys.argv[1])==1)
    print('overwritting')
except:
    overwrite=False
    print('skipping')
    
examples = ['y_net']
examples = ['frog_thk']#[f'y_net_hand_drawing/nref{i}' for i in [0]]#'frog_tongue'] 
#examples.append('y_net_hand_drawing/nref3')
#examples = ['y_net/']
#examples.append('y_net_hand_drawing/nref2')
#examples.append('y_net_hand_drawing/nref1')
#mask=['mask_large.png']#,'mask_large.png','mask_medium.png']
mask=['mask02.png']

nref=[0]
fems = ['DG0DG0']
gamma = [0.5]#, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
wd = [0]
wr = [0,1e-5]
ini = [0]
network_file = ['network.png']#,'mucnstp3.0e+00zero1.0e+07.npy','mup3.0e+00zero1.0e+07.npy','mupou3.0e+00zero1.0e+07']#,'mucnstp3.0e+00zero1.0e+01.npy']#,'network.png']
#network_file = ['network.png']
conf = ['ONE']#,'CORRUPTED','MASK']#,'MASK','CORRUPTED']
maps = [
   {'type':'identity'}, 
#    {'type':'heat', 'sigma': 1e-4},
#    {'type':'heat', 'sigma': 0.0005},
#    {'type':'pm', 'sigma': 0.0005, 'exponent_m': 2.0},
#    {'type':'pm', 'sigma': 1e-2, 'exponent_m': 2.0},
#    {'type':'pm', 'sigma': 1e-1, 'exponent_m': 2.0},
#     {'type':'pm', 'sigma': 5e-1, 'exponent_m': 2.0},
#     {'type':'pm', 'sigma': 1e0, 'exponent_m': 2.0},
]
tdens2image_scaling = [1.0]
method = [
    #'tdens_mirror_descent_explicit',
    #'tdens_mirror_descent_semi_implicit',
    #'gfvar_gradient_descent_explicit',
    'gfvar_gradient_descent_semi_implicit',
]



parameters=[examples,nref,fems,mask,gamma,wd,wr,network_file,ini,conf,maps,tdens2image_scaling,method]
combinations = list(itertools.product(*parameters))


def load_input(example, nref, mask, network_file):
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
        img_networks = f'{example}/{network_file}'
        np_network = i2d.image2numpy(img_networks,normalize=True,invert=True,factor=2**nref)
        print(f'using {img_networks}')
    except: 
        np_network = np.load(f'{example}/{network_file}')
        if nref != 0:
            np_network = zoom(np_network, 2**nref, order=0, mode='nearest')
        print(f'using {example}/{network_file}')
    np_mask = i2d.image2numpy(f'{example}/{mask}',normalize=True,invert=True,factor=2**nref)  


    return np_source, np_sink, np_network, np_mask
    
    

def fun(example,nref,fem,mask,gamma,wd,wr,network_file,ini,conf,tdens2image,tdens2image_scaling, method, comm=COMM_WORLD):
    labels_problem = labels(nref,fem,gamma,wd,wr,
                network_file,   
                ini,
                conf,
                tdens2image,
                tdens2image_scaling,
                method)
    label = '_'.join(labels_problem)
    print(f'inside {label=}')
    
    
    # load input    
    np_source, np_sink, np_network, np_mask = load_input(example, nref, mask, network_file)
    

    # create directory
    mask_name = os.path.splitext(mask)[0]
    if not os.path.exists(f'{example}/{mask_name}/'):
        try:
            os.makedirs(f'{example}/{mask_name}/')
        except:
            pass

    
    run = False
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
                            labels_problem=labels_problem,
                            comm=comm)   
    
def is_present(combination):
    labels_problem = labels(*combination[2:])
    
    example = combination[0]
    mask = combination[3]
    mask_name = os.path.splitext(mask)[0]
    label = '_'.join(labels_problem)
    directory=f'{example}/{mask_name}/'
    filename = os.path.join(directory, f'{label}.pvd')
    return os.path.exists(filename)

for combination in combinations:
    if is_present(combination) and not overwrite:
        combinations.remove(combination)

use_ensemble = False
if use_ensemble:
    n_processors = COMM_WORLD.size
    try:
        n_processors_x_ensemble = int(sys.argv[2])
    except:
        if len(combinations)>n_processors:
            n_processors_x_ensemble = 1
        else:
            n_processors_x_ensemble = 4

    my_ensemble = Ensemble(COMM_WORLD, n_processors_x_ensemble)
    n_ensemble = n_processors//n_processors_x_ensemble
    ensemble_rank = my_ensemble.ensemble_comm.rank
    comm = my_ensemble.comm
    print('ensemble_rank',ensemble_rank, 'comm rank',comm.rank, 'n_ensemble',n_ensemble)

    n_problem_x_ensemble = len(combinations)//n_ensemble

    combinations_per_ensemble = combinations[ensemble_rank*n_problem_x_ensemble:(ensemble_rank+1)*n_problem_x_ensemble]
    if comm.rank == 0:
        print(combinations)
        print(combinations_per_ensemble)

    for combination in combinations_per_ensemble:
        label='_'.join(labels(*combination[2:]))
        if comm.rank ==0:
            print(f'{label=}')
            print(*combination)
        example = combination[0]
        img_sources = f'{example}/source.png'
        np_source = i2d.image2numpy(img_sources,normalize=True,invert=True)
        mesh = i2d.build_mesh_from_numpy(np_source, mesh_type='cartesian', comm=my_ensemble.comm)
        
        print(comm.rank,'done')
        ierr = fun(*combination, my_ensemble.comm)
        if comm.rank ==0:
            print(f'{label=} ierr={ierr}')
else:
    with mp.Pool(processes = mp.cpu_count()) as p:
        p.starmap(fun, combinations)