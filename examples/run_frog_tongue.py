import multiprocessing as mp
import itertools
import os
from corrupt_and_reconstruct import corrupt_and_reconstruct

examples = []#[f'y_net_hand_drawing/nref{i}' for i in [0]]#'frog_tongue'] 
#examples.append('y_net_hand_drawing/nref3')
examples=['./frog_tongue_bw/']
#examples.append('y_net_hand_drawing/nref2')
#examples.append('y_net_hand_drawing/nref1')
mask=['mask02.png']

nref=[0,1]
fems = ['DG0DG0']
# gamm from 0.1 to 0.8
gamma = [0.5]#,0.3,0.4,0.5,0.6,0.7,0.8]
wd = [0]#,1e-4,1e-3,1e-2,1e-1]
wr = [1e-6,1e-5,1e-4]#, 1e-2]
ini = [1,0]
conf = ['ONE']#,'CORRUPTED','MASK']#,'MASK','CORRUPTED']
maps = [
    {'type':'identity'}, 
    #{'type':'heat', 'sigma': 1e-4},
    #{'type':'pm', 'sigma': 1e-4, 'exponent_m': 2.0},
]
tdens2image_scaling = [1e2]
method = [
    #'tdens_mirror_descent_explicit',
    #'tdens_mirror_descent_semi_implicit',
    #'gfvar_gradient_descent_explicit',
    'gfvar_gradient_descent_semi_implicit',
]

parameters=[examples,nref,fems,mask,gamma,wd,wr,ini,conf,maps,tdens2image_scaling,method]
combinations = list(itertools.product(*parameters))




def fun(example,nref,fem,mask,gamma,wd,wr,ini,conf,tdens2image,tdens2image_scaling, method):
    img_sources = f'{example}/source.png'
    img_sinks = f'{example}/sink.png'
    img_networks = f'{example}/network.png'
    img_masks = f'{example}/{mask}'
    mask_name = os.path.splitext(mask)[0]
    
    if not os.path.exists(f'{example}/{mask_name}/'):
        os.makedirs(f'{example}/{mask_name}/')

    run = True
    if run:
        ierr = corrupt_and_reconstruct(img_sources,img_sinks,img_networks,img_masks, 
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
                            )
    print('out',ierr)
    #print(example,fem,mask,gamma,wd,wr,ini,conf)
    
with mp.Pool(processes = mp.cpu_count()) as p:
    p.starmap(fun, combinations)

#for combination in combinations:
#    fun(*combination)
