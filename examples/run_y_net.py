import multiprocessing as mp
import itertools
import os
from corrupt_and_reconstruct import corrupt_and_reconstruct

examples = []#[f'y_net_hand_drawing/nref{i}' for i in [0]]#'frog_tongue'] 
examples.append('y_net_hand_drawing/nref3')
examples.append('y_net_hand_drawing/nref0')
examples.append('y_net_hand_drawing/nref2')
examples.append('y_net_hand_drawing/nref1')
fems = ['CR1DG0']
mask=['mask_large.png']
gamma = [0.5]
wd = [0]
wr = [0, 1e-4, 1e-3, 1e-2]
ini = [0]
conf = ['ONE']#,'CORRUPTED','MASK']#,'MASK','CORRUPTED']
maps = ['identity']#'heat','pm']#'identity','heat','pm']
sigma = [0.0]#1e-6,1e-4,1e-2]
tdens2image_scaling = [1e1]


parameters=[examples,fems,mask,gamma,wd,wr,ini,conf,maps,sigma,tdens2image_scaling]
combinations = list(itertools.product(*parameters))
print(combinations)


def fun(example,fem,mask,gamma,wd,wr,ini,conf,tdens2image,sigma,tdens2image_scaling):
    print('id', mp.current_process())
    print(example,fem,mask,gamma,wd,wr,ini,conf)
    img_sources = f'{example}/source.png'
    img_sinks = f'{example}/sink.png'
    img_networks = f'{example}/network.png'
    img_masks = f'{example}/{mask}'
    mask_name = os.path.splitext(mask)[0]
    weights = [wd,1,wr]
    run = True
    if run:
        corrupt_and_reconstruct(img_sources,img_sinks,img_networks,img_masks, 
                            scaling_size=1,
                            fem=fem,
                            gamma=gamma,
                            weights=weights,
                            corrupted_as_initial_guess=ini,
                            confidence=conf,
                            tdens2image=tdens2image,
                            directory=f'{example}/{mask_name}/',
                            sigma_smoothing=sigma,
                            tdens2image_scaling = tdens2image_scaling)
    print(example,fem,mask,gamma,wd,wr,ini,conf)
    
with mp.Pool(processes = mp.cpu_count()) as p:
    p.starmap(fun, combinations)

#for combination in combinations:
#    fun(*combination)