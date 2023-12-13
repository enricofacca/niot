import multiprocessing as mp
import itertools
import os
from corrupt_and_reconstruct import corrupt_and_reconstruct

examples = []#[f'y_net_hand_drawing/nref{i}' for i in [0]]#'frog_tongue'] 
#examples.append('frog_tongue/')
examples.append('y_net_hand_drawing/nref0')
#examples.append('asymmetric/nref0')
fems = ['DG0DG0']
#mask = ['mask02.png']#
mask=['mask_large.png']
gamma = [0.5]
wd = [0]
wr = [1e-4]
ini = [0]
conf = ['ONE']#,'CORRUPTED','MASK']#,'MASK','CORRUPTED']
maps = ['heat','pm']#'identity','heat','pm']
sigma = [1e-6,1e-4,1e-2]

scaling_forcing_y_net = [1e1]
scaling_forcing = [scaling_forcing_y_net]

parameters=[examples,fems,mask,gamma,wd,wr,ini,conf,maps,sigma,scaling_forcing]
combinations = list(itertools.product(*parameters))
print(combinations)


def fun(example,fem,mask,gamma,wd,wr,ini,conf,tdens2image,sigma,scaling_forcing):
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

        #np_source = i2d.image2numpy(img_source,normalize=True,invert=True,factor=scaling_size)
        #np_sink = i2d.image2numpy(img_sink,normalize=True,invert=True,factor=scaling_size) 
        #np_network = i2d.image2numpy(img_network,normalize=True,invert=True)
        #np_mask = i2d.image2numpy(img_mask,normalize=True,invert=True)
   
        #ndiv_0 = 52
        #np_source = np_network.copy()
        #np_sink = np_network.copy()
        #nref = np_network.shape[0]/ndiv_0
        #rows = range(0,2**nref)
        #cols = range(0,2**nref)

        #np_source = np_source[rows,:][:,cols]

        # taking just the support of the sources and sinks
        #np_source[np.where(np_source>0.0)] = 1.0
        #np_sink[np.where(np_sink>0.0)] = 1.0

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
                            scaling_forcing = scaling_forcing)
    print(example,fem,mask,gamma,wd,wr,ini,conf)
    
with mp.Pool(processes = mp.cpu_count()) as p:
    p.starmap(fun, combinations)

#for combination in combinations:
#    fun(*combination)