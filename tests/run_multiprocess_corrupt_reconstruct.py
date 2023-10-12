import multiprocessing as mp
import itertools
from corrupt_and_reconstruct import corrupt_and_reconstruct

examples = [f'y_net_hand_drawing/nref{i}' for i in [0]]#'frog_tongue'] 
#examples.append('frog_tongue/')
fems = ['DG0DG0']
mask = ['mask_large']#,'mask_medium','mask_large']
gamma = [0.2]#,0.5,0.2]
wd = [1e-1]#,1e-2,1e0]
wr = [0.0,1e-4]
ini = [0]
conf = ['ONE']#,'CORRUPTED','MASK']#,'MASK','CORRUPTED']
maps = ['laplacian_smoothing']
sigma = [1e-8,1e-6,1e-4]

parameters=[examples,fems,mask,gamma,wd,wr,ini,conf,maps,sigma]
combinations = list(itertools.product(*parameters))#fems,mask,gamma,wd,wr,ini,conf))
print(combinations)

def fun(example,fem,mask,gamma,wd,wr,ini,conf,tdens2image,sigma):
    print('id', mp.current_process())
    print(example,fem,mask,gamma,wd,wr,ini,conf)
    img_sources = f'{example}/source.png'
    img_sinks = f'{example}/sink.png'
    img_networks = f'{example}/network.png'
    img_masks = f'{example}/{mask}.png'   
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
                            directory=f'{example}/multi_{mask}/',
                            runs_directory=f'{example}/multi_{mask}/runs/',
                            sigma_smoothing=sigma)
    print(example,fem,mask,gamma,wd,wr,ini,conf)
    
with mp.Pool(processes = mp.cpu_count()) as p:
    p.starmap(fun, combinations)

#for combination in combinations:
#    fun(*combination)