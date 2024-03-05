import itertools
import os
from corrupt_and_reconstruct import corrupt_and_reconstruct,labels,run
import sys
import numpy as np
from niot import image2dat as i2d
from scipy.ndimage import zoom
from niot.conductivity2image import Barenblatt
from firedrake import COMM_WORLD, Ensemble, RectangleMesh
from firedrake.petsc import PETSc

overwrite=False
try:
    overwrite=(int(sys.argv[1])==1)
    print('overwritting')
except:
    overwrite=False
    print('skipping')
    
#examples = ['y_net_thk']
#examples = ['frog_thk']
examples = ['medium']
#examples.append('y_net_hand_drawing/nref3')
#examples = ['y_net_nref2/']
#examples.append('y_net_hand_drawing/nref2')
#examples.append('y_net_hand_drawing/nref1')
#mask=['mask_medium.png']#,'mask_small.png']
mask=['mask02.png']

nref=[0]
fems = ['DG0DG0']
gamma = [0.5]#, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
wd = [0, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]#,1e-1,1e0,1e1]#, 5e5, 1e3, 5e3, 1e4]#0,1e0,1e4]#,1e-5,1e-3,1e-1
wr = [0]
ini = [0,1e5]
#network_file = ['network_thick.png']
#network_file = ['network_artifacts.png']
network_file = ['mup3.0e+00zero5.0e+05.npy','mupou3.0e+00zero5.0e+05.npy']
#network_file = ['mup3.0e+00zero1.0e+07.npy']#,'mupou3.0e+00zero1.0e+07']#,'mucnstp3.0e+00zero1.0e+01.npy']#,'network.png']
#network_file = ['network.png']#,'thickness.npy']
conf = ['ONE','MASK']#,'CORRUPTED']
maps = [
    {'type':'identity','scaling': 100}, 
#    {'type':'pm', 'sigma': 1e-3, 'exponent_m': 2.0, 'scaling': 10},
#    {'type':'pm', 'sigma': 1e-4, 'exponent_m': 2.0, 'scaling': 10},
]

    #    {'type':'heat', 'sigma': 1e-4, 'scaling': 100},
#    {'type':'heat', 'sigma': 0.0005, 'scaling': 100},},
#    {'type':'pm', 'sigma': 1e-3, 'exponent_m': 2.0, 'scaling': 100},
#    {'type':'pm', 'sigma': 1e-3, 'exponent_m': 2.0, 'scaling': 10},
#    {'type':'pm', 'sigma': 1e-4, 'exponent_m': 2.0, 'scaling': 10},
#    {'type':'pm', 'sigma': 1e-3, 'exponent_m': 2.0, 'scaling': 1},
#    {'type':'pm', 'sigma': 1e-4, 'exponent_m': 2.0, 'scaling': 1},
    #    {'type':'pm', 'sigma': 1e-1, 'exponent_m': 2.0, 'scaling': 100},
#     {'type':'pm', 'sigma': 5e-1, 'exponent_m': 2.0, 'scaling': 100},
#     {'type':'pm', 'sigma': 1e0, 'exponent_m': 2.0,'scaling': 100},
method = [
    'tdens_mirror_descent_explicit',
    #'tdens_mirror_descent_semi_implicit',
    #'gfvar_gradient_descent_explicit',
    #'gfvar_gradient_descent_semi_implicit',
]



parameters=[examples,mask,nref,fems,gamma,wd,wr,network_file,ini,conf,maps,method]
combinations = list(itertools.product(*parameters))
    
def is_present(combination):
    labels_problem = labels(*combination[2:])
    example = combination[0]
    mask = combination[1]
    mask_name = os.path.splitext(mask)[0]
    label = '_'.join(labels_problem)
    directory=f'{example}/{mask_name}/'
    filename = os.path.join(directory, f'{label}.pvd')
    #if not os.path.exists(filename):
    #print(filename)
    
    return os.path.exists(filename)


ordered=len(combinations)
toremove=0
TODO=[]
for combination in combinations:
    if is_present(combination) and not overwrite : 
        toremove+=1
    else:
        TODO.append(combination)
    
        
print(f'{overwrite=}, {toremove=}, {ordered=}, {len(combinations)=} {len(TODO)=}')


for combination in TODO:
    labels_problem = labels(*combination[2:])
    example = combination[0]
    mask = combination[1]
    mask_name = os.path.splitext(mask)[0]
    label = '_'.join(labels_problem)
    if not is_present(combination):
        print (label )

del combinations
combinations = TODO

use_ensemble = False#
use_mpi = False
if use_mpi:
    if len(combinations)>1:
        raise ValueError('use_mpi=True is not compatible with len(combinations)>1')
    run(*combinations[0], comm=COMM_WORLD)
    exit()



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

    # this leads to a problem if the number of processors 
    # is not a multiple of n_processors_x_ensemble
    if n_ensemble * n_processors_x_ensemble != COMM_WORLD.size:
        PETSc.Sys.Print(f'rank={COMM_WORLD.size} is not compatible with\n'
                        +f'{n_processors_x_ensemble=}\n'
                        +f'{n_ensemble=}',comm=COMM_WORLD)
        raise ValueError('rank is not a multiple of n_processors_x_ensemble')
    
    comm = my_ensemble.comm
    #PETSc.Sys.Print(f'{ensemble_rank=} {comm.rank=} {n_ensemble=} {comm.size=}',comm=comm)
    comm.barrier()
    n_problem_x_ensemble = max(1,len(combinations)//n_ensemble)

    combinations_per_ensemble = combinations[ensemble_rank*n_problem_x_ensemble:(ensemble_rank+1)*n_problem_x_ensemble]
    #PETSc.Sys.Print(f'njobs={len(combinations)} {n_ensemble=}',comm=comm)
    comm.barrier()
    for i in range(n_ensemble):
        if i == ensemble_rank:
            PETSc.Sys.Print(f'{ensemble_rank=} with {comm.size} processors will do',comm=comm)
            for combination in combinations_per_ensemble:
                label = '_'.join(labels(*combination[2:]))
                PETSc.Sys.Print(f'  {label=}',comm=comm)
                # syncronize all processors
        COMM_WORLD.barrier()

    for i, combination in enumerate(combinations_per_ensemble):
        PETSc.Sys.Print(f'ENSEMBLE {my_ensemble.ensemble_comm.rank} will do {i+1} over {len(combinations_per_ensemble)}: job={label}',comm=comm)
        run(*combination, comm=my_ensemble.comm,n_ensemble=ensemble_rank)
        label = '_'.join(labels(*combination[2:]))
        PETSc.Sys.Print(f'ENSEMBLE {my_ensemble.ensemble_comm.rank} completed {i+1} over {len(combinations_per_ensemble)}: job={label}',comm=comm)

    PETSc.Sys.Print(f'FINISHED {my_ensemble.ensemble_comm.rank}',comm=comm)
else:
    import multiprocessing as mp
    with mp.Pool(processes = 20) as p:
        p.starmap(run, combinations)
