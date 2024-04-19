"""
Script running the examples in parallel.

Example of usage:
    python run_parallel.py --help (show the help)
    python run_parallel.py (run all the examples in parallel)
"""

import argparse
from corrupt_and_reconstruct import labels, load_input, corrupt_and_reconstruct
import itertools
import multiprocessing as mp
import os
from firedrake import COMM_WORLD


def is_present(combination):
    """
    Check if pvd file already exists.
    """
    labels_problem = labels(*combination[2:])
    example = combination[0]
    mask = combination[1]
    mask_name = os.path.splitext(mask)[0]
    label = '_'.join(labels_problem)
    directory=f'{example}/{mask_name}/'
    filename = os.path.join(directory, f'{label}.pvd')    
    return os.path.exists(filename)



#
# common setup
#
fems = ['DG0DG0']
wr = [0.0]
method = [
    'tdens_mirror_descent_explicit',
]


def figure2():
    #
    # Combinations producting the data for Figure 2
    #
    examples = ['y_net/']
    mask = ['mask_medium.png']
    nref=[0,1,2]
    gamma = [0.2,0.5,0.8]
    wd = [0] # set the discrepancy to zero
    ini = [0]
    # the following are not influent since wd=weight discrepancy is zero
    network_file = ['network.png'] #
    conf = ['ONE']
    maps = [
        {'type':'identity','scaling': 1.0}, 
    ]
    parameters=[examples,mask,nref,fems,gamma,wd,wr,network_file,ini,conf,maps,method]
    combinations = list(itertools.product(*parameters))

    return combinations

def figure3():
    #
    # combinations for Figure 3
    #
    examples = ['y_net_nref2/']
    mask = ['mask_medium.png']
    nref=[0]
    gamma = [0.5]
    wd = [1e-2,5e-2,1e-1,1e0] # set the discrepancy to zero
    ini = [0,1e5]
    network_file = ['network.png'] #
    conf = ['MASK']
    maps = [
        {'type':'identity','scaling': 10},
        {'type':'pm', 'sigma': 1e-3, 'exponent_m': 2.0, 'scaling': 50},
    ]

    parameters=[examples,mask,nref,fems,gamma,wd,wr,network_file,ini,conf,maps,method]
    combinations_figure_3_mask = list(itertools.product(*parameters))

    ini = [0]
    conf = ['ONE']
    parameters=[examples,mask,nref,fems,gamma,wd,wr,network_file,ini,conf,maps,method]
    combinations_figure_3_one = list(itertools.product(*parameters))
    
    combinations_figure_3 = combinations_figure_3_mask + combinations_figure_3_one

    return combinations_figure_3


def figure4():
    #
    # figure for figure 4, 
    #
    examples = ['y_net_nref2/']
    mask = ['mask_medium.png']
    nref=[0]
    gamma = [0.5]
    wd = [1e-2,5e-2,1e-1,1e0] # set the discrepancy to zero
    ini = [0,1e5]
    network_file = ['network_artifacts.png'] #
    conf = ['ONE']
    maps = [
        {'type':'identity','scaling': 10},
    ]
    
    parameters=[examples,mask,nref,fems,gamma,wd,wr,network_file,ini,conf,maps,method]
    combinations_figure_4 = list(itertools.product(*parameters))

    return combinations_figure_4

def figure5():
    #
    # combination for Figure 5 b
    #
    examples = ['frog_tongue/']
    mask = ['mask.png']
    nref=[0]
    gamma = [0.5]
    wd = [0] # set the discrepancy to zero
    ini = [0]
    # the following are not influent since wd=weight discrepancy is zero
    network_file = ['network.png'] 
    conf = ['ONE']
    maps = [
        {'type':'identity','scaling': 50}, 
    ]
    parameters=[examples,mask,nref,fems,gamma,wd,wr,network_file,ini,conf,maps,method]
    combinations_figure_5 = list(itertools.product(*parameters))

    return combinations_figure_5
    
def figure6():
    #
    # combination for Figure 6
    #
    examples = ['frog_tongue']
    mask = ['mask.png']
    nref=[0]
    gamma = [0.5]
    wd = [1e-3,1e-1,1e0]
    ini = [0]
    network_file = ['network.png'] 
    conf = ['ONE']
    maps = [
        {'type':'identity','scaling': 50}, 
    ]
    
    parameters=[examples,mask,nref,fems,gamma,wd,wr,network_file,ini,conf,maps,method]
    combinations_figure_6 = list(itertools.product(*parameters))

    return combinations_figure_6


def figure7():
    #
    # combination for Figure 7
    #
    examples = ['frog_tongue/']
    mask = ['mask.png']
    nref=[0]
    gamma = [0.5]
    wd = [1e-2,1e-1,1e0]
    ini = [0]
    network_file = ['network_shifted.png'] 
    conf = ['ONE']
    maps = [
        {'type':'identity','scaling': 50}, 
    ]
    
    parameters=[examples,mask,nref,fems,gamma,wd,wr,network_file,ini,conf,maps,method]
    combinations_figure_7 = list(itertools.product(*parameters))

    return combinations_figure_7

def figure8():
    #
    # combination for Figure 8
    #
    examples = ['frog_tongue/']
    mask = ['mask.png']
    nref=[0]
    gamma = [0.5]
    wd = [1e1,1e3,1e4] # set the discrepancy to zero
    ini = [0]
    network_file = ['mup3.0e+00zero5.0e+02.npy'] 
    conf = ['MASK','ONE']
    maps = [
        {'type':'identity','scaling': 1.0}, 
    ]
    parameters=[examples,mask,nref,fems,gamma,wd,wr,network_file,ini,conf,maps,method]
    combinations_figure_8 = list(itertools.product(*parameters))

    return combinations_figure_8

def run_single_experiment(example, mask, nref,fem,gamma,wd,wr,network_file,ini,conf,tdens2image, method, comm=COMM_WORLD, n_ensemble=1):
    """
    Function running the corrupt and reconstruct experiment
    given a list of parameters describing the problem
    and the method to solve it.
    """

    # set a list of labels 
    labels_problem = labels(nref,fem,gamma,wd,wr,
                network_file,   
                ini,
                conf,
                tdens2image,
                method)
    

    # load input    
    np_source, np_sink, np_network, np_mask = load_input('data/'+example, nref, mask, network_file, comm=comm)
    

    # check if the directory exists
    for directory in ['results',f'result/{example}',f'result/{example}/{mask_name}']:
        if not os.path.exists(directory):
            try:
                os.mkdir(directory)
            except:
                pass # directory already exists

    
    # run the experiment
    ierr = corrupt_and_reconstruct(np_source,
                                   np_sink,
                                   np_network,
                                   np_mask, 
                            fem=fem,
                            gamma=gamma,
                            wd=wd,wr=wr,
                            corrupted_as_initial_guess=ini,
                            confidence=conf,
                            tdens2image=tdens2image,
                            method=method,
                            directory=f'results/{example}/{mask_name}/',
                            labels_problem=labels_problem,
                            comm=comm,
                            n_ensemble=n_ensemble)
    return ierr 



if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Create data for the figures in the manuscript')
    parser.add_argument("-f","--figure", type=str,default='all', help="all(default) or the number of figure (2-8)")
    parser.add_argument("-o","--overwrite", type=int, default=0, help="1 = overwrite  0 = skip (default)")
    parser.add_argument("-n","--np", type=int, default=mp.cpu_count(), help="number of processors (max available=default)")
    args, unknown = parser.parse_known_args()

    
    combinations = []
    if ((args.figure == '2') or (args.figure == 'all')):
        combinations += figure2()
    if ((args.figure == '3') or (args.figure == 'all')):
        combinations += figure3()
    if ((args.figure == '4') or (args.figure == 'all')):
        combinations += figure4()
    if ((args.figure == '5') or (args.figure == 'all')):
        combinations += figure5()
    if ((args.figure == '6') or (args.figure == 'all')):
        combinations += figure6()
    if ((args.figure == '7') or (args.figure == 'all')):
        combinations += figure7()
    if ((args.figure == '8') or (args.figure == 'all')):
        combinations += figure8()

    
    ordered=len(combinations)
    toremove=0
    TODO=[]
    for combination in combinations:
        if is_present(combination) and args.overwrite == 1 : 
            toremove+=1
        else:
            TODO.append(combination)
    
        
    print(f'{args.overwrite=}, {toremove=}, {ordered=}, {len(combinations)=} {len(TODO)=}')


    for combination in TODO:
        labels_problem = labels(*combination[2:])
        example = combination[0]
        mask = combination[1]
        mask_name = os.path.splitext(mask)[0]
        label = '_'.join(labels_problem)
        #if not is_present(combination):
        #    print (label )

    del combinations
    combinations = TODO

    #for comb in combinations:
    #    labels_problem = labels(*comb[2:])
    #    print(f"labels_problem = ",'_'.join(labels_problem))
    #    run_single_experiment(*comb)
    comb = combinations[1]
    labels_problem = labels(*comb[2:])
    print(f"labels_problem = ",'_'.join(labels_problem))
    run_single_experiment(*comb)

    #with mp.Pool(processes = 4) as p:
    #    ierr = p.starmap(run_single_experiment, combinations)
    #    if ierr != 0:
    #        print(f"ierr = {ierr} for {combination}")
