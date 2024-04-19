"""
Script running the examples in parallel.

Example of usage:
    python run_parallel.py --help (show the help)
    python run_parallel.py (run all the examples in parallel)
"""

import argparse
from corrupt_and_reconstruct import labels, load_input, corrupt_and_reconstruct
import multiprocessing as mp
import os
from common import figure2, figure3, figure4, figure5, figure6, figure7, figure8
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

    figures = []
    combinations = []
    if ((args.figure == '2') or (args.figure == 'all')):
        figures += {'2': figure2()}
        combinations += figure2()
    if ((args.figure == '3') or (args.figure == 'all')):
        figures += {'3': figure3()}
        combinations += figure3()
    if ((args.figure == '4') or (args.figure == 'all')):
        figures += {'4': figure4()}
        combinations += figure4()
    if ((args.figure == '5') or (args.figure == 'all')):
        figures += {'5': figure5()}
        combinations += figure5()
    if ((args.figure == '6') or (args.figure == 'all')):
        figures += {'6': figure6()}
        combinations += figure6()
    if ((args.figure == '7') or (args.figure == 'all')):
        figures += {'7': figure7()}
        combinations += figure7()
    if ((args.figure == '8') or (args.figure == 'all')):
        figures += {'8': figure8()}
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

    with mp.Pool(processes = args.np) as p:
       ierr = p.starmap(run_single_experiment, combinations)
       if ierr != 0:
           print(f"ierr = {ierr} for {combination}")