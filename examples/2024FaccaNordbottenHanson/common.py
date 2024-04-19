import itertools
import os


def labels(nref,fem,
           gamma,wd,wr,
           network_file,
           corrupted_as_initial_guess,
           confidence,
           tdens2image, 
           method):
    """
    Return a list of labels that describe the problem
    and the method used to solve it.
    """

    # take filename and remove extension
    label_network = os.path.basename(network_file)
    # remove extension
    label_network = os.path.splitext(label_network)[0]
    label= [
        f'nref{nref}',
        f'fem{fem}',
        f'gamma{gamma:.1e}',
        f'wd{wd:.1e}',
        f'wr{wr:.1e}',
        f'net{label_network}',
        f'ini{corrupted_as_initial_guess:.1e}',
        f'conf{confidence}']
    if tdens2image['type'] == 'identity':
        label.append(f'mu2iidentity')
    elif tdens2image['type'] == 'heat':
        label.append(f"mu2iheat{tdens2image['sigma']:.1e}")
    elif tdens2image['type'] == 'pm':
        label.append(f"mu2ipm{tdens2image['sigma']:.1e}")
    else:
        raise ValueError(f'Unknown tdens2image {tdens2image}')
    label.append(f"scaling{tdens2image['scaling']:.1e}")  
    if method is not None:
        if method == 'tdens_mirror_descent_explicit':
            short_method = 'te'
        elif method == 'tdens_mirror_descent_semi_implicit':
            short_method = 'tsi'
        elif method == 'gfvar_gradient_descent_explicit':
            short_method = 'ge'
        elif method == 'gfvar_gradient_descent_semi_implicit':
            short_method = 'gsi'
        elif method == 'tdens_logarithmic_barrier':
            short_method = 'tlb'
        else:
            raise ValueError(f'Unknown method {method}')
    label.append(f'method{short_method}')
    return label


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
