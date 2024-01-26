import pyvista as pv
import numpy as np
# this is to use latex see https://github.com/pyvista/pyvista/discussions/2928
import vtk 
import sys
import os
import itertools

pv.start_xvfb()

def labels(nref,fem,
           gamma,wd,wr,
           network_file,
           corrupted_as_initial_guess,
           confidence,
           tdens2image,
           tdens2image_scaling, 
           method):
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
        f'ini{corrupted_as_initial_guess:d}',
        f'conf{confidence}']
    if tdens2image['type'] == 'identity':
        label.append(f'mu2iidentity')
    elif tdens2image['type'] == 'heat':
        label.append(f"mu2iheat{tdens2image['sigma']:.1e}")
    elif tdens2image['type'] == 'pm':
        label.append(f"mu2ipm{tdens2image['sigma']:.1e}")
    else:
        raise ValueError(f'Unknown tdens2image {tdens2image}')
    label.append(f'scaling{tdens2image_scaling:.1e}')  
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



def matrix_name(args):
    parameters = []
    tochange = []
    for i,a in enumerate(args):
        if len(a)>1:
            tochange.append(i)    
        parameters.append(a[0])
    labels_problem = labels(*parameters)
    for i in tochange:
        labels_problem[i]='@'
    return labels_problem


def fixed_changing(parameters):
    c = []
    f = []
    nel=0
    for i,a in enumerate(parameters):
        if len(a)>1:
            c.append(i) 
            nel+=1
        else:
            f.append(i)
    if nel > 2:
        raise ValueError('rows or cols must have exactly one element')
    return f,c

def make_matrix(pl,
                directory,
                parameters,
                variable='tdens',
                swap_rows_cols=False,
                width=1000, height=1000,
                network=True,
                mask=True,
):

    scaling = width/1000

    fixed,  changing = fixed_changing(parameters)

    changing_label=changing
    if swap_rows_cols:
        changing = changing[::-1]


    try:
        rows = parameters[changing[0]]
        nrows = len(rows)
        cols = parameters[changing[1]]
        ncols = len(cols)
    except:
        if swap_rows_cols:
            cols = parameters[changing[0]]
            ncols = len(cols)
            rows = []
            nrows = 1
        else:
            rows = parameters[changing[0]]
            nrows = len(rows)
            cols = []
            ncols = 1
        
    
    
    pl.clear_actors()

    
    
    for i in range(nrows):
        for j in range(ncols):        
            parameters_local = parameters.copy()
            try:
                parameters_local[changing[0]] = [rows[i]]
            except:
                parameters_local[changing[0]] = [cols[j]]

            try:
                parameters_local[changing[1]] = [cols[j]]
            except:
                pass
            # flatten list
            parameters_local = [item for sublist in parameters_local for item in sublist]
            #print(parameters_local)
            #print(labels(*parameters_local))
            labels_local = labels(*parameters_local)

            path_vtu = directory + '_'.join(labels(*parameters_local))+'.vtu'

            print(i,j,path_vtu)
        
            pl.subplot(i,j)
            
            vtu = pv.read(path_vtu)

            #
            # network
            #
            if network:
                try:
                    net_data = vtu.get_array("network")
                    net_support= net_data
                    net_support[net_support>1e-16]=1

                except:
                    mask_vtu = (directory 
                            + labels(*parameters_local)[0]
                            +'_network_mask.vtu')
                    mask_vtu = pv.read(mask_vtu)
                    net_data = mask_vtu.get_array("network")
                    net_support= net_data
                    net_support[net_support>1e-16]=1

                pl.add_mesh(mask_vtu,
                            scalars=net_support,
                            cmap="binary",
                            opacity = [0,0.1],
                            show_scalar_bar=False)
                
            #
            # TDENS
            #
            label_row = labels_local[changing_label[0]]
            try:
                label_col = labels_local[changing_label[1]]

                if changing[0] > changing[1]:
                    title = label_col+' '+label_row
                else:   
                    title = label_row+' '+label_col
            except:
                title = label_row
                
            sargs = dict(
                title=title,
                title_font_size=int(40*scaling),
                label_font_size=int(40*scaling),
                shadow=True,
                n_labels=2,
                #italic=True,
                fmt="%.0e",
                font_family="times",
                vertical=False,
                position_x=0.3,
                position_y=0.85,
                width=0.35,
                height=0.1,    
            )

            # this is to make values below min as white/transparent
            r=np.ones(50)
            r[0]=0
            if variable=='tdens':
                clim = [1e-5, 1e-1]
                clim = [1e-8, 1e-4]
            elif variable=='reconstruction':
                clim = None#[0, 1]
            pl.add_mesh(vtu.copy(),
                    scalars=variable, 
                    #cmap="hot_r",
                    cmap="gnuplot2_r",
                    opacity=list(0.89*r),
                    clim = clim,
                    #below_color='white',
                    #above_color='black',
                    show_scalar_bar=True,
                    scalar_bar_args=sargs
            )
            pl.view_xy(render=False)
            pl.add_bounding_box()
            var = vtu.get_array(variable)
            min_var = np.min(var)
            max_var = np.max(var)
            if variable=='tdens':
                msg=rf'${min_var:.1e}\leq \mu \leq {max_var:.1e}$'
            elif variable=='reconstruction':
                msg=rf'${min_var:.1e}\leq '+'\mathcal{I}'+rf'\leq {max_var:.1e}$'
                
            pl.add_text(
                text=msg,
                position=[0.25*width,0.8*height],
                font_size=int(20*scaling),
                color=None,
                font="times",
                shadow=False,
                name=None,
                viewport=False,
                orientation=0.0,
                font_file=None)
        
        
            #
            # MASK
            #
            if mask:
                try:
                    contours = vtu.contour(isosurfaces=2, scalars='mask_countour')
                except:
                    contours = mask_vtu.contour(isosurfaces=2, scalars='mask_countour')
                pl.add_mesh(contours, line_width=2, color="black",show_scalar_bar=False)

            #print(vtu.array_names)
            contour_image = False
            if contour_image:
                contour_tdens = vtu.contour(
                    isosurfaces=[1e-6,1e-3], scalars='reconstruction_countour')
                pl.add_mesh(contour_tdens, line_width=2,
                            cmap="bwr",
                            #color="blue",
                            show_scalar_bar=False)

        
            pl.camera_position = 'xy'
            pl.zoom_camera(1.4)

            #pl.window_size = (4000, 1000)
            #pl.link_views
            
            #pl.show()
            #pl.show(auto_close=False)
            #print(pl.camera_position)
            #if :
    return pl


directory="./mask_large/"
directory="./frog_thk/mask02/"

for x in ['ONE','MASK','CORRUPTED']:
    print(f'{x=}')
    nref=[0]#,2,3]
    fems = ['DG0DG0']
    #gamma = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #gamma = [0.8, 0.2, 0.5]#, 0.8]
    gamma = [0.5]
    wd = [1e5]
    wr = [0,1e-5]#,5e-4,1e-3,1e-2]#,1e-3,1e-2]#1e-3]#[1e-3]#, 1e-3, 1e-2]
    network_file = ['mup3.0e+00zero1.0e+01.npy']
    ini = [0,1]
    conf = ['MASK']
    maps = [
        #{'type':'identity'}, 
        #{'type':'pm', 'sigma': 1e-3, 'exponent_m': 2.0},
        #{'type':'pm', 'sigma': 1e-2, 'exponent_m': 2.0},
        #{'type':'pm', 'sigma': 1e-1, 'exponent_m': 2.0},
        {'type':'pm', 'sigma': 0.0005, 'exponent_m': 2.0},
        #{'type':'pm', 'sigma': 5e-1, 'exponent_m': 2.0},
        #{'type':'heat', 'sigma': 1e-4},
        #{'type':'heat', 'sigma': 5e-4},
        #{'type':'heat', 'sigma': 1e-3},
        #{'type':'pm', 'sigma': 1e-4, 'exponent_m': 2.0},

    ]
    tdens2image_scaling = [1]#[25]#2.5e+01]
    method = [
        #'tdens_mirror_descent_explicit',
        #'tdens_mirror_descent_semi_implicit',
        #'gfvar_gradient_descent_explicit',
        'gfvar_gradient_descent_semi_implicit',
    ]


    # set the order to be passed to labels function
    parameters=[nref,fems,gamma,wd,wr,network_file,ini,conf,maps,tdens2image_scaling,method]


    swap_rows_cols = False
    interactive = False
    width=500
    height=500
    variable='reconstruction'
    variable='tdens'
    plot_mask=False#
    plot_mask=True
    plot_network=False
    plot_network=True


    # set
    fixed,  changing = fixed_changing(parameters)
    rows = parameters[changing[0]]
    nrows = len(rows)
    try:
        cols = parameters[changing[1]]
        ncols = len(cols)
    except:
        cols = []
        ncols = 1

    shape=(nrows,ncols)
    window_size=(ncols*width,nrows*height)
    if swap_rows_cols:
        shape=(ncols,nrows)
        window_size=(nrows*width,ncols*height)

    pl = pv.Plotter(shape=shape,
                    border=True,
                    window_size=window_size,
                    off_screen=not interactive,
    )

    pl = make_matrix(
        pl,
        directory,
        parameters,
        variable=variable,
        swap_rows_cols=swap_rows_cols,
        width=500, height=500,
        network=plot_network,
        mask=plot_mask)


    class SetVisibilityCallback:
        """Helper callback to keep a reference to the actor being modified."""

        def __init__(self, pl):
            print('init')
            self.pl = pl

        def __call__(self, state):
            print('chaning', state)
            if state:
                parameters[5]=[1]
            else:
                parameters[5]=[0]

            print(parameters)
            self.pl=make_matrix(pl,
                                directory,
                                parameters,
                                variable='tdens',
                                swap_rows_cols=False,
                                width=500, height=500)



    if interactive:
        callback = SetVisibilityCallback(pl)
        pl.add_checkbox_button_widget(
            callback,
            value=True,
            position=(5.0, 1.0),
            size=10,
            border_size=1,
            color_on='blue',
            color_off='grey',
            background_color='grey',
        )
        pl.show()
    else:
        pdf_filename = directory+'matrix_'+'_'.join(matrix_name(parameters))+f'_{variable}.pdf'
        print('saved in')
        print(pdf_filename)
        print(f'{x=}')
#        pl.screenshot(pdf_filename)
        pl.save_graphic(pdf_filename)
        print(f'{x=}')
