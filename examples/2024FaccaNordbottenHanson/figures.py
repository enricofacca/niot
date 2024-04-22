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

def get_vtu(pathdotvtu):
    try:
        os.path.exists(pathdotvtu)
        print(pathdotvtu)
        vtu = pv.read(pathdotvtu)
        return vtu
    except:
        # get name of the file without directory
        filename = os.path.basename(pathdotvtu)
        dirname = os.path.dirname(pathdotvtu)   
        # remove last 5 characters with "_0.vtu"
        filename_cut = filename[:-6]

        print(filename)
        print(filename_cut)

        pathdotvtu = os.path.join(dirname,filename_cut,filename)
        print(pathdotvtu)
        vtu = pv.read(pathdotvtu)
        return vtu
        
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
        raise ValueError('Only combination of 1 or 2 parameters can be assembled ')
    return f,c

def single_plot(pl,parameters,
                scaling=1.0,
                zoom=1.0):
    
    example = parameters[0]
    mask = parameters[1]
    #remove extension
    mask_name = os.path.splitext(mask)[0]
    

    parameters_local = parameters[2:]

    directory = './results/'+example+'/'+mask_name+'/'
    try:
        path_vtu = os.path.abspath(directory
                +'_'.join(labels(*parameters_local))+'_0.vtu')
        vtu = pv.read(path_vtu)
        print(path_vtu)
    except:
        path_vtu = os.path.abspath(directory
                + '_'.join(labels(*parameters_local))+'/'
                +'_'.join(labels(*parameters_local))+'_0.vtu')
        vtu = pv.read(path_vtu)
        print(path_vtu)
      
    #
    # TDENS
    #
    bar_dict = {
        "position_x": 0.83,
        "position_y": 0.05,
        "width": 0.15,
        "height" : 0.9}

    sargs = dict(
        title="",
        title_font_size=int(60*scaling),
        #color="white",
        label_font_size=int(60*scaling),
        shadow=True,
        n_labels=3,
        #italic=True,
        fmt="%.0e",
        font_family="times",
        vertical=True,
        **bar_dict
    )

    # this is to make values below min as white/transparent
    r=np.ones(50)
    r[0]=0

    var = vtu.get_array("tdens")
    clim = [1e-5,1e-2] 
    
    pl.add_mesh(vtu.copy(),
            scalars=var, 
            #cmap="hot_r",
            cmap="gnuplot2_r",
            opacity=list(0.89*r),
            clim = clim,
            #below_color='white',
            #above_color='black',
            show_scalar_bar=True,
            scalar_bar_args=sargs,
                log_scale=False,#True,
    )
    pl.view_xy(render=False)
    pl.add_bounding_box()    
    pl.camera_position = 'xy'
    pl.zoom_camera(zoom)

    return pl
            

def plot_figure2(parameters,dir_vtu="./results/"):
    
    width=700
    height=500
    zoom=1.3
    scaling=0.5

    variable='tdens'
    window_size=(width,height)

    pl = pv.Plotter(border=False,#True,
                    window_size=window_size,
                    off_screen=False,
    )
    
    example = parameters[0]
    mask = parameters[1]
    #remove extension
    mask_name = os.path.splitext(mask)[0]
    

    parameters_local = parameters[2:]

    directory = dir_vtu+'/'+example+'/'+mask_name+'/'
    try:
        path_vtu = os.path.abspath(directory
                +'_'.join(labels(*parameters_local))+'_0.vtu')
        vtu = pv.read(path_vtu)
        print(path_vtu)
    except:
        path_vtu = os.path.abspath(directory
                + '_'.join(labels(*parameters_local))+'/'
                +'_'.join(labels(*parameters_local))+'_0.vtu')
        vtu = pv.read(path_vtu)
        print(path_vtu)
      
    #
    # TDENS
    #
    bar_dict = {
        "position_x": 0.83,
        "position_y": 0.05,
        "width": 0.15,
        "height" : 0.9}

    sargs = dict(
        title="",
        title_font_size=int(60*scaling),
        #color="white",
        label_font_size=int(60*scaling),
        shadow=True,
        n_labels=3,
        #italic=True,
        fmt="%.0e",
        font_family="times",
        vertical=True,
        **bar_dict
    )

    # this is to make values below min as white/transparent
    r=np.ones(50)
    r[0]=0

    var = vtu.get_array("tdens")
    clim = [1e-5,1e-2] 
    
    pl.add_mesh(vtu.copy(),
            scalars=var, 
            #cmap="hot_r",
            cmap="gnuplot2_r",
            opacity=list(0.89*r),
            clim = clim,
            #below_color='white',
            #above_color='black',
            show_scalar_bar=True,
            scalar_bar_args=sargs,
                log_scale=False,#True,
    )
    pl.view_xy(render=False)
    pl.add_bounding_box()    
    pl.camera_position = 'xy'
    pl.zoom_camera(zoom)

    pdf_filename = directory+'matrix_'+'_'.join(labels(*parameters[2:]))+f'_{variable}.pdf'
    print('saved in')
    print(pdf_filename)
    pl.save_graphic(pdf_filename,painter=False)

def plot_figure3(parameters, dir_vtu="./results/"):
    width=500
    height=600
    zoom=1.3
    scaling=0.5
    
    variable='tdens'
    mask=True
    network = 1 #0=no, 1=remove were the mask is applied, 2= show everywhere
    sink = False 
    bounds = False

    
    example = parameters[0]
    mask = parameters[1]
    #remove extension
    mask_name = os.path.splitext(mask)[0]
    

    parameters_local = parameters[2:]
    
    window_size=(width,height)
    pl = pv.Plotter(border=False,
            window_size=window_size,
            off_screen=True,
    )

    directory = dir_vtu+example+'/'+mask_name+'/'
    try:
        path_vtu = os.path.abspath(directory,
                +'_'.join(labels(*parameters_local))+'_0.vtu')
        vtu = pv.read(path_vtu)
        print(path_vtu)
    except:
        path_vtu = os.path.abspath(directory
                + '_'.join(labels(*parameters_local))+'/'
                +'_'.join(labels(*parameters_local))+'_0.vtu')
        vtu = pv.read(path_vtu)
        print(path_vtu)

    #
    # network
    #
    if network>=1:
        try:
            net_data = vtu.get_array("network")
            net_support= net_data
        except:
            try:
                mask_vtu_file = (directory 
                    + labels(*parameters_local)[0]
                    +'_'+ labels(*parameters_local)[5]
                    +'_network_mask_0.vtu')
                mask_vtu = pv.read(mask_vtu_file)                        
            except:
                mask_vtu_file = (directory
                            + labels(*parameters_local)[0]
                    +'_'+ labels(*parameters_local)[5]
                            +'_network_mask/'
                    + labels(*parameters_local)[0]
                    +'_'+ labels(*parameters_local)[5]
                    +'_network_mask_0.vtu')
                mask_vtu = pv.read(mask_vtu_file)

        
        print(mask_vtu_file)
        net_data = mask_vtu.get_array("network")
        net_support= net_data
        thr = 1e-3
        net_support[net_support>=thr]=1
        net_support[net_support<thr]=0
        if network == 1:
            mask_data = mask_vtu.get_array("mask")
            net_support *= (1.0-mask_data)

        pl.add_mesh(mask_vtu,
                    scalars=net_support,
                    cmap="binary",
                    opacity = [0,0.2],
                    show_scalar_bar=False)
        
    #
    # TDENS
    #
    
    bar_dict = {
        "position_x": 0.1,
        "position_y": 0.00,
        "width": 0.8,
        "height" : 0.15}
    
    sargs = dict(
        title="",#(i*10+j)*" ",#title, trick to change lenght
        title_font_size=int(60*scaling),
        #color="white",
        label_font_size=int(60*scaling),
        shadow=True,
        n_labels=3,
        #italic=True,
        fmt="%.0e",
        font_family="times",
        vertical=False,
        **bar_dict
    )

    # this is to make values below min as white/transparent
    r=np.ones(50)
    r[0]=0

    plot_variable = True
    if plot_variable:
        var = vtu.get_array(variable)
        #var = np.log10(vtu.get_array(variable))
        if variable=='tdens':
            clim = [1e-5,1e-2] 
        elif variable=='reconstruction':
            clim = [1e-8,1e-1]

        pl.add_mesh(vtu.copy(),
                scalars=var, 
                #cmap="hot_r",
                cmap="gnuplot2_r",
                opacity=list(0.89*r),
                clim = clim,
                #below_color='white',
                #above_color='black',
                show_scalar_bar=True,
                scalar_bar_args=sargs,
                    log_scale=False,#True,
        )
        pl.view_xy(render=False)
        pl.add_bounding_box()
        var = vtu.get_array(variable)
        min_var = np.min(var)
        max_var = np.max(var)

    if bounds:
        if variable=='tdens':
            msg=rf'${min_var:.1e}\leq \mu \leq {max_var:.1e}$'
        elif variable=='reconstruction':
            msg=rf'${min_var:.1e}\leq '+'\mathcal{I}'+rf'\leq {max_var:.1e}$'

        pl.add_text(
                text=msg,
                position=[0.31*width,0.87*height],
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
        pl.add_mesh(contours, line_width=2, color="red",show_scalar_bar=False)


    if sink:
        try:
            file_btp = (directory 
                        + labels(*parameters_local)[0]
                        +'_btp_0.vtu')
            print(file_btp)
            btp_vtu = pv.read(file_btp)
        except:
            file_btp = (directory
                        + labels(*parameters_local)[0]
                        +'_btp/'
                        + labels(*parameters_local)[0]
                        +'_btp_0.vtu')
            print(file_btp)
            btp_vtu = pv.read(file_btp)
            
        contours = btp_vtu.contour(isosurfaces=2, scalars='sink_countour')
        pl.add_mesh(contours, line_width=2, color="black",show_scalar_bar=False)

    contour_image = parameters_local[-2]['type'] == "pm"
    if contour_image:
        contour_tdens = vtu.contour(
            isosurfaces=[1e-4], scalars='reconstruction_countour')
        pl.add_mesh(contour_tdens, line_width=2,
                    #cmap="bwr",
                    color="blue",
                    show_scalar_bar=False)
        
    pl.camera_position = 'xy'
    pl.zoom_camera(zoom)

    pdf_filename = directory+'matrix_'+'_'.join(labels(*parameters[2:]))+f'_{variable}.pdf'
    print(f'rendering:{pdf_filename})')
    pl.save_graphic(pdf_filename,painter=False)
    print(f'done:\n {pdf_filename})')


def plot_figure4(parameters, dir_vtu="./results/"):
    width=500
    height=600
    zoom=1.3
    scaling=0.5
    
    variable='tdens'
    mask=True
    network = 1 #0=no, 1=remove were the mask is applied, 2= show everywhere
    sink = False 
    bounds = False

    
    example = parameters[0]
    mask = parameters[1]
    #remove extension
    mask_name = os.path.splitext(mask)[0]
    

    parameters_local = parameters[2:]
    
    window_size=(width,height)
    pl = pv.Plotter(border=False,
            window_size=window_size,
            off_screen=True,
    )

    directory = dir_vtu+example+'/'+mask_name+'/'
    try:
        path_vtu = os.path.abspath(directory,
                +'_'.join(labels(*parameters_local))+'_0.vtu')
        vtu = pv.read(path_vtu)
        print(path_vtu)
    except:
        path_vtu = os.path.abspath(directory
                + '_'.join(labels(*parameters_local))+'/'
                +'_'.join(labels(*parameters_local))+'_0.vtu')
        vtu = pv.read(path_vtu)
        print(path_vtu)

    #
    # network
    #
    if network>=1:
        try:
            net_data = vtu.get_array("network")
            net_support= net_data
        except:
            try:
                mask_vtu_file = (directory 
                    + labels(*parameters_local)[0]
                    +'_'+ labels(*parameters_local)[5]
                    +'_network_mask_0.vtu')
                mask_vtu = pv.read(mask_vtu_file)                        
            except:
                mask_vtu_file = (directory
                            + labels(*parameters_local)[0]
                    +'_'+ labels(*parameters_local)[5]
                            +'_network_mask/'
                    + labels(*parameters_local)[0]
                    +'_'+ labels(*parameters_local)[5]
                    +'_network_mask_0.vtu')
                mask_vtu = pv.read(mask_vtu_file)

        
        net_data = mask_vtu.get_array("network")
        net_support= net_data
        thr = 1e-3
        net_support[net_support>=thr]=1
        net_support[net_support<thr]=0
        if network == 1:
            mask_data = mask_vtu.get_array("mask")
            net_support *= (1.0-mask_data)

        pl.add_mesh(mask_vtu,
                    scalars=net_support,
                    cmap="binary",
                    opacity = [0,0.2],
                    show_scalar_bar=False)
        
    #
    # TDENS
    #
    
    bar_dict = {
        "position_x": 0.1,
        "position_y": 0.00,
        "width": 0.8,
        "height" : 0.15}
    
    sargs = dict(
        title="",#(i*10+j)*" ",#title, trick to change lenght
        title_font_size=int(60*scaling),
        #color="white",
        label_font_size=int(60*scaling),
        shadow=True,
        n_labels=3,
        #italic=True,
        fmt="%.0e",
        font_family="times",
        vertical=False,
        **bar_dict
    )

    # this is to make values below min as white/transparent
    r=np.ones(50)
    r[0]=0

    plot_variable = True
    if plot_variable:
        var = vtu.get_array(variable)
        #var = np.log10(vtu.get_array(variable))
        if variable=='tdens':
            clim = [1e-5,1e-2] 
        elif variable=='reconstruction':
            clim = [1e-8,1e-1]

        pl.add_mesh(vtu.copy(),
                scalars=var, 
                #cmap="hot_r",
                cmap="gnuplot2_r",
                opacity=list(0.89*r),
                clim = clim,
                #below_color='white',
                #above_color='black',
                show_scalar_bar=True,
                scalar_bar_args=sargs,
                    log_scale=False,#True,
        )
        pl.view_xy(render=False)
        pl.add_bounding_box()
        var = vtu.get_array(variable)
        min_var = np.min(var)
        max_var = np.max(var)

    if bounds:
        if variable=='tdens':
            msg=rf'${min_var:.1e}\leq \mu \leq {max_var:.1e}$'
        elif variable=='reconstruction':
            msg=rf'${min_var:.1e}\leq '+'\mathcal{I}'+rf'\leq {max_var:.1e}$'

        pl.add_text(
                text=msg,
                position=[0.31*width,0.87*height],
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
        pl.add_mesh(contours, line_width=2, color="red",show_scalar_bar=False)

        
    pl.camera_position = 'xy'
    pl.zoom_camera(zoom)

    pdf_filename = directory+'matrix_'+'_'.join(labels(*parameters[2:]))+f'_{variable}.pdf'
    print(f'rendering:{pdf_filename})')
    pl.save_graphic(pdf_filename,painter=False)
    print(f'done:\n {pdf_filename})')

def plot_figure5(directory):
    plot_mask=True
    plot_network =2 #0=no, 1=remove were the mask is applied, 2= show everywhere
    plot_sink = True
    plot_bounds = False


    
    width=int(1000*0.5)
    height=int(1550*0.5)
    zoom=1.62
    
    window_size=(width,height)
    pl = pv.Plotter(border=False,
                window_size=window_size,
                off_screen=True)

    mask_vtu_file = (directory 
                     +'nref0_netnetwork'
                     +'_network_mask_0.vtu')
    mask_vtu = get_vtu(mask_vtu_file)



    scaling = width/1000

    bar_dict = {
        "position_x": 0.1,
        "position_y": 0.00,
        "width": 0.8,
        "height" : 0.15}
    
    net_data = mask_vtu.get_array("network")
    net_support= net_data
    binary = False
    network = 2
    if binary: 
        thr = 1e-3
        net_support[net_support>=thr]=1
        net_support[net_support<thr]=0
        network = 2

    r=np.ones(50)
    r[0]=0

    sargs = dict(
        title="",#(i*10+j)*" ",#title, trick to change lenght
        title_font_size=int(60*scaling),
        #color="white",
        label_font_size=int(60*scaling),
        shadow=True,
        n_labels=3,
        #italic=True,
        fmt="%.0e",
        font_family="times",
        vertical=False,
        **bar_dict
    )

    clim = [1e-3,2e-2] 
    pl.add_mesh(mask_vtu,
            scalars=net_support, 
            #cmap="hot_r",
            cmap="gnuplot2_r",
            opacity=list(0.89*r),
            clim = clim,
            #below_color='white',
            #above_color='black',
            show_scalar_bar=False,#True,
            scalar_bar_args=sargs,
            log_scale=False
    )


    boundary_vtu_file = ('./medium/boundary_inside2/boundary_inside2_0.vtu')
    boundary_vtu = pv.read(boundary_vtu_file)

    b_contours = boundary_vtu.contour(isosurfaces=2, scalars='img_contour')
    pl.add_mesh(b_contours, line_width=2, color="black",show_scalar_bar=False)


    

    if plot_sink:
        file_btp = (directory 
                    + 'nref0'
                    +'_btp_0.vtu')
        btp_vtu = get_vtu(file_btp)
        
    contours = btp_vtu.contour(isosurfaces=2, scalars='sink_countour')
    pl.add_mesh(contours, line_width=2, color="black",show_scalar_bar=False)

    letters = True#False
    if letters:

        contours = mask_vtu.contour(isosurfaces=2, scalars='mask_countour')
        pl.add_mesh(contours, line_width=2, color="red",show_scalar_bar=False)

        correction = 0.95
        letter_size=30
        pl.add_text(
            text="a",
            #position=[0.3*width,0.16*correction*height],
            position=[0.3*width,0.22*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)

        pl.add_text(
            text="b",
            #position=[0.23*width,0.3*correction*height],
            position=[0.23*width,0.31*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)

        pl.add_text(
            text="c",
            #position=[0.36*width,0.66*correction*height],
            position=[0.36*width,0.62*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)
        pl.add_text(
            text="d",
            #position=[0.64*width,0.56*height],
            position=[0.64*width,0.55*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)
        pl.add_text(
            text="e",
            #position=[0.7*width,0.72*height],
            position=[0.7*width,0.66*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)
        pl.add_text(
            text="f",
            #position=[0.85*width,0.35*correction*height],
            position=[0.85*width,0.41*correction*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)
        pl.add_text(
            text="g",
            #position=[0.65*width,0.12*correction*height],
            position=[0.65*width,0.19*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)
    


    pl.camera_position = 'xy'
    pl.zoom_camera(zoom)
    pdf_filename=directory+"medium_shifted.pdf"
    pl.save_graphic(pdf_filename,painter=False)
    print(pdf_filename)


    width=int(1000*0.5)
    height=int(1550*0.5)
    zoom=1.62
    
    plot_mask=True
    plot_network =1 #0=no, 1=remove were the mask is applied, 2= show everywhere
    plot_sink = True
    plot_bounds = False


    window_size=(width,height)
    pl = pv.Plotter(border=False,
                    window_size=window_size,
                    off_screen=True)
    
    try:
        mask_vtu_file = (directory 
                         +'nref0_netnetwork_shifted_inside2'
                         +'_network_mask_0.vtu')
        mask_vtu = pv.read(mask_vtu_file)                        
    except:
        mask_vtu_file = (directory
                         + 'nref0_netnetwork_shifted_inside2'
                         +'_network_mask/'
                         +'nref0_netnetwork_shifted_inside2'
                         +'_network_mask_0.vtu')
        mask_vtu = pv.read(mask_vtu_file)



    scaling = width/1000

    bar_dict = {
        "position_x": 0.1,
        "position_y": 0.00,
        "width": 0.8,
        "height" : 0.15}
    
    print(mask_vtu_file)
    net_data = mask_vtu.get_array("network")
    net_support= net_data
    binary = False
    network = plot_network
    if binary: 
        thr = 1e-3
        net_support[net_support>=thr]=1
        net_support[net_support<thr]=0
        
    if network == 1:
        mask_data = mask_vtu.get_array("mask")
        net_support *= (1.0-mask_data)



    r=np.ones(50)
    r[0]=0

    sargs = dict(
        title="",#(i*10+j)*" ",#title, trick to change lenght
        title_font_size=int(60*scaling),
        #color="white",
        label_font_size=int(60*scaling),
        shadow=True,
        n_labels=3,
        #italic=True,
        fmt="%.0e",
        font_family="times",
        vertical=False,
        **bar_dict
    )

    clim = [1e-3,2e-2] 
    pl.add_mesh(mask_vtu,
                scalars=net_support, 
                #cmap="hot_r",
                cmap="gnuplot2_r",
                opacity=list(0.89*r),
                clim = clim,
                #below_color='white',
                #above_color='black',
                show_scalar_bar=False,#True,
                scalar_bar_args=sargs,
            log_scale=False
    )


    try:
        boundary_vtu_file = ('./medium/' 
                             +'boundary_inside2.vtu')
        boundary_vtu = pv.read(boundary_vtu_file)
    except:
        boundary_vtu_file = ('./medium/'
                             + 'boundary_inside2/'
                             +'boundary_inside2_0.vtu')
        boundary_vtu = pv.read(boundary_vtu_file)

    b_contours = boundary_vtu.contour(isosurfaces=2, scalars='img_contour')
    pl.add_mesh(b_contours, line_width=2, color="black",show_scalar_bar=False)
    



    if plot_sink:
        try:
            file_btp = (directory 
                        + 'nref0'
                        +'_btp_0.vtu')
            print(file_btp)
            btp_vtu = pv.read(file_btp)
        except:
            file_btp = (directory
                            + 'nref0'
                            +'_btp/'
                            + 'nref0'
                            +'_btp_0.vtu')
            print(file_btp)
            btp_vtu = pv.read(file_btp)

        contours = btp_vtu.contour(isosurfaces=2, scalars='sink_countour')
        pl.add_mesh(contours, line_width=2, color="black",show_scalar_bar=False)

    letters = True#False
    if letters:

        contours = mask_vtu.contour(isosurfaces=2, scalars='mask_countour')
        pl.add_mesh(contours, line_width=2, color="red",show_scalar_bar=False)

        correction = 0.95
        letter_size=30
        pl.add_text(
            text="a",
            #position=[0.3*width,0.16*correction*height],
            position=[0.3*width,0.22*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)

        pl.add_text(
                    text="b",
                    #position=[0.23*width,0.3*correction*height],
                    position=[0.23*width,0.31*height],
                    font_size=int(letter_size*scaling),
                    color=None,
                    font="times",
                    shadow=False,
                    name=None,
                    viewport=False,
                    orientation=0.0,
                    font_file=None)

        pl.add_text(
                    text="c",
                    #position=[0.36*width,0.66*correction*height],
                    position=[0.36*width,0.62*height],
                    font_size=int(letter_size*scaling),
                    color=None,
                    font="times",
                    shadow=False,
                    name=None,
                    viewport=False,
                    orientation=0.0,
                    font_file=None)
        pl.add_text(
                    text="d",
                    #position=[0.64*width,0.56*height],
                    position=[0.64*width,0.55*height],
                    font_size=int(letter_size*scaling),
                    color=None,
                    font="times",
                    shadow=False,
                    name=None,
                    viewport=False,
                    orientation=0.0,
                    font_file=None)
        pl.add_text(
                    text="e",
                    #position=[0.7*width,0.72*height],
                    position=[0.7*width,0.66*height],
                    font_size=int(letter_size*scaling),
                    color=None,
                    font="times",
                    shadow=False,
                    name=None,
                    viewport=False,
                    orientation=0.0,
                    font_file=None)
        pl.add_text(
                    text="f",
                    #position=[0.85*width,0.35*correction*height],
                    position=[0.85*width,0.41*correction*height],
                    font_size=int(letter_size*scaling),
                    color=None,
                    font="times",
                    shadow=False,
                    name=None,
                    viewport=False,
                    orientation=0.0,
                    font_file=None)
        pl.add_text(
                    text="g",
                    #position=[0.65*width,0.12*correction*height],
                    position=[0.65*width,0.19*height],
                    font_size=int(letter_size*scaling),
                    color=None,
                    font="times",
                    shadow=False,
                    name=None,
                    viewport=False,
                    orientation=0.0,
                    font_file=None)



    pl.camera_position = 'xy'
    pl.zoom_camera(zoom)
    pdf_filename=directory+"medium_shifted.pdf"
    print('saved in')
    print(pdf_filename)
    pl.save_graphic(pdf_filename,painter=False)


    #
    # combination for Figure 5 b
    #
    examples = ['medium']#'frog_tongue/']
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
    method = [
    'tdens_mirror_descent_explicit',
    ]
    ps=[examples,mask,[0],['DG0DG0'],gamma,wd,[0.0],network_file,ini,conf,maps,method]
    parameters = list(itertools.product(*ps))[0]

    
    width = int(1000*0.5)
    height = int(1550*0.5)
    zoom = 1.62
    scaling = 0.5
    
    variable='tdens'
    mask=False#True
    network = 2 #0=no, 1=remove were the mask is applied, 2= show everywhere
    sink = True
    

    # example labels
    example = parameters[0]
    mask = parameters[1]
    #remove extension
    
    mask_name = os.path.splitext(mask)[0]

    parameters_local = parameters[2:]

    window_size=(width,height)
    pl = pv.Plotter(border=False,#True,
                    window_size=window_size,
                    off_screen=False)


    # result file
    filename = os.path.join(directory,
                            '_'.join(labels(*parameters_local))+'_0.vtu')
    
    vtu = get_vtu(filename)
    
    #
    # network
    #
    if network>=1:
        mask_vtu_file = (directory 
                    + labels(*parameters_local)[0]
                    +'_'+ labels(*parameters_local)[5]
                    +'_network_mask_0.vtu')
        mask_vtu = get_vtu(mask_vtu_file)
        
        net_data = mask_vtu.get_array("network")
        net_support= net_data
        thr = 1e-3
        net_support[net_support>=thr]=1
        net_support[net_support<thr]=0
        if network == 1:
            mask_data = mask_vtu.get_array("mask")
            net_support *= (1.0-mask_data)

        pl.add_mesh(mask_vtu,
                    scalars=net_support,
                    cmap="binary",
                    opacity = [0,0.2],
                    show_scalar_bar=False)
        
    #
    # TDENS
    #
    bar_dict = {
        "position_x": 0.1,
        "position_y": 0.00,
        "width": 0.8,
        "height" : 0.15}
    
    sargs = dict(
        title="",#(i*10+j)*" ",#title, trick to change lenght
        title_font_size=int(60*scaling),
        #color="white",
        label_font_size=int(60*scaling),
        shadow=True,
        n_labels=3,
        #italic=True,
        fmt="%.0e",
        font_family="times",
        vertical=False,
        **bar_dict
    )

    # this is to make values below min as white/transparent
    r=np.ones(50)
    r[0]=0

    var = vtu.get_array(variable)
    clim = [1e-6,5e-2] 
    
    pl.add_mesh(vtu.copy(),
            scalars=var, 
            #cmap="hot_r",
            cmap="gnuplot2_r",
            opacity=list(0.89*r),
            clim = clim,
            #below_color='white',
            #above_color='black',
            show_scalar_bar=True,
            scalar_bar_args=sargs,
            log_scale=True,
    )
            
                
    pl.camera_position = 'xy'
    pl.zoom_camera(zoom)

    pdf_filename = directory+'matrix_'+'_'.join(labels(*parameters[2:]))+f'_{variable}.pdf'
    print(f'rendering:{pdf_filename})')
    pl.save_graphic(pdf_filename,painter=False)
    print(f'done:\n {pdf_filename})')

    
    
def plot_figure678(parameters,dir_vtu="./results/"):

    width = int(1000*0.5)
    height = int(1550*0.5)
    zoom = 1.62
    scaling = 0.5
    
    variable='tdens'
    mask=False#True
    network = 2 #0=no, 1=remove were the mask is applied, 2= show everywhere
    sink = True
    

    # example labels
    example = parameters[0]
    mask = parameters[1]
    #remove extension
    mask_name = os.path.splitext(mask)[0]
    directory = dir_vtu + example+'/'+mask_name+'/'

    parameters_local = parameters[2:]

    window_size=(width,height)
    pl = pv.Plotter(border=False,#True,
                    window_size=window_size,
                    off_screen=False)


    # result file
    filename = os.path.join(directory,
                            '_'.join(labels(*parameters_local))+'_0.vtu')
    
    vtu = get_vtu(filename)
    
    #
    # network
    #
    if network>=1:
        mask_vtu_file = (directory 
                    + labels(*parameters_local)[0]
                    +'_'+ labels(*parameters_local)[5]
                    +'_network_mask_0.vtu')
        mask_vtu = get_vtu(mask_vtu_file)
        
        net_data = mask_vtu.get_array("network")
        net_support= net_data
        thr = 1e-3
        net_support[net_support>=thr]=1
        net_support[net_support<thr]=0
        if network == 1:
            mask_data = mask_vtu.get_array("mask")
            net_support *= (1.0-mask_data)

        pl.add_mesh(mask_vtu,
                    scalars=net_support,
                    cmap="binary",
                    opacity = [0,0.2],
                    show_scalar_bar=False)
        
    #
    # TDENS
    #
    bar_dict = {
        "position_x": 0.1,
        "position_y": 0.00,
        "width": 0.8,
        "height" : 0.15}
    
    sargs = dict(
        title="",#(i*10+j)*" ",#title, trick to change lenght
        title_font_size=int(60*scaling),
        #color="white",
        label_font_size=int(60*scaling),
        shadow=True,
        n_labels=3,
        #italic=True,
        fmt="%.0e",
        font_family="times",
        vertical=False,
        **bar_dict
    )

    # this is to make values below min as white/transparent
    r=np.ones(50)
    r[0]=0

    var = vtu.get_array(variable)
    clim = [1e-6,5e-2] 
    
    pl.add_mesh(vtu.copy(),
            scalars=var, 
            #cmap="hot_r",
            cmap="gnuplot2_r",
            opacity=list(0.89*r),
            clim = clim,
            #below_color='white',
            #above_color='black',
            show_scalar_bar=True,
            scalar_bar_args=sargs,
            log_scale=True,
    )
    
    #
    # MASK
    #
    if mask:
        contours = mask_vtu.contour(isosurfaces=2, scalars='mask_countour')
        pl.add_mesh(contours, line_width=2, color="red",show_scalar_bar=False)


    if sink:
        file_btp = (directory 
                        + labels(*parameters_local)[0]
                        +'_btp_0.vtu')
        btp_vtu = get_vtu(file_btp)

        contours = btp_vtu.contour(isosurfaces=2, scalars='sink_countour')
        pl.add_mesh(contours, line_width=2, color="black",show_scalar_bar=False)

    letters = True
    if letters:
        correction = 0.95
        letter_size=30
        pl.add_text(
            text="a",
            #position=[0.3*width,0.16*correction*height],
            position=[0.3*width,0.22*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)

        pl.add_text(
            text="b",
            #position=[0.23*width,0.3*correction*height],
            position=[0.23*width,0.31*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)

        pl.add_text(
            text="c",
            #position=[0.36*width,0.66*correction*height],
            position=[0.36*width,0.62*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)
        pl.add_text(
            text="d",
            #position=[0.64*width,0.56*height],
            position=[0.64*width,0.55*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)
        pl.add_text(
            text="e",
            #position=[0.7*width,0.72*height],
            position=[0.7*width,0.66*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)
        pl.add_text(
            text="f",
            #position=[0.85*width,0.35*correction*height],
            position=[0.85*width,0.41*correction*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)
        pl.add_text(
            text="g",
            #position=[0.65*width,0.12*correction*height],
            position=[0.65*width,0.19*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)

        
        
    #print(vtu.array_names)
    contour_image = False#True
    if contour_image:
        contour_tdens = vtu.contour(
            isosurfaces=[1e-4], scalars='reconstruction_countour')
        pl.add_mesh(contour_tdens, line_width=2,
                    #cmap="bwr",
                    color="blue",
                    show_scalar_bar=False)
        
    pl.camera_position = 'xy'
    pl.zoom_camera(zoom)

    pdf_filename = directory+'matrix_'+'_'.join(labels(*parameters[2:]))+f'_{variable}.pdf'
    print(f'rendering:{pdf_filename})')
    pl.save_graphic(pdf_filename,painter=False)
    print(f'done:\n {pdf_filename})')
