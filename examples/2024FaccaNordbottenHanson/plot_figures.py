import pyvista as pv
import numpy as np
import os
import argparse
from common import labels, figure1, figure2, figure3, figure4, figure5, figure6, figure7, figure8
import copy

# this is to use latex see https://github.com/pyvista/pyvista/discussions/2928
import vtk 
import sys
import itertools

pv.start_xvfb()


def get_vtu(pathdotvtu):
    try:
        os.path.exists(pathdotvtu)
        vtu = pv.read(pathdotvtu)
        return vtu
    except:
        # get name of the file without directory
        filename = os.path.basename(pathdotvtu)
        dirname = os.path.dirname(pathdotvtu)   
        # remove last 5 characters with "_0.vtu"
        filename_cut = filename[:-6]

        pathdotvtu = os.path.join(dirname,filename_cut,filename)
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
    network = 2 #0=no, 1=remove were the mask is applied, 2= show everywhere
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
    labels_name = '_'.join(labels(*parameters_local))
    try:
        print(labels_name)
        path_vtu = os.path.abspath(directory+labels_name+'_0.vtu')
        vtu = pv.read(path_vtu)
        print(path_vtu)
    except:
        path_vtu = os.path.abspath(directory
                + labels_name+'/'
                + labels_name+'_0.vtu')
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

def plot_figure1(parameters, dir_vtu="./results/"):
    width=500
    height=600
    zoom=1.3
    scaling=0.5
    
    mask=True
    network = 1 #0=no, 1=remove were the mask is applied, 2= show everywhere
    sink = False 
    bounds = False

    
    example = parameters[0]
    mask = parameters[1]
    #remove extension
    mask_name = os.path.splitext(mask)[0]
    directory = dir_vtu+example+'/'+mask_name+'/'

    parameters_local = parameters[2:]
    
    window_size=(width,height)
    pl = pv.Plotter(border=False,
            window_size=window_size,
            off_screen=True,
    )


    #
    # network
    #
    if network>=1:
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
                    opacity = [0,1],
                    show_scalar_bar=False)
        
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

    pdf_filename = directory+'network_artifacts_masked.pdf'
    print(f'rendering:{pdf_filename})')
    pl.save_graphic(pdf_filename,painter=False)
    print(f'done:\n {pdf_filename})')


    # figure 1g
    width=500
    height=600
    zoom=1.3
    scaling=0.5
    
    variable='tdens'
    mask=True
    network = 2 #0=no, 1=remove were the mask is applied, 2= show everywhere
    sink = False 
    
    
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
        mask_vtu_file = (directory
                             + labels(*parameters_local)[0]
                             +'_' + "netnetwork"# labels(*parameters_local)[5]
                             +'_network_mask/'
                             + labels(*parameters_local)[0]
                             +'_' + "netnetwork"#labels(*parameters_local)[5]
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


        # artifacts_vtu_file = "data/y_net_frog200/artifacts/artifacts_0.vtu"
        # artifacts_vtu = pv.read(artifacts_vtu_file)

        # art_data = artifacts_vtu.get_array("network")
        # art_support= art_data
        # thr = 1e-3
        # art_support[art_support>=thr]=1
        # art_support[art_support<thr]=0
        # mask_data = mask_vtu.get_array("mask")
        # art_support *= (1.0-mask_data)

        # pl.add_mesh(artifacts_vtu,
        #             scalars=art_support,
        #             cmap="Reds",
        #             opacity = [0,0.5],
        #             show_scalar_bar=False)

        
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

    pdf_filename = directory+'reconstruction_figure1.pdf'
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
    network = 2 #0=no, 1=remove were the mask is applied, 2= show everywhere
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
            mask_vtu_file = (directory
                             + labels(*parameters_local)[0]
                             +'_' + "netnetwork"# labels(*parameters_local)[5]
                             +'_network_mask/'
                             + labels(*parameters_local)[0]
                             +'_' + "netnetwork"#labels(*parameters_local)[5]
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


        artifacts_vtu_file = "data/y_net_frog200/artifacts/artifacts_0.vtu"
        artifacts_vtu = pv.read(artifacts_vtu_file)

        art_data = artifacts_vtu.get_array("network")
        art_support= art_data
        thr = 1e-3
        art_support[art_support>=thr]=1
        art_support[art_support<thr]=0
        mask_data = mask_vtu.get_array("mask")
        art_support *= (1.0-mask_data)

        pl.add_mesh(artifacts_vtu,
                    scalars=art_support,
                    cmap="Reds",
                    opacity = [0,0.5],
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
                cmap="spring_r",
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
    plot_network = 2 #0=no, 1=remove were the mask is applied, 2= show everywhere
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
    mask_data = mask_vtu.get_array("mask")

    if network == 1:
        print("mask")
        net_support *= (1.0-mask_data)

    r=np.ones(50)
    r[0]=0

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

    if plot_sink:
        file_btp = (directory 
                    + 'nref0'
                    +'_btp_0.vtu')
        btp_vtu = get_vtu(file_btp)
        contours = btp_vtu.contour(isosurfaces=2, scalars='sink_countour')
        pl.add_mesh(contours, line_width=2,
                    color="black",
                    show_scalar_bar=False)

    correction = 0.95
    letter_size=30
    pl.add_text(
            text="L",
            #position=[0.3*width,0.16*correction*height],
            position=[0.4*width,0.1*height],
            font_size=int(letter_size*scaling),
            color=None,
            font="times",
            shadow=False,
            name=None,
            viewport=False,
            orientation=0.0,
            font_file=None)

    pl.add_text(
            text="R",
            #position=[0.23*width,0.3*correction*height],
            position=[0.6*width,0.1*height],
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
    pdf_filename=directory+"frog_network.pdf"
    pl.save_graphic(pdf_filename,painter=False)
    print(pdf_filename)
    

    # figure 5.b
    
    plot_mask=True
    plot_network = 1 #0=no, 1=remove were the mask is applied, 2= show everywhere
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
    network = 1
    if binary: 
        thr = 1e-3
        net_support[net_support>=thr]=1
        net_support[net_support<thr]=0
    mask_data = mask_vtu.get_array("mask")

    if network == 1:
        print("mask")
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


    #boundary_vtu_file = ('./data/frog_tongue/rectangle_shift/rectangle_shift_0.vtu')
    #boundary_vtu = pv.read(boundary_vtu_file)

    #b_contours = boundary_vtu.contour(isosurfaces=2, scalars='img_contour')
    #pl.add_mesh(b_contours, line_width=2, color="black",show_scalar_bar=False)


    artifacts_vtu_file = "data/frog_tongue/artifacts/artifacts_0.vtu"
    artifacts_vtu = pv.read(artifacts_vtu_file)

    art_data = artifacts_vtu.get_array("network")
    art_support= art_data
    thr = 1e-3
    art_support[art_support>=thr]=1
    art_support[art_support<thr]=0
    mask_data = mask_vtu.get_array("mask")
    art_support *= (1.0-mask_data)

    pl.add_mesh(artifacts_vtu,
                scalars=art_support,
                cmap="Reds",
                opacity = [0,0.5],
                show_scalar_bar=False)

    

    if plot_sink:
        file_btp = (directory 
                    + 'nref0'
                    +'_btp_0.vtu')
        btp_vtu = get_vtu(file_btp)
        
    contours = btp_vtu.contour(isosurfaces=2, scalars='sink_countour')
    pl.add_mesh(contours, line_width=2, color="black",show_scalar_bar=False)

    contours = mask_vtu.contour(isosurfaces=2, scalars='mask_countour')
    pl.add_mesh(contours, line_width=2, color="red",show_scalar_bar=False)

    
    letters = False
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
    


    pl.camera_position = 'xy'
    pl.zoom_camera(zoom)
    pdf_filename=directory+"frog_tongue_shifted.pdf"
    pl.save_graphic(pdf_filename,painter=False)
    print(pdf_filename)


    # Figure 5d 

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
    
    mask_vtu_file = (directory
                     + 'nref0_netmup3.0e+00zero5.0e+02_network_mask'
                     +'/nref0_netmup3.0e+00zero5.0e+02_network_mask_0.vtu')
    mask_vtu = pv.read(mask_vtu_file)



    scaling = width/1000

    bar_dict = {
        "position_x": 0.1,
        "position_y": 0.00,
        "width": 0.8,
        "height" : 0.15}
    
    net_data = mask_vtu.get_array("network")
    net_support= net_data
    network = plot_network
        
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
                cmap="spring_r",
                opacity=list(0.89*r),
                clim = clim,
                #below_color='white',
                #above_color='black',
                show_scalar_bar=True,
                scalar_bar_args=sargs,
            log_scale=False
    )



    if plot_sink:
        file_btp = (directory
                            + 'nref0'
                            +'_btp/'
                            + 'nref0'
                            +'_btp_0.vtu')
        btp_vtu = pv.read(file_btp)

        contours = btp_vtu.contour(isosurfaces=2, scalars='sink_countour')
        pl.add_mesh(contours, line_width=2, color="black",show_scalar_bar=False)

    contours = mask_vtu.contour(isosurfaces=2, scalars='mask_countour')
    pl.add_mesh(contours, line_width=2, color="red",show_scalar_bar=False)

    letters = False
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



    pl.camera_position = 'xy'
    pl.zoom_camera(zoom)
    pdf_filename=directory+"mup3.0e+00zero5.0e+02_network.pdf"
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
    #clim = [1e-6,1e-1] 
    
    pl.add_mesh(vtu.copy(),
            scalars=var, 
            #cmap="hot_r",
            cmap="spring_r",
            opacity=list(0.89*r),
            clim = clim,
            #below_color='white',
            #above_color='black',
            show_scalar_bar=True,
            scalar_bar_args=sargs,
            log_scale=True,
    )

    if plot_sink:
        file_btp = (directory
                            + 'nref0'
                            +'_btp/'
                            + 'nref0'
                            +'_btp_0.vtu')
        btp_vtu = pv.read(file_btp)

        contours = btp_vtu.contour(isosurfaces=2, scalars='sink_countour')
        pl.add_mesh(contours, line_width=2, color="black",show_scalar_bar=False)

            
                
    pl.camera_position = 'xy'
    pl.zoom_camera(zoom)

    pdf_filename = directory+'matrix_'+'_'.join(labels(*parameters[2:]))+f'_{variable}.pdf'
    pl.save_graphic(pdf_filename,painter=False)
    print(f'{pdf_filename}')

    
    
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
        # frog_tongue/mask/nref0_netnetwork_network_mask
        mask_vtu_file = (directory 
                    + labels(*parameters_local)[0]
                    +'_'+'netnetwork'# labels(*parameters_local)[5]
                    +'_network_mask_0.vtu')
        mask_vtu = get_vtu(mask_vtu_file)
        
        net_data = mask_vtu.get_array("network")
        net_support= net_data
        thr = 1e-3
        net_support[net_support>=thr]=1
        net_support[net_support<thr]=0

        var = copy.deepcopy(vtu.get_array(variable))
        thr=1e-5
        var[var>=thr]=1.0
        var[var<thr]=0.0

        net_support = net_support*abs(net_support-var)

        mask_data = mask_vtu.get_array("mask")
        if network == 1:
            
            net_support *= (1.0-mask_data)

            
            
        pl.add_mesh(mask_vtu,
                    scalars=net_support,
                    #cmap="summer_r",
                    cmap="binary",
                    #cmap="Wistia",
                    #cmap="Greens",
                    opacity = [0,0.9],
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
    clim = [1e-6,5e-2] 
    
    pl.add_mesh(vtu.copy(),
            scalars=var, 
            #cmap="hot_r",
            #cmap="summer_r",
            #    cmap="Reds",
            cmap="spring_r",
            #    cmap="rainbow",
            #    cmap="cubehelix_r",
            #cmap="GnBu",
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
        #contours = mask_vtu.contour(isosurfaces=2, scalars='mask_countour')
        #pl.add_mesh(contours, line_width=2, color="red",show_scalar_bar=False)
        mask_data[mask_data<1e-1]=0.0
        pl.add_mesh(mask_vtu.copy(),
            scalars=mask_data, 
            #cmap="hot_r",
            #cmap="summer_r",
            #    cmap="Reds",
            cmap="Greens_r",
            #    cmap="rainbow",
            #    cmap="cubehelix_r",
            #cmap="GnBu",
            opacity=list(0.5*r),
            clim = clim,
            #below_color='white',
            #above_color='black',
            show_scalar_bar=False,
            scalar_bar_args=sargs,
            #log_scale=True,
     )
        

    if sink:
        file_btp = (directory 
                        + labels(*parameters_local)[0]
                        +'_btp_0.vtu')
        btp_vtu = get_vtu(file_btp)

        contours = btp_vtu.contour(isosurfaces=2, scalars='sink_countour')
        pl.add_mesh(contours, line_width=2, color="black",show_scalar_bar=False)

    letters = False#True
    if letters:
        correction = 0.95
        letter_size=30
        pl.add_text(
            text="a",
            #position=[0.3*width,0.16*correction*height],
            position=[0.3*width,0.215*height],
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
            position=[0.23*width,0.32*height],
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
            position=[0.36*width,0.625*height],
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
            position=[0.64*width,0.54*height],
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
            position=[0.71*width,0.67*height],
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
            position=[0.84*width,0.4*correction*height],
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
            position=[0.64*width,0.19*height],
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

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Create data for the figures in the manuscript')
    parser.add_argument("-f","--figure", type=str,default='all', help="all(default) or the number of figure (2-8)")
    args, unknown = parser.parse_known_args()

    figures = []
    combinations = []
    vtu_dir = "./results/"

    if ((args.figure == '1') or (args.figure == 'all')):
        combinations = figure1()
        for c in combinations:
            plot_figure1(c,vtu_dir)

    
    if ((args.figure == '2') or (args.figure == 'all')):
        combinations = figure2()
        for c in combinations:
            plot_figure2(c,vtu_dir)
        
    if ((args.figure == '3') or (args.figure == 'all')):
        combinations = figure3()
        for c in combinations:
            plot_figure3(c,vtu_dir)

    if ((args.figure == '4') or (args.figure == 'all')):
        combinations = figure4()
        for c in combinations:
            plot_figure4(c,vtu_dir)

    if ((args.figure == '5') or (args.figure == 'all')):
        plot_figure5(vtu_dir+"frog_tongue/mask/")

    if ((args.figure == '6') or (args.figure == 'all')):
        combinations = figure6()
        for c in combinations:
            plot_figure678(c,vtu_dir)
        
    if ((args.figure == '7') or (args.figure == 'all')):
        combinations = figure7()
        for c in combinations:
            plot_figure678(c,vtu_dir)
        
    if ((args.figure == '8') or (args.figure == 'all')):
        combinations = figure8()
        for c in combinations:
            plot_figure678(c,vtu_dir)
