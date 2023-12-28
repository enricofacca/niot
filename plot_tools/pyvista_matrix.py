import pyvista as pv
import numpy as np
# this is to use latex see https://github.com/pyvista/pyvista/discussions/2928
import vtk 
import sys
import os
import itertools

def labels(nref,fem,
           gamma,wd,wr,
           corrupted_as_initial_guess,
           confidence,
           tdens2image,
           tdens2image_scaling, 
           method):
    label= [
        f'nref{nref}',
        f'fem{fem}',
        f'gamma{gamma:.1e}',
        f'wd{wd:.1e}',
        f'wr{wr:.1e}',
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
        else:
            raise ValueError(f'Unknown method {method}')
    label.append(f'method{short_method}')
    return label


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
    if (len(c) != 2):
        raise ValueError('Only due elments can more more than one')
    return f,c


directory="../examples/y_net/mask_large/"


mask=['mask_large.png']
nref=[0]
fems = ['DG0DG0']
gamma = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
wd = [0]#1e-2]
wr = [1e-4]
ini = [0,1]
conf = ['ONE']
maps = [
    {'type':'identity'}, 
#    {'type':'heat', 'sigma': 1e-4},
#    {'type':'pm', 'sigma': 1e-4, 'exponent_m': 2.0},
]
tdens2image_scaling = [1e1]
method = [
    #'tdens_mirror_descent_explicit',
    #'tdens_mirror_descent_semi_implicit',
    #'gfvar_gradient_descent_explicit',
    'gfvar_gradient_descent_semi_implicit',
]

swap_rows_cols = False

# set the order to be passed to labels function
parameters=[nref,fems,gamma,wd,wr,ini,conf,maps,tdens2image_scaling,method]
fixed,  changing = fixed_changing(parameters)
if swap_rows_cols:
    changing = changing[::-1]

rows = parameters[changing[0]]
cols = parameters[changing[1]]

    
# set images
width=1000
height=1000
nrows = len(rows)
ncols = len(cols)
pl = pv.Plotter(shape=(nrows,ncols),
                border=True,
                window_size=(ncols*width,nrows*height),
                off_screen=True,
                )

for i in range(nrows):
    for j in range(ncols):        
        parameters_local = parameters.copy()
        parameters_local[changing[0]] = [rows[i]]
        parameters_local[changing[1]] = [cols[j]]
        # flatten list
        parameters_local = [item for sublist in parameters_local for item in sublist]
        #print(parameters_local)
        #print(labels(*parameters_local))
        labels_local = labels(*parameters_local)
        label_row = labels_local[changing[0]]
        label_col = labels_local[changing[1]]

        path_vtu = directory + '_'.join(labels(*parameters_local))+'.vtu'

        print(i,j,path_vtu)
        
        pl.subplot(i,j)

        vtu = pv.read(path_vtu)
        pl.add_mesh(vtu,
                    scalars="network", 
                    cmap="binary",
                    opacity = [0,0.1],
                    show_scalar_bar=False)
        
        #
        # TDENS
        # 
        if changing[0] > changing[1]:
            title = label_col+' '+label_row
        else:   
            title = label_row+' '+label_col
        sargs = dict(
            title=title,
            title_font_size=40,
            label_font_size=40,
            shadow=True,
            n_labels=2,
            #italic=True,
            fmt="%.0e",
            font_family="arial",
            vertical=False,
            position_x=0.3,
            position_y=0.85,
            width=0.35,
            height=0.1,    
        )

        # this is to make values below min as white/transparent
        r=np.ones(50)
        r[0]=0
        pl.add_mesh(vtu.copy(),
                    scalars='tdens', 
                    #cmap="hot_r",
                    cmap="gnuplot2_r",
                    opacity=list(0.89*r),
                    clim = [1e-4, 1e-2],
                    #below_color='white',
                    #above_color='red',
                    show_scalar_bar=True,
                    scalar_bar_args=sargs
        )
        pl.view_xy(render=False)
        pl.add_bounding_box()
        tdens = vtu.get_array('tdens')
        min_tdens = np.min(tdens)
        max_tdens = np.max(tdens)
        pl.add_text(
            text=rf'${min_tdens:.1e}\leq \mu \leq {max_tdens:.1e}$',
                position=[250,850],
                font_size=20,
                color=None,
                font=None,
                shadow=False,
                name=None,
                viewport=False,
                orientation=0.0,
                font_file=None)
        
        
        #
        # MASK
        #
        contours = vtu.contour(isosurfaces=2, scalars='mask_countour')
        pl.add_mesh(contours, line_width=2, color="black",show_scalar_bar=False)

        
        pl.camera_position = 'xy'
        pl.zoom_camera(1.4)

#pl.window_size = (4000, 1000)
#pl.link_views

#pl.show()
pl.show(auto_close=False)
#print(pl.camera_position)
#if :
print('saved in')
pdf_filename = 'matrix.pdf'  
pl.save_graphic(pdf_filename)
#else:
#    pl.screenshot(f"foo.png",
#                        transparent_background=False, window_size=(nx*1000,1000))

#pl.screenshot('foo.png', window_size=(nx*1000,1000))




exit()

nel=0
for p in parameters:
    print(p)
    print(len(p))
    print(nel)
    if len(p)>1:
        nel+=1
        if nel==1: 
            rows = p
        elif nel==2:
            cols = p
        else:
            raise ValueError('rows or cols must have exactly one element')
if swap_rows_cols:
    rows, cols = cols, rows


directory="../examples/y_net/mask_large/"
tail='tdens_net_mask'
pdf_filename = tail+'.pdf'


#
# Trying to merge
#


combinations = list(itertools.product(*parameters))
arguments = list(itertools.product(*[rows,cols]))
changing_list = changing(parameters)


files = []
texts = []
labels_changing = []
for combination in combinations:
    labels_problem = labels(*combination)
    files.append(directory+ '_'.join(labels_problem)+'.vtu')
    labels_changing.append([labels_problem[i] for i in changing_list])

if swap_rows_cols:
    n = len(rows)
else:
    n = len(cols)
file_grid=list(zip(*[iter(files)]*len(cols)))
label_grid=list(zip(*[iter(labels_changing)]*len(cols)))



width=1000
height=1000
nrows = len(label_grid)
ncols = len(label_grid[0])

print(nrows,ncols)
print(label_grid)
print(file_grid)
exit()

pl = pv.Plotter(shape=(nrows,ncols),
                border=True,
                window_size=(ncols*width,nrows*height),
                off_screen=True,
                )

for i, row in enumerate(file_grid):
    for j, path_vtu in enumerate(row):
        print(i,j,path_vtu)
        print(label_grid[i][j])
        # get row and column from i
        #row = i//2
        #col = i%2
        
        #print(i,row,col)
        
        #pl.subplot(row,col)
        pl.subplot(i,j)

        vtu = pv.read(path_vtu)
        pl.add_mesh(vtu,
                    scalars="network", 
                    cmap="binary",
                    opacity = [0,0.1],
                    show_scalar_bar=False)
        
        #
        # TDENS
        # 
        plot_labels = label_grid[i][j]
        print(plot_labels)
        title = f"{plot_labels[0]} {plot_labels[1]}"
        sargs = dict(
            title=title,
            title_font_size=40,
            label_font_size=40,
            shadow=True,
            n_labels=2,
            #italic=True,
            fmt="%.0e",
            font_family="arial",
            vertical=False,
            position_x=0.3,
            position_y=0.85,
            width=0.35,
            height=0.1,    
        )

        # this is to make values below min as white/transparent
        r=np.ones(50)
        r[0]=0
        pl.add_mesh(vtu.copy(),
                    scalars='tdens', 
                    #cmap="hot_r",
                    cmap="gnuplot2_r",
                    opacity=list(0.89*r),
                    clim = [1e-4, 1e-2],
                    #below_color='white',
                    #above_color='red',
                    show_scalar_bar=True,
                    scalar_bar_args=sargs
        )
        pl.view_xy(render=False)
        pl.add_bounding_box()
        tdens = vtu.get_array('tdens')
        min_tdens = np.min(tdens)
        max_tdens = np.max(tdens)
        pl.add_text(
            text=rf'${min_tdens:.1e}\leq \mu \leq {max_tdens:.1e}$',
                position=[250,850],
                font_size=20,
                color=None,
                font=None,
                shadow=False,
                name=None,
                viewport=False,
                orientation=0.0,
                font_file=None)
        
        
        #
        # MASK
        #
        contours = vtu.contour(isosurfaces=2, scalars='mask_countour')
        pl.add_mesh(contours, line_width=2, color="black",show_scalar_bar=False)

        
        #pl.show()
        #pl.window_size = 1000, 1000
        
        a=2
        pl.camera_position = 'xy'#[(1, 1, 1),
                            #(0.5, 0.5, 0.5),
                            #(0, 1.0, 0.0)]
        pl.zoom_camera(1.33)

#pl.window_size = (4000, 1000)
#pl.link_views

#pl.show()
pl.show(auto_close=False)
#print(pl.camera_position)
#if :
print('saved in')
print(pdf_filename)
pdf_filename = 'matrix.pdf'  
pl.save_graphic(pdf_filename)
#else:
#    pl.screenshot(f"foo.png",
#                        transparent_background=False, window_size=(nx*1000,1000))

#pl.screenshot('foo.png', window_size=(nx*1000,1000))


