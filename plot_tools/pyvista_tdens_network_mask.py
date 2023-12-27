import pyvista as pv
import numpy as np
# this is to use latex see https://github.com/pyvista/pyvista/discussions/2928
import vtk 
import sys
import os

try:
    path_vtu = sys.argv[1]
except:
    path_vtu="squares2.vtu"
vtu = pv.read(path_vtu)

try:
    pdf_tail=sys.argv[2]
except:
    pdf_tail='tdens_net_mask'

pdf_filename = os.path.splitext(path_vtu)[0]+'_'+pdf_tail+'.pdf'

#
# Trying to merge
#
pl = pv.Plotter()
pl.add_mesh(vtu,
            scalars='network', 
            cmap="binary",
            opacity = [0,0.1],
            show_scalar_bar=False)
#
# MASK
#
contours = vtu.contour(isosurfaces=2, scalars='mask_countour')
pl.add_mesh(contours, line_width=2, color="black",show_scalar_bar=False)

#
# TDENS
# 
sargs = dict(
    title="",
    title_font_size=40,
    label_font_size=40,
    shadow=True,
    n_labels=2,
    #italic=True,
    fmt="%.0e",
    font_family="arial",
    vertical=False,
    position_x=0.3,
    position_y=0.92,
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
#pl.show()
pl.window_size = 1000, 1000
pl.zoom_camera('tight')
print('saved in')
print(pdf_filename)
pl.save_graphic(pdf_filename)
#pl.screenshot('foo.png', window_size=[400,400])




