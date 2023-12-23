import pyvista as pv
import numpy as np
path_vtu="squares2.vtu"
vtu = pv.read(path_vtu)

# plot network
# pl = pv.Plotter()
# pl.add_mesh(vtu,
#             scalars='network', 
#             cmap="binary",
#             opacity = [0,1],
#             show_scalar_bar=True)
# pl.view_xy()
# pl.add_bounding_box()
# pl.save_graphic("network.pdf")


pl = pv.Plotter()
pl.add_mesh(vtu.copy(),
            scalars="tdens", 
            #cmap="hot",
#            cmap="magma",
            #cmap="gray",
            cmap="Reds",
            #cmap='glasbey',
            #flip_scalars=True,
            clim=[1e-4,1e-2],
            opacity=[0,1,1,1,1,1,1,1,1],
            #below_color='w',
            #above_color='red',
            categories=True,
            show_scalar_bar=True,
            pbr=False,
)
pl.view_xy()
#pl.background_color = "pink"
#pl.global_theme.below_range_color = "w"
#lut = pv.LookupTable()
#lut.below_range_color = [1,1,1]
#lut.below_range_opacity = 0
#pl.add_bounding_box(color='w')
pl.set_background('white')
pl.save_graphic("tdens.pdf")

# Trying to merge

pl = pv.Plotter()
pl.add_mesh(vtu,
            scalars='network', 
            cmap="binary",
            opacity = [0,0.1],
            show_scalar_bar=False)
r=np.ones(50)
r[0]=0

sargs = dict(
    title="",
    title_font_size=20,
    label_font_size=30,
    shadow=True,
    n_labels=2,
    #italic=True,
    fmt="%.0e",
    font_family="arial",
    vertical=False,
    position_x=0.25,
    position_y=0.75,
    width=0.4,
    height=0.1,    
)


contours = vtu.contour(scalars='mask_countour')
pl.add_mesh(contours, line_width=1, color="b",show_scalar_bar=False)
#print(contours)


#n_contours = 2
#rng = [200, 500]
#contours, edges = vtu.contour_banded(n_contours)#, scalars='mask_countour')
#pl.add_mesh(edges, line_width=1, render_lines_as_tubes=True, color='k')

#pl.add_mesh(contours)

pl.add_mesh(vtu.copy(),
            scalars='tdens', 
            cmap="hot_r",
            opacity=list(0.89*r),
            clim = [1e-4, 1e-2],
            show_scalar_bar=True,
            scalar_bar_args=sargs
)
pl.view_xy()
pl.add_bounding_box()
pl.save_graphic("merge.pdf")
pl.window_size = 1000, 1000
pl.screenshot('foo.png', window_size=[400,400])

# Hi
#I am trying to plot some scalars (tdens and network) from a vtu file.
#I would like to have the network in the background with the tdens variable on top.
#I tried to play with different opacity level, the order of the plots but I did not get the expected results, that you can see in squares.png.
# Attacehd the input vtu file.




