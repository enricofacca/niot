import drawSvg as draw
from copy import deepcopy as cp

#
# Passing the filters did not worked.
# We have to passit "manually" to the xml structure
#
import xml.etree.ElementTree as ET

def include_filters(no_filter_svg):
    SVG_NS = "http://www.w3.org/2000/svg"
    svg = ET.fromstring(no_filter_svg)
    defs = svg.find('.//{%s}defs' % SVG_NS)
    gauss_filter='''<filter id="f1"> <feGaussianBlur  stdDeviation="1"/></filter>'''
    ele=ET.fromstring(gauss_filter)
    defs.insert(100,ele)
    tree = ET.ElementTree(svg)
    return tree

import subprocess

def svg2png_ink(draw, output_png_path):
    """
    Save drawSvg.Drawing object as png file using inkscape
    since drawSvg does not support filters.
    """
    width = draw.width
    height = draw.height

    
    svg_str = draw.asSvg()

    inkscape = 'inkscape'
    # svg string -> write png file
    subprocess.run([inkscape, '--export-type=png', f'--export-filename={output_png_path}', f'--export-width={width}', f'--export-height={height}', '--pipe'], input=svg_str.encode())
        

# define filters
blur = draw.Filter()
blur.append(draw.FilterItem('feGaussianBlur', in_='SourceGraphic', stdDeviation=5))

# coordinate of the network nodes 
# connection topology
# forcing term (should be balance and lenght equal to number of nodes)
coord=[[10,10],[10,90],[20,10],[40,90]]
topol=[[0,1],[2,3]]
forcing=[-1,1,-1,1]


d = draw.Drawing(50, 100,  origin=(0,0), idPrefix='d', displayInline=False,
                  **{"xmlns:inkscape" : "http://www.inkscape.org/namespaces/inkscape"}
)
d.setPixelScale(2)  # Set number of pixels per geometry unit

# se back ground color
r=draw.Rectangle(0,0,100,100, stroke_width=0, fill='white')
d.append(r)

# source and sinks are the same
sources = cp(d)
sinks = cp(d)
masks = cp(d)


# Draw network
network=[]
for edge in topol:
    n1,n2=edge
    network.append(draw.Line(coord[n1][0], coord[n1][1],coord[n2][0], coord[n2][1],
                         stroke='black', stroke_width=6, fill='none'))
for line in network:
    d.append(line)
networks = cp(d)

    
# Draw the sources and sinks
source=[]
sink=[]
for node, value in enumerate(forcing):
    if (value>0):
        source.append(draw.Circle(coord[node][0], coord[node][1], 3,
                                   fill='red'))
    elif (value<0):
        sink.append(draw.Circle(coord[node][0], coord[node][1], 3,
                                   fill='blue'))
        
# Draw mask
r=draw.Rectangle(10,20,80,40, stroke='black', stroke_width=0, fill='black', filter=blur)
d.append(r)
masks.append(r)

for circles in source:
    d.append(circles)
    sources.append(circles)

for circles in sink:
    d.append(circles)
    sinks.append(circles)

        
d.saveSvg('lines.svg')
sources.saveSvg('sources.svg')
sinks.saveSvg('sinks.svg')
networks.saveSvg('networks.svg')
masks.saveSvg('masks.svg')

svg2png_ink(d, 'lines.png')
svg2png_ink(sources, 'sources.png')
svg2png_ink(sinks, 'sinks.png')
svg2png_ink(networks, 'networks.png')
svg2png_ink(masks, 'masks.png')

