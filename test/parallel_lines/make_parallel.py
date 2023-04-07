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
r=draw.Rectangle(10,20,80,40, stroke='black', stroke_width=0, fill='black', filter="url(#f1)")
d.append(r)
masks.append(r)

for circles in source:
    d.append(circles)
    sources.append(circles)

for circles in sink:
    d.append(circles)
    sinks.append(circles)

        

#d.setRenderSize(400,200)  # Alternative to setPixelScale
#temp_file='temp'+'lines.svg'
#d.saveSvg(temp_file)

#for img in [d,sources,sinks,networks,masks]:
#    img = include_filters(img.asSvg())
d = include_filters(d.asSvg())
sources = include_filters(sources.asSvg())
sinks = include_filters(sinks.asSvg())
networks = include_filters(networks.asSvg())
masks = include_filters(masks.asSvg())

d.write('lines.svg')
sources.write('sources.svg')
sinks.write('sinks.svg')
networks.write('networks.svg')
masks.write('masks.svg')

print(d)

from cairosvg import svg2png

svg2png(ET.tostring(d,encoding="unicode"),write_to='lines.png')
svg2png(bytestring=sources,write_to='sources.png')
svg2png(bytestring=sinks,write_to='sinks.png')
svg2png(bytestring=networks,write_to='networks.png')
svg2png(bytestring=masks,write_to='masks.png')

# def add_filters(no_filter_svg_file, corrected_svg_file):
#     SVG_NS = "http://www.w3.org/2000/svg"
#     svg = ET.parse(no_filter_svg_file).getroot()
#     defs = svg.find('.//{%s}defs' % SVG_NS)
#     gauss_filter='''<filter id="f1"> <feGaussianBlur  stdDeviation="1"/></filter>'''
#     ele=ET.fromstring(gauss_filter)
#     defs.insert(100,ele)
#     tree = ET.ElementTree(svg)
#     tree.write(corrected_svg_file)


# add_filters(temp_file,'lines.svg')

