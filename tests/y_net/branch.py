# -*- coding: utf-8 -*-
"""
Crate a y-shaped network with 3 Dirac masses 
and a given branch exponent alpha.
"""

#!/usr/bin/env python
import sys
import numpy as np
import drawSvg as draw
from copy import deepcopy as cp

import xml.etree.ElementTree as ET

import subprocess

def svg2png_ink(draw, output_png_path):
    """
    Save drawSvg.Drawing object as png file using inkscape
    since drawSvg does not support filters.
    """
    width = draw.width*2
    height = draw.height*2

    
    svg_str = draw.asSvg()

    inkscape = 'inkscape'
    # svg string -> write png file
    subprocess.run([inkscape, '--export-type=png', f'--export-filename={output_png_path}', f'--export-width={width}', f'--export-height={height}', '--pipe'], input=svg_str.encode())

def svg2pdf(svg,pdf):
    inkscape = 'inkscape'
    print(svg)
    print(pdf)
    command=f'{inkscape} {svg} --export-pdf={pdf}'# --export-latex'
    subprocess.run(command, shell=True, stderr=subprocess.STDOUT)

        

# define filters
blur = draw.Filter()
blur.append(
    draw.FilterItem('feGaussianBlur', 
                    in_='SourceGraphic', 
                    stdDeviation=1))

def y_branch(coordinates, masses, alpha):
    """
    Given a forcing term with 3 Dirac masses and the branch exponent alpha,
    returns the optimal topology of the network and the points of the network.
    See 
    @article{xia2003optimal,
    title={Optimal paths related to transport problems},
    author={Xia, Qinglan},
    journal={Communications in Contemporary Mathematics},
    volume={5},
    number={02},
    pages={251--279},
    year={2003},
    publisher={World Scientific}
    }

    Args:
    2dcordinates:list of the coordinate of the forcing term
    masses: the values of the forcing term (they must sum to zero)
    alpha: branching exponent

    return:
    v: topology of the optimal network
    p: points in the optimal network( if the y-shape is optimal and there is
       a biforcation node, the node coordinate are appended to 2dcoordinates)  
    """

    points=np.array(coordinates)
    masses=abs(np.array(masses))

    print(points)

    
    # there are 3 masses combinations since sum(masses)=0
    # +, -, -
    # +, +, - => -, -, + (reverse flow)
    # -, +, + => +, -, - (reverse flow)
    # -, -, + 
    # but everthing can be reduced to the first configuration
    # where one source sends to two sink

    coord_O = points[0,:]
    coord_Q = points[1,:]
    coord_P = points[2,:]


    m_O=abs(masses[0])
    m_Q=abs(masses[1])
    m_P=abs(masses[2])

    OP=coord_P-coord_O
    OQ=coord_Q-coord_O
    QP=coord_P-coord_Q

    OQP=np.arccos( np.dot(-OQ, QP)/ ( np.sqrt(np.dot(OQ,OQ)) * np.sqrt(np.dot(QP,QP) )))
    QPO=np.arccos( np.dot(-OP,-QP)/ ( np.sqrt(np.dot(OP,OP)) * np.sqrt(np.dot(QP,QP) )))
    POQ=np.arccos( np.dot( OQ, OP)/ ( np.sqrt(np.dot(OQ,OQ)) * np.sqrt(np.dot(OP,OP) )))
    
    k_1=(m_P/m_O)**(2*alpha)
    k_2=(m_Q/m_O)**(2*alpha)

    theta_1=np.arccos( (k_2-k_1-1)/(2*np.sqrt(k_1))     )
    theta_2=np.arccos( (k_1-k_2-1)/(2*np.sqrt(k_2))     )
    theta_3=np.arccos( (1-k_1-k_2)/(2*np.sqrt(k_1*k_2)) )
    
    v=[];
    if (POQ>=theta_3):
        B_opt=coord_O
        v.append([0,1])
        v.append([0,2])
        p = cp(points)
    elif ( (OQP>=theta_1) & (POQ<theta_3)):
        B_opt=coord_Q
        v[:,0]=[0,1]
        v[:,1]=[1,2]
        p = cp(points)
    elif ( (QPO>=theta_2) & (POQ<theta_3)):
        B_opt=coord_P
        v[:,0]=[0,2]
        v[:,1]=[1,2]
        p = cp(points)
    else:
        QM=np.dot(OP,OQ)/np.dot(OP,OP) * OP - OQ
        PH=np.dot(OP,OQ)/np.dot(OQ,OQ) * OQ - OP
        
        R=(coord_O+coord_P)/2.0 - (np.cos(theta_1)/np.sin(theta_1))/2.0 * np.sqrt(np.dot(OP,OP)/ np.dot(QM,QM) )* QM
        S=(coord_O+coord_Q)/2.0 - (np.cos(theta_2)/np.sin(theta_2))/2.0 * np.sqrt(np.dot(OQ,OQ)/ np.dot(PH,PH) ) * PH
        RO=coord_O-R
        RS=S-R
        
        B_opt=2*( (1-np.dot(RO,RS) / np.dot(RS,RS)) *R + np.dot(RO,RS)/np.dot(RS,RS)*S)-coord_O
        
        #p_B=tuple([tuple(i) for i in B_opt])
        #p_B=[i for i in enumerate(B_opt)]
        p_B=B_opt.tolist()

        p = np.zeros([4,2])
        p[0:3,:] = points
        p[3,:] = p_B
        v.append([0,3])
        v.append([1,3])
        v.append([2,3])
        
    return p,v

if (__name__ == '__main__') :
    width = int(sys.argv[1])
    left = float(sys.argv[2])
    right = float(sys.argv[3])
    alpha = float(sys.argv[4])
    label = sys.argv[5]

    x_len = 50
    y_len = 100
    coord_points = [[x_len/2,0],[0,y_len],[x_len,y_len]]
    masses = [1.0,-left,-right]
    coord, topol = y_branch(coord_points,masses,alpha)

    fluxes = [abs(f) for f in masses]
    
    print('optimal topol')
    print(topol)
    print('coordinates')
    print(coord)

    
    
    background = draw.Drawing(x_len, y_len,  origin=(0,0))
    background.setPixelScale(50)  # Set number of pixels per geometry unit

    # set back ground color
    r = draw.Rectangle(0,0,x_len, y_len, stroke_width=0, fill='white')
    background.append(r)

    # source and sinks are the same
    sources = cp(background)
    sinks = cp(background)
    masks = cp(background)
    networks = cp(background)
    all = cp(background)
    base = cp(background)


    scale = 2
    # Draw network
    if len(topol) == 2:
        # v-shaped network
        fluxes=fluxes[1:]

    for i, edge in enumerate(topol):
        n1,n2=edge
        line = draw.Line(coord[n1][0], coord[n1][1],coord[n2][0], coord[n2][1],
                            stroke='black', stroke_width=scale*fluxes[i], fill='none')
        
        networks.append(line)
        all.append(line)
    
    if (len(coord)==4):
        node=3
        circle = draw.Circle(coord[node][0], coord[node][1], scale/2,
                                       fill='black')
        all.append(circle)
        base.append(circle)
    

    # Draw sources and sinks
    scale=scale/2 
    for node, value in enumerate(masses):
        if (value > 0):
            circle = draw.Circle(coord[node][0], coord[node][1], scale*np.sqrt(abs(value)),
                                       fill='red')
            
            sources.append(circle)
            all.append(circle)
            base.append(circle)
            
        elif (value < 0):
            circle = draw.Circle(coord[node][0], coord[node][1], scale*np.sqrt(abs(value)),
                                     fill='blue')

            sinks.append(circle)
            all.append(circle)
            base.append(circle)
            
    # Draw maskssucced
    r=draw.Rectangle(10,10,30,10+width, stroke='black', stroke_width=0, fill='black', filter=blur)
    masks.append(r)

    corrupted = cp(networks)
    r=draw.Rectangle(10,10,30,10+width, fill='white', filter=blur)
    corrupted.append(r)


    
    

    sources.saveSvg(label+'sources.svg')
    sinks.saveSvg(label+'sinks.svg')
    networks.saveSvg(label+'networks.svg')
    masks.saveSvg(label+'masks'+str(width)+'.svg')
    corrupted.saveSvg(label+'corrupted'+str(width)+'.svg')
    all.saveSvg(label+'all.svg')
    base.saveSvg(label+'base.svg')

    svg2pdf(label+'all.svg',label+'all.pdf')
    svg2pdf(label+'corrupted'+str(width)+'.svg',label+'corrupted'+str(width)+'.pdf')
    

    svg2png_ink(sources, label+'sources.png')
    svg2png_ink(sinks, label+'sinks.png')
    svg2png_ink(networks, label+'networks.png')
    svg2png_ink(masks, label+'masks'+str(width)+'.png')
    svg2png_ink(corrupted, label+'corrupted'+str(width)+'.png')
    svg2png_ink(all, label+'all.png')
    svg2png_ink(base, label+'base.png')
