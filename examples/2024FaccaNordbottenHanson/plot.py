import os
import argparse
from common import labels, figure2, figure3, figure4, figure5, figure6, figure7, figure8
from figures import plot_figure2, plot_figure3, plot_figure4, plot_figure5, plot_figure678

if (__name__ == '__main__'):
    parser = argparse.ArgumentParser(description='Create data for the figures in the manuscript')
    parser.add_argument("-f","--figure", type=str,default='all', help="all(default) or the number of figure (2-8)")
    args, unknown = parser.parse_known_args()

    figures = []
    combinations = []
    vtu_dir = "./results/"
    
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
