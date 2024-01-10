# -*- coding:utf-8 -*-

import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.animation as animation
from cycler import cycler
# Scripts
from csv_plot import CSVClass


class PlotFigure(CSVClass):
  fontsize = 27
  fontdict = {
      'family': 'Times New Roman',
      'style': 'normal',  # italic
      'weight': 'normal',
      'color': 'black',
      'size': 27,
  }
  fontdict_prop = {
      'family': 'simsun',  # Times New Roman
      'style': 'normal',  # italic
      'weight': 'normal',
      'size': 27,
  }

  def __init__(self) -> None:
    self.inits_plots()

  def inits_plots(self):
    plt.close("all")
    rc('mathtext',
        default='it',  # 'rm', 'cal', 'it', 'tt', 'sf', 'bf', 'default', 'bb', 'frak', 'scr', 'regular'
        fontset='custom',
        bf='sans:bold',
        cal='cursive',
        it='Times New Roman:italic',
        rm='sans',
        sf='sans',
        tt='monospace'
       )
    # rc('text', usetex=True)
    plt.rcParams['font.family'] = ['Times New Roman']
    rc('font', size=self.fontsize, stretch='normal', style='normal')
    rc('lines',
       linestyle='-',
       linewidth=5.0,
       color='#1f77b4',
       marker='p',
       markersize=6,
       markeredgewidth=0.5
       )
    rc('xtick',
       labelsize=self.fontsize,
       alignment='center',
       bottom=True,
       color='black',
       direction='out',
       labelbottom=True,
       labeltop=False,
       top=False)
    rc('xtick.major', bottom=True, pad=3.5, size=3.5, top=True, width=0.8)
    rc('xtick.minor',
       bottom=True,
       size=2.0,
       top=True,
       visible=True,
       width=0.6)
    rc('yaxis', labellocation='center')
    rc('ytick',
       labelsize=self.fontsize,
       alignment='center_baseline',
       color='black',
       direction='out',
       labelleft=True,
       labelright=False,
       left=True,
       right=False)
    rc('ytick.major', left=True, pad=3.5, right=True, size=3.5, width=0.8)
    rc('ytick.minor',
       left=True,
       pad=3.4,
       right=True,
       size=2.0,
       visible=False,
       width=0.6)
    custom_cycler = cycler(
        color=['b', 'g', 'y', 'k'],  # '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'
        linewidth=[1.8, 1.8, 1.8, 1.8],
        linestyle=['-', '--', '-.', ':'],
        # marker=['^', 'p', '^', 's']
    )
    rc('axes',
       titlesize=self.fontsize,
       labelsize=self.fontsize,
       grid=True,
       prop_cycle=custom_cycler)
    rc('grid', alpha=1.0, color='#b0b0b0', linestyle='--', linewidth=0.8)
    rc('legend',
        borderaxespad=0.5,  # the pad between the axes and legend border, in font-size units.
        borderpad=0.4,  # the fractional whitespace inside the legend border, in font-size units.
        loc='best',  # best, upper right, upper left, lower left, lower right, right, center left, center right, lower center, upper center, center
        fontsize=self.fontsize,  # int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        edgecolor='0.8',  # "inherit" or color. the legend's background patch edge color.
        facecolor='inherit',  # "inherit" or color. the legend's background color.
        fancybox=True,  # whether round edges should be enabled around the fancybboxpatch which makes up the legend's background.
        framealpha=0.8,  # the alpha transparency of the legend's background.
        frameon=True,  # whether the legend should be drawn on a patch (frame).
        handleheight=0.7,
        handlelength=2.0,  # the length of the legend handles, in font-size units.
        handletextpad=0.8,  # the pad between the legend handle and text, in font-size units.
        labelspacing=0.5,  # the vertical space between the legend entries, in font-size units.
        columnspacing=2.0,  # the spacing between columns, in font-size units.
        markerscale=1.0,  # the relative size of legend markers compared with the originally drawn ones.
        numpoints=1,  # the number of marker points in the legend when creating a legend entry for a line2d (line).
        scatterpoints=1,  # the number of marker points in the legend when creating a legend entry for a pathcollection (scatter plot).
        shadow=False,  # whether to draw a shadow behind the legend.
        title_fontsize=None  # int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
       )
    # figure
    rc('figure', dpi=120, figsize=(10, 6.18))  # 5.9, 2.8
    rc('savefig', dpi=300, transparent=True)  # 对于SVG格式，DPI参数不起作用了
    # the amount of width/height reserved for space between subfigures, expressed as a fraction of the average subfigure width/height.
    rc('figure.subplot',
        # hspace=None,
        # wspace=None,
        # bottom=None,
        # top=None,
        # left=None,
        # right=None
       )

  def plot_show(self):
    plt.show()

  def create_plot(self):
    fig, ax = plt.subplots()
    return fig, ax
