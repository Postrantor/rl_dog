# -*- coding:utf-8 -*-

# %%
# Libs
import time
# Math
import numpy as np
from scipy import interpolate
from scipy import signal
# Plot
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import colors, markers, rc
from cycler import cycler
# Scripts
from scripts.csv_plot import CSVClass
from scripts.make_directory import mk_dir

# %%
fontsize = 18


# %%
class PlotFigure(CSVClass):
  # ==================================================
  #                      Initial_Parameters
  # ==================================================
  def inits_plots(self):
    plt.close("all")
    # font
    # rc(
    #     'mathtext',
    #     default=
    #     'it',  # 'rm', 'cal', 'it', 'tt', 'sf', 'bf', 'default', 'bb', 'frak', 'scr', 'regular'
    #     fontset='custom',
    #     fallback='cm',
    #     fallback_to_cm=None,
    #     bf='sans:bold',
    #     cal='cursive',
    #     it='Times New Roman:italic',
    #     rm='sans',
    #     sf='sans',
    #     tt='monospace')
    # rc('text', usetex=True)
    plt.rcParams['font.family'] = ['Times New Roman']
    rc('font', size=fontsize, stretch='normal', style='normal')
    # lines
    # rc('lines',
    #    linestyle='-',
    #    linewidth=2.0,
    #    color='#1f77b4',
    #    marker=None,
    #    markersize=6,
    #    markeredgewidth=0.5)
    # xtick
    rc('xtick',
       labelsize=fontsize,
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
    # ytick
    rc('yaxis', labellocation='center')
    rc('ytick',
       labelsize=fontsize,
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
    # axes
    custom_cycler = cycler(
        color=['b', 'g', 'y',
               'k'],  # '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'
        linewidth=[1.8, 1.8, 1.8, 1.8],
        linestyle=['-', '--', '-.', ':'],
        # marker=['^', 'p', '^', 's']
    )
    rc('axes',
       titlesize=fontsize,
       labelsize=fontsize,
       grid=True,
       prop_cycle=custom_cycler)
    rc('grid', alpha=1.0, color='#b0b0b0', linestyle='--', linewidth=0.8)
    rc(
        'legend',
        borderaxespad=0.5,  # The pad between the axes and legend border, in font-size units.
        borderpad=0.4,  # The fractional whitespace inside the legend border, in font-size units.
        loc='best',  # best, upper right, upper left, lower left, lower right, right, center left, center right, lower center, upper center, center
        fontsize=16,  # int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
        edgecolor='0.8',  # "inherit" or color. The legend's background patch edge color.
        facecolor='inherit',  # "inherit" or color. The legend's background color.
        fancybox=True,  # Whether round edges should be enabled around the FancyBboxPatch which makes up the legend's background.
        framealpha=0.8,  # The alpha transparency of the legend's background.
        frameon=True,  # Whether the legend should be drawn on a patch (frame).
        handleheight=0.7,
        handlelength=2.0,  # The length of the legend handles, in font-size units.
        handletextpad=0.8,  # The pad between the legend handle and text, in font-size units.
        labelspacing=0.5,  # The vertical space between the legend entries, in font-size units.
        columnspacing=2.0,  # The spacing between columns, in font-size units.
        markerscale=1.0,  # The relative size of legend markers compared with the originally drawn ones.
        numpoints=1,  # The number of marker points in the legend when creating a legend entry for a Line2D (line).
        scatterpoints=1,  # The number of marker points in the legend when creating a legend entry for a PathCollection (scatter plot).
        shadow=False,  # Whether to draw a shadow behind the legend.
        title_fontsize=None  # int or {'xx-small', 'x-small', 'small', 'medium', 'large', 'x-large', 'xx-large'}
    )
    # figure
    rc('figure', dpi=120, figsize=(10, 6.18))  # 5.9, 2.8
    rc('savefig', dpi=300, transparent=True)  # 对于SVG格式，DPI参数不起作用了
    # rc(
    #     'figure.subplot',
    #     # The amount of width/height reserved for space between subfigures, expressed as a fraction of the average subfigure width/height.
    #     hspace= None,
    #     wspace=None,
    #     bottom=None,
    #     top=None,
    #     left=None,
    #     right=None)

    self.fontdict = {
        'family': 'Times New Roman',
        'style': 'normal',  # italic
        'weight': 'normal',
        'color': 'black',
        'size': 27,
    }
    self.fontdict_prop = {
        'family': 'simsun',  # Times New Roman
        'style': 'normal',  # italic
        'weight': 'normal',
        'size': 27,
    }  # [matplotlib字体设置]

  # ==================================================
  #                   Figure Configuration
  # ==================================================
  def figure_configuration(self,
                           figure=0,
                           labels='',
                           xlable='',
                           ylable='',
                           title='',
                           figsize=None,
                           margin=[None] * 6):
    fig = plt.figure(figure, figsize=figsize)
    # === 辅助显示层 ===
    plt.legend(labels=labels,
               framealpha=0.8,
               prop=self.fontdict_prop,
               loc='best')
    plt.title("{}".format(title), fontdict=self.fontdict)
    plt.xlabel("{}".format(xlable), fontdict=self.fontdict)
    plt.ylabel("{}".format(ylable), fontdict=self.fontdict)
    # === 图边界 ===
    # ax = plt.gca()  # [matplotlib 坐标轴比例]
    # ax.set_aspect(1)
    # 设置坐标上下限
    # plt.xlim(xlime_low, xlime_up)
    # plt.ylim(ylime_low, ylime_up)
    # 设置图框与边界距离，距离单位为图像宽度的比例
    plt.subplots_adjust(margin[0],
                        margin[1],
                        margin[2],
                        margin[3],
                        margin[4],
                        margin[5])  # [subplots_adjust]
    plt.margins(0.05)  # [margins]
    # return fig, ax

  def plot_show(self):
    plt.show()

  def save2figure(self, num=0):
    plt.savefig(self.path_save('figure', 'png', num))

  # ==================================================
  #                         Draw
  # ==================================================
  def set_point_annotate(self,
                         point=[0.0, 0.0],
                         arrow=[10, 10, 0.2],
                         lable="point",
                         showline=1,
                         size=50,
                         color='g',
                         fontsize=fontsize):
    '''
    对某个点添加注释
    '''
    x0 = point[0]
    y0 = point[1]
    plt.scatter(x0, y0, s=size, color=color, zorder=28)  # 描一个点
    plt.plot(
        [x0, x0],
        [y0, 0],
        'k--',  # k表示黑色，--表示虚线
        linewidth=2,
        alpha=showline,
        zorder=20)  # 通过两个点画一条线段
    plt.annotate(
        # The text of the annotation.
        lable,
        size=fontsize,
        zorder=29,
        # The point (x, y) to annotate. The coordinate system is determined by xycoords.
        xy=(x0, y0),
        # The position (x, y) to place the text at.
        xytext=(arrow[0], arrow[1]),
        # The coordinate system that xy is given in.
        xycoords='data',
        # The coordinate system that xytext is given in.
        textcoords='offset points',
        # The properties used to draw a FancyArrowPatch arrow between the positions xy and xytext.
        arrowprops=dict(arrowstyle="->",
                        connectionstyle="arc3, rad={}".format(arrow[2])),
        # The style of the fancy box.
        bbox=dict(boxstyle='round,pad=0.2',
                  facecolor='#ffffff',
                  edgecolor='#ffffff',
                  lw=1,
                  alpha=0.8))

  def draw_curve(self, data):
    plt.plot(data[0], data[1])

  def draw_point(self, p, size, marker):
    plt.scatter(p[0], p[1], s=size, marker=marker, zorder=10)

  def draw_line(self, p1_x, p1_y, p2_x, p2_y):
    plt.plot((p1_x, p2_x), (p1_y, p2_y), linewidth=1., zorder=10)

  # ==================================================
  #                        Figure
  # ==================================================
  # line: [10, 6.2], point: [10., 7.], curve: [5., 7.3],
  # circle: [10., 4.7], orign: [8.1, 6.2], initial_orign: [10, 4.5]
  def plot_fig(self, type, d, density=30, figure=0):
    labels = [r'跟踪轨迹', r'期望轨迹']
    lable_x = r'$\mathrm{X} \enspace (\mathrm{m})$'
    lable_y = r'$\mathrm{Y} \enspace (\mathrm{m})$'
    margin = [.09, .14, .95, .95, None,
              None]  # left, bottom, right, top,
    self.figure_configuration(figure,
                              labels=labels,
                              xlable=lable_x,
                              ylable=lable_y,
                              figsize=[10., 6.18],
                              margin=margin)
    if type == "error":
      self.draw_curve(d)
    if type == "curve":
      self.draw_curve(d[:2])
      self.draw_curve(d[2:])
    if type == "point":
      self.draw_point([d[0][::density], d[1][::density]], 20, 'o')
      self.draw_point([d[2][::density], d[3][::density]], 20, 'o')
    if type == "line":
      self.draw_line(d[0][::density], d[1][::density], d[2][::density],
                     d[3][::density])

  def plot_fig_error_rho(self, d, figure=1):
    lable_x = r'$\mathrm{time} \enspace (\mathrm{s})$'
    lable_y = r'$\rho_{e} \enspace (\mathrm{m})$'
    margin = [.11, .14, .95, .95, None,
              None]  # left, bottom, right, top,
    self.figure_configuration(figure,
                              #   labels=labels,
                              xlable=lable_x,
                              ylable=lable_y,
                              figsize=[10., 6.18],
                              margin=margin)
    plt.ylim(-10, 10)
    self.draw_curve(d)

  def plot_fig_error_phi(self, d, figure=2):
    lable_x = r'$\mathrm{time} \enspace (\mathrm{s})$'
    lable_y = r'$\phi_{e} \enspace (\mathrm{deg})$'
    margin = [.10, .14, .95, .95, None,
              None]  # left, bottom, right, top,
    self.figure_configuration(figure,
                              #   labels=labels,
                              xlable=lable_x,
                              ylable=lable_y,
                              figsize=[10., 6.18],
                              margin=margin)
    # plt.ylim(ylime_low, ylime_up)
    self.draw_curve(d)

  def plot_fig_error_theta(self, d, figure=3):
    lable_x = r'$\mathrm{time} \enspace (\mathrm{s})$'
    lable_y = r'$\theta_{e} \enspace (\mathrm{deg})$'
    margin = [.12, .14, .95, .95, None,
              None]  # left, bottom, right, top,
    self.figure_configuration(figure,
                              #   labels=labels,
                              xlable=lable_x,
                              ylable=lable_y,
                              figsize=[10., 6.18],
                              margin=margin)
    # plt.ylim(ylime_low, ylime_up)
    self.draw_curve(d)


# %% annotation
'''
[matplotlib 坐标轴比例](https://blog.csdn.net/weixin_43012215/article/details/106988287)
[scatter](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.scatter.html?highlight=scatter#matplotlib.axes.Axes.scatter)
[annotate](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.annotate.html?highlight=annotate#matplotlib.pyplot.annotate)
[FancyArrowPatch](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyArrowPatch.html#matplotlib.patches.FancyArrowPatch)
[FancyBboxPatch](https://matplotlib.org/stable/api/_as_gen/matplotlib.patches.FancyBboxPatch.html#matplotlib.patches.FancyBboxPatch)
[markers](https://matplotlib.org/stable/api/markers_api.html#module-matplotlib.markers)
[legend](https://matplotlib.org/stable/api/_as_gen/matplotlib.axes.Axes.legend.html#matplotlib.axes.Axes.legend)
[margins](`https://vimsky.com`/zh-tw/examples/usage/matplotlib-pyplot-margins-function-in-python.html)
[subplots_adjust](https://vimsky.com/examples/usage/matplotlib-pyplot-subplots_adjust-in-python.html)
[matplotlib字体设置](https://blog.csdn.net/Strive_For_Future/article/details/119319968)
[plt.legend()创建图例](https://zhuanlan.zhihu.com/p/111108841)
[Matplotlib 中英文及公式字体设置](https://zhuanlan.zhihu.com/p/118601703)
[matplotlib标签字体设置为斜体](https://www.csdn.net/tags/MtTaEgxsMTg1NzU0LWJsb2cO0O0O.html)
'''
