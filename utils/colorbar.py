# -*- coding:utf-8 -*-

import numpy as np
import pylab as plt
from matplotlib import cm, colors, colorbar

if __name__ == '__main__':
  # Colors是一些自选颜色列表
  # Colors=('#DDDDFF','#7D7DFF','#0000C6','#000079','#CEFFCE','#28FF28','#007500','#FFFF93','#8C8C00','#FFB5B5','#FF0000','#CE0000','#750000')
  # cs=m.contourf(xi, yi, z, colors=Colors, levels=levels, extend='both')  # 这里m是一个basemap实例
  fig = plt.figure(figsize=(3, 8))
  ax3 = fig.add_axes([0.3, 0.2, 0.2, 0.5])  # 四个参数分别是左、下、宽、长

  cmap = cm.Spectral_r
  norm = colors.Normalize(vmin=1.3, vmax=2.5)
  bounds = [round(elem, 2) for elem in np.linspace(1.3, 2.5, 14)]
  cb3 = colorbar.ColorbarBase(ax3, cmap=cmap,
                              norm=norm,
                              # to use 'extend', you must specify two extra boundaries:
                              boundaries=[1.2] + bounds + [2.6],
                              extend='both',
                              ticks=bounds,  # optional
                              spacing='proportional',
                              orientation='vertical')
  plt.show()

'''
[matplotlib自定义colorbar颜色条](https://blog.csdn.net/weixin_45342712/article/details/95965398)
'''
