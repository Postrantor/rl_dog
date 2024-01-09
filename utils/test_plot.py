import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


class PlotConfig:
  def __init__(self, xlim=(0, 10), ylim=(0, 1), linestyle='-', linewidth=1.0):
    self.xlim = xlim
    self.ylim = ylim
    self.linestyle = linestyle
    self.linewidth = linewidth


class CustomPlot:
  def __init__(self, config):
    self.config = config
    self.fig, self.ax = plt.subplots()
    self.line, = self.ax.plot([], [], linestyle=self.config.linestyle, linewidth=self.config.linewidth)
    self.ax.set_xlim(self.config.xlim)
    self.ax.set_ylim(self.config.ylim)
    self.x_data = []
    self.y_data = []

  def update_line(self, data):
    self.x_data.append(data[0])
    self.y_data.append(data[1])
    self.line.set_data(self.x_data, self.y_data)
    self.ax.relim()
    self.ax.autoscale_view()

  def init_animation(self):
    self.line.set_data([], [])
    return self.line,

  def animate(self, i):
    x = i / 10.0
    y = np.sin(x)
    self.update_line([x, y])
    return self.line,

  def show(self):
    anim = FuncAnimation(self.fig, self.animate, init_func=self.init_animation, blit=True)
    plt.show()


# 创建一个配置对象
config = PlotConfig(xlim=(0, 10), ylim=(0, 1), linestyle='-', linewidth=1.0)

# 创建一个自定义绘图对象
plot = CustomPlot(config)

# 显示曲线图
plot.show()
