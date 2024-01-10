import matplotlib.pyplot as plt
from plot_figure import PlotFigure


class MyClass(PlotFigure):
  def __init__(self):
    super().__init__()

  def plot_data(self, x, y):
    fig, ax = self.create_plot()
    ax.plot(x, y)
    plt.show()


# 使用 MyClass 类进行绘图
x = [1, 2, 3, 4, 5]
y = [1, 4, 9, 16, 25]

obj = MyClass()
obj.plot_data(x, y)
