"""

"""
from IPython import get_ipython
import matplotlib.pyplot as plt


class Monitor():

    def __init__(
        self,
        x, y,
        title: str = None,
        xlim: float = None,
        ylim: float = None,
        plot_kwargs: dict = {},
        xlabel: str = "",
        ylabel: str = "",
        pause: float = 0.05,
    ) -> None:

        ipython = get_ipython()
        if ipython is None:
            raise Exception("There is no IPython available. ")
        ipython.run_line_magic('matplotlib', 'qt6')

        self.title = title
        self.xlim = xlim
        self.ylim = ylim
        self.fig, \
            self.ax = plt.subplots()
        self.pause = pause

        self.ax.set_title(self.title)
        self.line = self.ax.plot(
            x, y, **plot_kwargs)[0]
        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        if self.xlim:
            self.ax.set_xlim(self.xlim)
        if self.ylim:
            self.ax.set_ylim(self.ylim)
    
    def update(self, x = None, y = None):
        if not x and not y:
            raise ValueError("x and y must be not None")

        if x: self.line.set_xdata(x)

        if y: self.line.set_ydata(x)

        if self.xlim:
            self.ax.set_xlim(self.xlim)
        elif len(x)>0:
            self.ax.set_xlim(min(x), max(x))

        if self.ylim:
            self.ax.set_ylim(self.ylim)
        elif len(y)>0:
            self.ax.set_ylim(min(y), max(y))

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        plt.pause(self.pause)