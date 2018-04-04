import psutil
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from collections import deque

import scipy.ndimage.filters
import scipy.interpolate


fig = plt.figure()
ax = plt.axes(xlim=(0, 200), ylim=(0, 100))
line, = ax.plot([],[], color="b", label="cpu")
mean_line, = ax.plot([],[], linestyle="--", color="k",label="mean")
filter_line, = ax.plot([],[], linewidth=2, color="r", label="gauss filter")
interp_line, = ax.plot([],[], linewidth=1.5, color="purple", label="spline")

plt.legend()
y_list = deque([-1]*200)
x_list = deque(np.arange(200,0,-1))


def init():
    line.set_data([],[])
    return line,


def animate(i):
    y_list.pop()
    y_list.appendleft(psutil.cpu_percent(None,False))
    line.set_data(x_list,y_list)
    x = np.array(x_list)
    y = np.array(y_list)
    filtered = scipy.ndimage.filters.gaussian_filter1d(y, sigma=4)

    mean_line.set_data(x, np.ones_like(x)*y.mean())
    filter_line.set_data(x,filtered)
    try:
        tck = scipy.interpolate.splrep(x[::-1], y[::-1], s=50000)
        interpolated = scipy.interpolate.splev(x[::-1], tck, der=0)
        interp_line.set_data(x,interpolated[::-1])
    except:
        pass

    return line,filter_line,mean_line,interp_line,

anim = animation.FuncAnimation(fig, animate, init_func=init,
                           frames=200, interval=100, blit=True)

plt.show()