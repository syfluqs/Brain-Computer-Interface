__author__ = "Haider Zulfiqar"

import numpy as np
import matplotlib.pyplot as plt
import time
from drawnow import *
from pynput import keyboard
from threading import Thread, Event

event = Event()
y1 = []
y2 = []
x = []
r = []
plt.ion()  # make the graph interactive
c = 0  # initialize counter


def record():   # function to record 10 data elements on pressing r
    while len(x) < 10:
        # In order to synchronise data, we make use of Thread.Event()
        event.wait()    # waits for flag to be set
        x.append(y1[c])
        event.clear()   # clears flag
    print(x)
    x.clear()


def on_press(key):
    pass


def on_release(key):
    if key == keyboard.KeyCode(char='r'):
        record()


def makefig():
    plt.plot(y1, 'g-', y2, 'c-')  # plot the graph
    plt.ylim(-1, 11)  # set the axis limits to avoid auto-scale
    plt.xlim(0, 50)
    plt.grid(True)


def data_plot():  # create plot function for thread
    while True:
        global c  # call global c
        y1.append(ch0_read)  # append channel 0 data into queue 1
        y2.append(ch1_read)  # append channel 1 data into queue 2

        c += 1  # increment counter
        event.set()   # sets flag
        if c > 51:  # making data live
            y1.pop(0)
            y2.pop(0)
        drawnow(makefig)
        plt.pause(0.000001)


t1 = Thread(target=data_plot)  # create thread
t1.start()  # start thread
with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:  # start listening for keys
    listener.join()
