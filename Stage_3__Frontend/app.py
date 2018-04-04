import sys
import serial
import threading
import logging
import numpy as np
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pyautogui import press
import random, math

config = {
    'baud_rate' : 115200,
    'sample_time_period' : 10
}

ch0_read = 0
ch1_read = 0
ser = None
temp = None
com_port = None
pressed = 0

threshold = 1000

class Scope(object):
    def __init__(self, ax, maxt=200, dt=10):
        self.ax = ax
        self.dt = dt
        self.maxt = maxt
        self.tdata = [0]
        self.ydata = [0]
        self.line = Line2D(self.tdata, self.ydata)
        self.ax.add_line(self.line)
        # self.ax.set_ylim(0, 4096)
        self.ax.set_ylim(0, 256)
        self.ax.set_xlim(0, self.maxt)

    def update(self, y):
        lastt = self.tdata[-1]
        if lastt > self.tdata[0] + self.maxt:  # reset the arrays
            self.tdata = [self.tdata[-1]]
            self.ydata = [self.ydata[-1]]
            self.ax.set_xlim(self.tdata[0], self.tdata[0] + self.maxt)
            self.ax.figure.canvas.draw()

        t = self.tdata[-1] + self.dt
        self.tdata.append(t)
        self.ydata.append(y)
        self.line.set_data(self.tdata, self.ydata)
        return self.line,


def emitter(p=0.03):
    global ch0_read, ch1_read

    while True:
        yield ch0_read


fig, ax = plt.subplots()
scope = Scope(ax)

logging.basicConfig(level=logging.DEBUG,format='[%(levelname)s] (%(threadName)-10s) %(message)s',)

def serial_init():
    global ser
    logging.debug('Attempting to open {}'.format(com_port))
    ser = serial.Serial(com_port, config['baud_rate'], timeout=100)
    logging.debug('Opened port {}'.format(ser.name))
    # Set sampling time period
    ser.write(int((config['sample_time_period']-1)/4).to_bytes(1,byteorder='big'))


def plt_dynamic(x, y, ax, colors=['b']):
    for color in colors:
        ax.plot(x, y, color)
    fig.canvas.draw()


def serial_worker():
    global ser
    global temp
    global ch0_read, ch1_read
    global pressed
    while (1):
        while (ser.read()!=b'}'):
            pass
        temp = list(ser.read(7))
        ch0_read = int(temp[2]) + 256*int(temp[1])
        ch1_read = int(temp[5]) + 256*int(temp[4])
        # if (ch0_read < threshold and not pressed):
        #     pressed = 1
        #     press(' ')
        #     print("pressed")
        # if (ch0_read > threshold and pressed):
        #     pressed = 0

if __name__=="__main__":

    try:
        if (len(sys.argv)>1):
            com_port = sys.argv[1]
        else:
            raise Exception("COM Port not provided\nProvide COM port like \'python {} /dev/ttyUSB0\'".format(sys.argv[0]))

        # serial_init()

        # Starting serial worker thread
        # serial_thread = threading.Thread(target = serial_worker)
        # serial_thread.setName('serial_worker')
        # serial_thread.start()

        ani = animation.FuncAnimation(fig, scope.update, emitter, interval=100,blit=True)


        plt.show()
        # while (1) :
            # print(temp,ch0_read,ch1_read)
            # pass


    except KeyboardInterrupt:
        # quit
        ser.close()
        sys.exit()