import sys
import serial
import threading
import matplotlib.pyplot as plt
import numpy as np
import logging

config = {
    'baud_rate' : 115200,
    'sample_time_period' : 100
}

ch0_read = 0
ch1_read = 0
ser = None
temp = None
com_port = None

logging.basicConfig(level=logging.DEBUG,format='[%(levelname)s] (%(threadName)-10s) %(message)s',)

def serial_init():
    global ser
    logging.debug('Attempting to open {}'.format(com_port))
    ser = serial.Serial(com_port, config['baud_rate'], timeout=100)
    logging.debug('Opened port {}'.format(ser.name))
    # Set sampling time period
    ser.write(int((config['sample_time_period']-1)/4).to_bytes(1,byteorder='big'))

hl, = plt.plot([], [])

def plt_dynamic(x, y, ax, colors=['b']):
    for color in colors:
        ax.plot(x, y, color)
    fig.canvas.draw()


def serial_worker():
    global ser
    global temp
    global ch0_read, ch1_read
    while (1):
        while (ser.read()!=b'{'):
            pass
        temp = ser.read(2)
        ch0_read = temp[0] + 16*temp[1]
        while (ser.read() != b','):
            pass
        temp = ser.read(2)
        ch1_read = temp[0] + 16*temp[1]


if __name__=="__main__":

    try:
        if (len(sys.argv)>1):
            com_port = sys.argv[1]
        else:
            raise Exception("COM Port not provided\nProvide COM port like \'python {} /dev/ttyUSB0\'".format(sys.argv[0]))

        serial_init()

        # Starting serial worker thread
        serial_thread = threading.Thread(target = serial_worker)
        serial_thread.setName('serial_worker')
        serial_thread.start()

        fig,ax = plt.subplots(1,1)
        ax.set_xlabel('X') ; ax.set_ylabel('Y')
        ax.set_xlim(0,360) ; ax.set_ylim(-1,1)
        xs, ys = [], []

    except KeyboardInterrupt:
        # quit
        ser.close()
        sys.exit()