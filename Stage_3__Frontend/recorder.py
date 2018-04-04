import numpy as np
import serial
import sys
from pynput import keyboard
import os
import threading
import math
import random
import time

com_port = '/dev/ttyUSB0'

config = {
    'baud_rate' : 115200,
    'sample_time_period' : 10
}


# logging.debug('Attempting to open {}'.format(com_port))
ser = serial.Serial(com_port, config['baud_rate'], timeout=100)
# logging.debug('Opened port {}'.format(ser.name))
# Set sampling time period
ser.write(int((config['sample_time_period']-1)/4).to_bytes(1,byteorder='big'))

file = 5
rec = False

data0 = []
data1 = []

os.chdir('data')

def end_record():
    global data0, data1, file
    print(data0)
    print(data1)
    np.savez('sample_'+str(file), data0=np.array(data0), data1=np.array(data1))
    data0 = []
    data1 = []
    ch0_read = 0
    ch1_read = 0
    file += 1

def on_press(key):
    pass
def on_release(key):
    global rec
    if key==keyboard.Key.delete:
        rec = not rec
        if not rec:
            print("saving record number "+str(file))
            end_record()
        else:
            print("started recording number "+str(file))

def serial_worker():
    # global ser
    global ch0_read, ch1_read
    global rec
    while (ser.read()!=b'}'):
            pass
    while (1):
        if rec:
            temp = list(ser.read(7))
            print(temp)
            # (ch0_read, ch1_read) = (math.floor(random.random()*50),math.floor(random.random()*50))
            ch0_read = int(temp[2]) + 256*int(temp[1])
            ch1_read = int(temp[5]) + 256*int(temp[4])
            data0.append(ch0_read)
            data1.append(ch1_read)
            time.sleep(config['sample_time_period']/1000)

try:
    # Starting serial worker thread
    serial_thread = threading.Thread(target = serial_worker)
    serial_thread.setName('serial_worker')
    serial_thread.start()
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:  # start listening for keys
        listener.join()

    
        
except KeyboardInterrupt:
    ser.close()
    if rec:
        end_record()
