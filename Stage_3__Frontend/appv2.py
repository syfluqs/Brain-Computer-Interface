import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.animation as animation
from collections import deque

import scipy.ndimage.filters

import serial
import sys
from pynput import keyboard
import os
import threading
import math
import random
import time

from termcolor import *
import argparse
import sys
import logging

import PSD
import DWT
import neural_net

# Configuration Dictionary
# Values can be overriden by using command-line arguements
config = {
    'com_port' : '/dev/ttyUSB0',
    'baud_rate' : 9600,
    'sample_time_period' : 10, # in ms
    'dip_detect': False,
    'ch0_dip_threshold': 1000,
    'ch1_dip_threshold': 1000,
    'x_window' : 1000,
    'feature_size' : 100, # feature vector size will be 3*feature_size (for PSD)
    'ch0_gauss_filter_sigma' : 2,
    'ch1_gauss_filter_sigma' : 2,
    'sigma_delta' : 0.5,

    'feat_method' : 'psd',

    'nn_cluster_name' : 'c1',
    'nn_learning_rate' : 10,
    'nn_learning_reps' : 100,
    'nn_current_train_label' : np.array([0,1]),

    'ch0_label' : 'ch0',
    'ch1_label' : 'ch1',
    'ch0_gaussed_label' : 'ch0_gaussed',
    'ch1_gaussed_label' : 'ch1_gaussed',
    'app_description' : 'Thought Recognition stage3 module',
}

# Globals and pre-init
parser = argparse.ArgumentParser(description=config['app_description'])
serial_obj = None
fig = plt.figure(figsize=(18, 3))
# three plot, sch0, ch1 and fft
ax_ch0 = plt.subplot2grid((2,2),(0,0))
ax_ch0.set_xlim([0,config['x_window']])
ax_ch0.set_ylim([0,4095])
ax_ch1 = plt.subplot2grid((2,2),(1,0))
ax_ch1.set_xlim([0,config['x_window']])
ax_ch1.set_ylim([0,4095])
ax_fft = plt.subplot2grid((2,2),(0,1))
ax_nn = plt.subplot2grid((2,2),(1,1))
ax_nn.set_xlim([0,config['x_window']])
ax_nn.set_ylim([0,3.5])
if config['feat_method']=='psd':
    ax_fft.set_ylim([-2,5])
    ax_fft.set_xlim([0,300])
elif config['feat_method']=='dwt':
    ax_fft.set_ylim([0,30])
    ax_fft.set_xlim([0,30])
else:
    raise KeyError("invalid feat method")
ch0_line, = ax_ch0.plot([],[], linewidth=0.5, color="k", label=config['ch0_label'])
ch1_line, = ax_ch1.plot([],[], linewidth=0.5, color="k", label=config['ch1_label'])
ch0_line_gaussed, = ax_ch0.plot([],[], linewidth=2, color="r", label=config['ch0_gaussed_label'])
ch1_line_gaussed, = ax_ch1.plot([],[], linewidth=2, color="r", label=config['ch1_gaussed_label'])
ch0_fft_line, = ax_fft.plot([],[], linewidth=0.75, color="g", label='ch0 feat')
ch1_fft_line, = ax_fft.plot([],[], linewidth=0.75, color="b", label='ch1 feat')
ch0_nn_line, = ax_nn.plot([],[], linewidth=2, color="g", label='ch0 out')
ch1_nn_line, = ax_nn.plot([],[], linewidth=2, color="b", label='ch1 out')
# called after all axes are added
fig.tight_layout()
# plt.legend()
kboard = keyboard.Controller()
paused = False
anim = None
feat0 = None
feat1 = None
new_feature = True
nn_active = False
nn_train = False
input_counter = 0
if config['feat_method']=='psd':
    PSD.n = config['feature_size']  # number of elements in feature-vector
    nn0 = neural_net.neural_net(config['nn_cluster_name']+'_ch0_psd',(config['feature_size']*3,100,25,2),init='zeros')
    nn1 = neural_net.neural_net(config['nn_cluster_name']+'_ch1_psd',(config['feature_size']*3,100,25,2),init='zeros')
elif config['feat_method']=='dwt':
    nn0 = neural_net.neural_net(config['nn_cluster_name']+'_ch0_dwt',(30,25,10,2),init='zeros')
    nn1 = neural_net.neural_net(config['nn_cluster_name']+'_ch1_dwt',(30,25,10,2),init='zeros')
else:
    raise KeyError("invalid feat method")

ch0_dipped = 0
ch1_dipped = 0

# Deque containers to store incoming data
ch0_list = deque([-1]*config['x_window'])
ch1_list = deque([-1]*config['x_window'])
x_list = deque(np.arange(config['x_window'],0,-1))
x_list_np = np.array(x_list)
nn0_out = deque([-1]*config['x_window'])
nn1_out = deque([-1]*config['x_window'])



def check_args():
    ''' Parse command line arguments and update config dict accordingly
    '''
    global config
    parser.add_argument('-b','--baud-rate', help="Baud rate for serial communication. (default=%d)"%(config['baud_rate']),default=config['baud_rate'],nargs=1,metavar=('baud'))
    parser.add_argument('-p','--port', help="COM port to use for serial communication. (default=%s)"%(config['com_port']),default=config['com_port'],nargs=1,metavar=('port'))
    parser.add_argument('-t','--sample-time', help="Time period (in ms) for sample acquisition by ADC. (default=%d)"%(config['sample_time_period']),default=config['sample_time_period'],nargs=1,metavar=('time'))
    parser._optionals.title = 'Arguments'
    # start parsing arguments
    args = parser.parse_args()
    # Adjust config dict according to arguments
    if type(args.port)==list:
        config['com_port'] = args.port[0]
    config['baud_rate'] = args.baud_rate
    config['sample_time_period'] = args.sample_time
    return args

def serial_init():
    ''' Initialise serial communication and negotiate sampling time period
    '''
    global serial_obj
    global config
    try:
        logging.debug('Attempting to open {}'.format(config['com_port']))
        serial_obj = serial.Serial(config['com_port'], config['baud_rate'], timeout=100)
        logging.debug('Opened port {}'.format(serial_obj.name))
        # # Set sampling time period
        # to_send = int((config['sample_time_period']-1)/4).to_bytes(1,byteorder='big')
        # for i in range(3): # send 3 times just to be sure
        #     serial_obj.write(to_send)
    except:
        logging.debug('Error opening serial port')
        serial_obj = None

def serial_worker():
    ''' A separately threaded function which reads from serial port and fills the 
        deque
        Also process dips in signal which can be caused as artifacts from blinking
    '''
    global serial_obj
    global ch0_list, ch1_list
    global ch0_dipped, ch1_dipped
    global config, input_counter
    if serial_obj!=None:
        while (1):
            if not paused:
                while (serial_obj.read()!=b';'):
                    pass
                read_data = serial_obj.read(8)
                # print(read_data,)
                # temp = list(read_data)
                # ch0_val = int(temp[2]) + 256*int(temp[1])
                # ch1_val = int(temp[5]) + 256*int(temp[4])
                # print(read_data)
                ch0_val = int(read_data[0:4])
                ch1_val = int(read_data[4:8])
                # print(ch0_val, ch1_val)
                # print(ch0_val,ch1_val)
                ch0_list.pop()
                ch1_list.pop()
                ch0_list.appendleft(ch0_val)
                ch1_list.appendleft(ch1_val)
                if input_counter==config['x_window']-1:
                    logging.debug(color.cyan("Starting neural net"))
                if input_counter<config['x_window']:
                    input_counter += 1
                # Detect signal going down towards dip
                if config['dip_detect']:
                    if (ch0_val < config['ch0_dip_threshold'] and not ch0_dipped):
                        ch0_dipped = 1
                        try:
                            dip_down_callback(0)
                        except NameError:
                            pass
                        logging.debug("Dip down on {}".format(config['ch0_label']))
                    elif (ch1_val < config['ch1_dip_threshold'] and not ch1_dipped):
                        ch1_dipped = 1
                        try:
                            dip_down_callback(1)
                        except NameError:
                            pass
                        logging.debug("Dip down on {}".format(config['ch1_label']))
                    # Detect signal coming back up from dip
                    if (ch0_val > config['ch0_dip_threshold'] and ch0_dipped):
                        ch0_dipped = 0
                        try:
                            dip_up_callback(0)
                        except NameError:
                            pass
                        logging.debug("Dip up on {}".format(config['ch0_label']))
                    elif (ch1_val > config['ch1_dip_threshold'] and ch1_dipped):
                        ch1_dipped = 0
                        try:
                            dip_up_callback(1)
                        except NameError:
                            pass
                        logging.debug("Dip up on {}".format(config['ch1_label']))
            # sleep to sync incoming and outgoing datarates
            time.sleep(config['sample_time_period']/1000)
    else:
        # serial object not defined, running in debug mode
        logging.debug('serial object not defined, running in debug mode')
        while (1):
            if not paused:
                ch0_list.pop()
                ch1_list.pop()
                ch0_list.appendleft(math.floor(random.random()*4096))
                ch1_list.appendleft(math.floor(random.random()*4096))
                # sleep to sync incoming and outgoing datarates
                time.sleep(config['sample_time_period']/1000)

def plot_init():
    ''' Set initial data to blank, or else a spike is observed in plot
    '''
    global ch0_line, ch1_line
    ch0_line.set_data([],[])
    ch1_line.set_data([],[])
    return (ch0_line,ch1_line)

def plot_worker(frame):
    ''' Calculate ch0 and ch1 filtered data. Then calculate feature vectors according to the method 
        selected
    
        Raises:
            KeyError -- Error raised if feat_method set wrong in config dict
    '''
    global ch0_line, ch1_line, ch0_line_gaussed, ch1_line_gaussed, ch0_fft_line, ch1_fft_line
    global ch0_list, ch1_list
    global x_list, new_feature
    global feat0, feat1
    if not paused:
        ch0_line.set_data(x_list,ch0_list)
        ch1_line.set_data(x_list,ch1_list)
        # Gaussian filtering
        gauss_inp_ch0 = np.array(ch0_list)
        filtered_ch0 = scipy.ndimage.filters.gaussian_filter1d(gauss_inp_ch0, sigma=config['ch0_gauss_filter_sigma'])
        gauss_inp_ch1 = np.array(ch1_list)
        filtered_ch1 = scipy.ndimage.filters.gaussian_filter1d(gauss_inp_ch1, sigma=config['ch1_gauss_filter_sigma'])
        ch0_line_gaussed.set_data(x_list_np,filtered_ch0)
        ch1_line_gaussed.set_data(x_list_np,filtered_ch1)
        # ==========================================================================================
        # # fft plot
        # N = np.arange(config['x_window'])
        # fft0 = np.fft.fft(filtered_ch0)
        # fft1 = np.fft.fft(filtered_ch1)
        # freq = np.fft.fftfreq(config['x_window'],d=(config['sample_time_period']/1000))*2000
        # ch0_fft_line.set_data(freq, fft0.real)
        # ch1_fft_line.set_data(freq, fft1.real)
        # ==========================================================================================
        if config['feat_method']=='psd':
            # PSD extract
            feat0 = PSD.PSD_extractor(ch0_list)
            feat1 = PSD.PSD_extractor(ch1_list)
            N = np.arange(config['feature_size']*3)
            ch0_fft_line.set_data(N, np.array(feat0))
            ch1_fft_line.set_data(N, np.array(feat1))
        # ==========================================================================================
        elif config['feat_method']=='dwt':
            # DWT extract
            feat0 = DWT.DWT_extractor(ch0_list)
            feat1 = DWT.DWT_extractor(ch1_list)
            N = np.arange(len(feat0))
            ch0_fft_line.set_data(N, np.array(feat0))
            ch1_fft_line.set_data(N, np.array(feat1))
        # ==========================================================================================
        else:
            raise KeyError("invalid feat method")
        new_feature = True
        time.sleep(config['sample_time_period']/1000)
    return ch0_line, ch1_line, ch0_line_gaussed, ch1_line_gaussed, ch0_fft_line, ch1_fft_line

def neural_net_worker():
    ''' Start neural net feed forward if the x_window is filled with incoming data
        Backpropagate features extracted from current x_window when key pressed
    '''
    global nn0, nn1, feat0 ,feat1
    global config
    global nn0_out, nn1_out
    global new_feature, nn_train, input_counter
    global x_list, ch0_nn_line, ch1_nn_line
    while (1):
        if new_feature and (not paused) and input_counter==config['x_window']:
            o0 = nn0.feedforward(feat0)[-1]
            o1 = nn1.feedforward(feat1)[-1]
            nn0_out.pop()
            nn1_out.pop()
            n0__ = o0[1]+2*o0[0]
            n1__ = o1[1]+2*o1[0]
            print(o0,o1,n0__,n1__)
            nn0_out.appendleft(n0__)
            nn1_out.appendleft(n1__)
            new_feature = False
            if nn_train:
                logging.debug(color.green("Training net with current data set"))
                nn0.backpropagate(feat0, config['nn_current_train_label'], config['nn_learning_rate'], config['nn_learning_reps'])
                nn1.backpropagate(feat1, config['nn_current_train_label'], config['nn_learning_rate'], config['nn_learning_reps'])
                nn_train = False
                logging.debug(color.green("Training done..."))
            # plot neural net output
            ch0_nn_line.set_data(x_list, np.array(nn0_out))
            ch1_nn_line.set_data(x_list, np.array(nn1_out))
        else:
            pass

def key_listener_worker():
    # plt.show()
    # Engage Keyboard listener
    with keyboard.Listener(on_press=key_press_callback, on_release=key_release_callback) as listener:
        listener.join()

def dip_down_callback(stat):
    if (stat==0):
        # Dip in channel 0
        # kboard.press(keyboard.Key.space)
        # kboard.release(keyboard.Key.space)
        pass
    elif (stat==1):
        # Dip in channel 1
        kboard.press(keyboard.Key.space)
        kboard.release(keyboard.Key.space)
        pass

def key_press_callback(key):
    pass

def key_release_callback(key):
    global paused, anim, config, nn_train, paused
    if key==keyboard.Key.space:
        paused = not paused
        if paused:
            # anim.event_source.stop()
            logging.debug("Plotting paused")
        else:
            # anim.event_source.start()
            logging.debug("Plotting resumed")
    elif key==keyboard.KeyCode.from_char('+'):
        config['ch0_gauss_filter_sigma'] += config['sigma_delta']
        config['ch1_gauss_filter_sigma'] += config['sigma_delta']
        logging.debug('Increased sigma to {}'.format(config['ch0_gauss_filter_sigma']))
    elif key==keyboard.KeyCode.from_char('-'):
        if config['ch0_gauss_filter_sigma']>config['sigma_delta']:
            config['ch0_gauss_filter_sigma'] -= config['sigma_delta']
            config['ch1_gauss_filter_sigma'] -= config['sigma_delta']
            logging.debug('Decreased sigma to {}'.format(config['ch0_gauss_filter_sigma']))
    elif key==keyboard.Key.f2:
        logging.debug(yellow("Toggling dip detection"))
        config['dip_detect'] = not config['dip_detect']
    elif key==keyboard.Key.f8:
        nn_train = True


if __name__ == '__main__':
    try:
        # Configure loggin level
        logging.basicConfig(level=logging.DEBUG,format='[%(levelname)s] (%(threadName)-10s) %(message)s',)

        # check command line arguments passed
        args = check_args()
        logging.debug("Arguments = "+color.bold(color.green(str(args))))

        # Initialise serial communication
        serial_init()

        # Start serial worker thread
        serial_thread = threading.Thread(target = serial_worker)
        serial_thread.setName('serial_worker')
        serial_thread.start()
        logging.debug(color.darkcyan("serial_worker started"))
        
        # Plot animator
        anim = animation.FuncAnimation(fig, plot_worker, init_func=plot_init, frames=config['x_window'], 
            interval=config['sample_time_period'], blit=True)
            # interval=config['sample_time_period'])
        logging.debug(color.darkcyan("plot animator instantiated"))

        # Show animated plot
        # ax_ch0.legend(loc="upper right")
        # ax_ch1.legend(loc="upper right")
        ax_fft.legend(loc="upper right")
        
        # Start keyboar listener thread
        listener_thread = threading.Thread(target = key_listener_worker)
        listener_thread.setName('listener_thread')
        listener_thread.start()
        logging.debug(color.darkcyan("keyboard listener started"))

        # Start neural net thread
        nn_thread = threading.Thread(target = neural_net_worker)
        nn_thread.setName('neural_net_worker')
        nn_thread.start()
        logging.debug(color.darkcyan("neural net worker started"))

        # matplotlib main loop
        plt.show()

        

    except KeyboardInterrupt:
        # quit
        if serial_obj!=None:
            serial_obj.close()
        plt.close()
        sys.exit(0)