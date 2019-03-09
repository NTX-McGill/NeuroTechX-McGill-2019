#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 28 18:07:36 2019

@author: marley
"""

import socket, time, struct
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import ast
import fcntl, os
import errno

''' CONFIGURABLE '''
plot_specgrams = True   # whether or not to plot the spectrograms
lim_hz = 40             # the upper frequency limit we want to plot on our spectrogram
single_electrode = 0    # the single electrode we want to plot
set_1 = [0,1,2,3]       # the set of electrodes for right imagery (left brain, C1 C3 etc.)
set_2 = [4,5,6,7]       # the set of electrodes for left imagery (right brain, C2 C4 etc.)
spec_length = 30        # length of spectrogram (multiply by ~0.4 to get units in seconds,
                        # e.g. spec_length of 30 gives 0.4 * 30 = 12 seconds in spectrogram)
                        # I found 30 to be best since it tends to slow down with more to plot
port = 12347            # which port to listen on
''' DO NOT EDIT PAST THIS BLOCK '''

def animate(args, client_socket, arr, plot_specgrams, specgrams, lim_hz, single_electrode, set_1, set_2, spec_length):
    data = None
    #x = 0
    while True:
        try:
            data = client_socket.recv(12000)
            #x +=1
        except socket.error as e:
            #if e.args[0] == errno.EAGAIN:
            #    print(x)
            break
    if data:
        fig.clear()
        ffts = np.array(ast.literal_eval(data.decode("utf-8"))['data'])
        mu = np.mean(ffts[:, 6:13], axis=1)
        
        arr.append(mu)
        if len(arr) > spec_length:
            arr.pop(0)
        
        plt.subplot(321)
        plt.ylim(0.1,2)
        plt.bar([i+1 for i in range(8)],mu)
        
        plt.subplot(322)
        plt.bar(['left', 'right'], [np.mean(mu[set_1]), np.mean(mu[set_2])])
        
        plt.subplot(323)
        # plt.ylim(0.1,5)
        arr_ = np.array(arr)
        plt.plot(np.mean(arr_[:,set_1], axis=1), label='left')
        plt.plot(np.mean(arr_[:,set_2], axis=1), label='right')
        plt.legend(loc='upper right')
        
        if plot_specgrams:
            PSD = np.log10(np.abs(ffts[:, :lim_hz]) + 1)
            specgrams.append([PSD[single_electrode], np.mean(PSD[set_1], axis=0), np.mean(PSD[set_2], axis=0)])
            if len(specgrams) > spec_length:
                specgrams_ = np.array(specgrams)
                specgram1 = specgrams_[:,0,:]
                specgram2 = specgrams_[:,1,:]
                specgram3 = specgrams_[:,2,:]
                plt.subplot(324)
                plt.pcolor([i for i in range(len(specgram1))],[i for i in range(len(specgram1[0]))], specgram1.T)
                plt.subplot(325)
                plt.pcolor([i for i in range(len(specgram2))],[i for i in range(len(specgram2[0]))], specgram2.T)
                plt.subplot(326)
                plt.pcolor([i for i in range(len(specgram3))],[i for i in range(len(specgram3[0]))], specgram3.T)
                specgrams.pop(0)
                
sampling_freq = 250
arr, specgrams = [], []

client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)    # UDP protocol
client_socket.bind(('127.0.0.1', port))
fcntl.fcntl(client_socket, fcntl.F_SETFL, os.O_NONBLOCK)            # make socket non-blocking


fig = plt.figure(figsize=(10,10))
plt.yscale('log')
anim = animation.FuncAnimation(fig, animate, fargs=[client_socket, arr, plot_specgrams, specgrams, lim_hz, single_electrode, set_1, set_2, spec_length],interval=300)
plt.show()
#client_socket.close()
