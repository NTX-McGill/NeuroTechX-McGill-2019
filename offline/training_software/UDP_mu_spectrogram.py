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

sampling_freq = 250
lim_hz = 40         # the upper frequency limit we want to plot on our spectrogram


client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP protocol
client_socket.bind(('127.0.0.1', 12345))

fig = plt.figure(figsize=(10,10))
plt.yscale('log')
arr = []
specgrams = []
set_1 = [0,1,2,3]   # the first set of electrodes we want to plot
set_2 = [4,5,6,7]   # the second set of electrodes we want to plot
i = 0

def animate(args,client_socket, arr,specgrams, lim_hz, set_1, set_2):
    data = client_socket.recvfrom(12000)
    fig.clear()
    ffts = np.array(ast.literal_eval(data[0].decode("utf-8"))['data'])
    mu = np.mean(ffts[:, 6:13], axis=1)
    
    arr.append(mu)
    if len(arr) > 50:
        arr.pop(0)
    
    plt.subplot(321)
    plt.ylim(0.1,10)
    plt.bar([i+1 for i in range(8)],mu)
    
    plt.subplot(322)
    plt.bar(['left', 'right'], [np.mean(mu[set_1]), np.mean(mu[set_2])])
    
    plt.subplot(323)
    plt.ylim(0.1,10)
    arr_ = np.array(arr)
    plt.plot(np.mean(arr_[:,set_1], axis=1))
    plt.plot(np.mean(arr_[:,set_2], axis=1))
    
    
    PSD = np.log10(np.abs(ffts[:, :lim_hz]) + 1)
    specgrams.append([PSD[0], np.mean(PSD[set_1], axis=0), np.mean(PSD[set_2], axis=0)])
    if len(specgrams) > 30:
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

anim = animation.FuncAnimation(fig, animate, fargs=[client_socket, arr, specgrams, lim_hz, set_1, set_2],interval=200)
plt.show()
#client_socket.close()
