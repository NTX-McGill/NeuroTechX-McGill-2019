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

sampling_freq = 250
NFFT = 256
bin_size = sampling_freq/NFFT  # mu frequencies are between 7 and 13 Hz


client_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM) # UDP protocol
client_socket.bind(('127.0.0.1', 12345))
data = client_socket.recvfrom(1024)

fig = plt.figure()
plt.yscale('log')
arr = []
specgram = []

def animate(args,client_socket, arr,specgram):
    data = client_socket.recvfrom(1024)
    fig.clear()
    data = data[0].decode("utf-8")[23:]
    data = np.fromstring(data, dtype=np.float, sep=',' )
    mean = np.mean(data[6:13])
    arr.append(mean)
    if len(arr) > 50:
        arr.pop(0)
    
    plt.subplot(311)
    plt.ylim(0.1,15)
    plt.bar(0,mean)
    
    plt.subplot(312)
    plt.ylim(0.1,15)
    plt.plot(arr)
    
    
    PSD = np.log10(np.abs(data[:60]) + 1)
    specgram.append(PSD)
    if len(specgram) > 50:
        plt.subplot(313)
        plt.pcolor([i for i in range(len(specgram))],[i for i in range(len(specgram[0]))], np.array(specgram).T)
        specgram.pop(0)

anim = animation.FuncAnimation(fig, animate, fargs=[client_socket, arr, specgram],interval=200)
plt.show()
#client_socket.close()
