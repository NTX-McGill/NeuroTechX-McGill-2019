#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 15 17:01:24 2019

@author: jenisha
"""

import numpy as np

import math
import cmath

class RealTimeProcessWavelet():
    
    def __init__(self, input_raw, name_channels=["C3,C4"], num_channels = 2):
        """
        The input array should be 
        a channel x window data
        
        """
        self.sample_rate = 250 #Default sampling rate for OpenBCI
        self.name_channels = name_channels
        self.nyquist = math.floor(self.sample_rate/2) #Nyquist frequency
        
        self.time_range = 0.05 #should account for lag between acquisition an process
        self.data = data

        self.num_channels = 2
        
        self.crop = math.floor(0.25* self.sample_rate)
        
        self.tr = (np.linspace(0, len(input_raw)/self.sample_rate, len(input_raw)))[self.crop:-self.crop]
    
        
    def wavelet(self, frequency):
        
        self.frequency = frequency
        self.time_range = 0.05
        
        #Order
        order_function = np.flip(-1*(np.floor(np.logspace(np.log10(1), np.log10(11),self.nyquist ))-13))
        self.order = order_function[self.frequency]
        self.time = np.arange(-self.time_range, self.time_range, 1/self.sample_rate)
        
        self.std_gaussian = self.order/(2*math.pi*self.frequency )
        self.sineWave = cmath.e**(1j*self.frequency*2*math.pi*self.time)
        self.gauss = cmath.e**((-self.time**2)/(2*self.std_gaussian**2))
        
        
        self.wavelet =  self.sineWave*self.gauss
        

    def convolve_wavelet_all_channels(self):
        
        
        try:
            fData = np.ma.zeros((np.ma.size(self.data, 0), np.ma.size(self.data, 1)),
                                dtype=complex)
            power = np.ma.zeros((np.ma.size(self.data, 0), np.ma.size(self.data, 1)))
            for i in range(self.num_channels):
                fData[:, i] = np.convolve(self.wavelet, self.data[:, i], mode='same')
                power[:, i] = fData[:, i]*np.conj(fData[:, i])
                power = power.real
        except (ValueError, IndexError) as e:
            nCh = 1
            fData = np.convolve(self.wavelet, self.data, mode='same')
            power = fData*np.conj(fData)
            power = power.real
        #return fData, power
        self.fData= fData[self.crop:-self.crop]
        self.power = power[self.crop:-self.crop]
    
    def all_power_plots(self):

        plt.figure()
        try:
            for i in range(self.num_channels):
                plt.subplot(self.num_channels, 1, i+1)
                plt.plot(self.tr, self.power[:, i])
                plt.grid(True)
            if (i == 0):
                plt.title('Power at ' + str(self.frequency)+' Hz')
                plt.ylabel('Ch ' + str(i+1) + 'Power ' + r'($\mu$V $^2$)')
            if (i == self.num_channels-1):
                plt.xlabel('Time (sec)')
        except (ValueError, IndexError) as e:
            nCh = 1
            plt.plot(self.tr, self.power)
            plt.grid(True)
            plt.title('Power at ' + str(self.freq)+' Hz')
            plt.ylabel('Ch ' + str(1) + 'Power ' + r'($\mu$V $^2$)')
            plt.xlabel('Time (sec)')
    
      
path='/Users/jenisha/Desktop/NeuroTechX-McGill-2019/offline/data/March_11/'
fname= path +  '1_JawRest_JawRightClench_10s.txt'

raw_eeg_data = np.loadtxt(fname, delimiter=',',skiprows=7, usecols=[1,8])
data= raw_eeg_data[0:500,::]
data2= raw_eeg_data[5000:5500,::]

t = RealTimeProcessWavelet(data)
t.wavelet(10)
t.convolve_wavelet_all_channels()
t.all_power_plots()

t2 = RealTimeProcessWavelet(data2)
t2.wavelet(10)
t2.convolve_wavelet_all_channels()
t2.all_power_plots()
        
        
        

        
        
        