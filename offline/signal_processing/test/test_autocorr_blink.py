#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 00:48:10 2019

@author: jenisha
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.signal import find_peaks_cwt,detrend

import matplotlib.mlab as mlab

path='../data/March30_420/'
fname= path +  '1_2COMBINED_420_rest_20s_blink3s_jawclenchA3s_rest20s-2019-3-30-18-0-3.txt'

raw_eeg_data = np.loadtxt(fname, delimiter=',',skiprows=7,usecols=[1,2,7,8])
sample_rate = 250
time_epoch =2
epoch = time_epoch  * sample_rate

i = 0
while(i + epoch < raw_eeg_data.shape[0]):
    print("Time " + str(i/sample_rate) + "to" + str((i+ epoch)/sample_rate))
    for c in range(4):
        ch = detrend(raw_eeg_data[i:i + epoch,c])
        corr = np.correlate(ch, ch, 'full') 
        peaks = find_peaks_cwt(corr/np.max(corr),[10,20,30,40,50,500])
        
    print(peaks[0])
    
    
    i = i + epoch

def plots(raw_eeg_data,num_channels=4, name_channel=["1","2","6","7"]):
        """
       
        Plot the raw and filtered data 
        
        Input:
            - num_channels: number of channels to plot
        
        """

        fig = plt.figure()
        for channel in range(num_channels):  
            #fig = plt.figure()

            #fig.suptitle(self.name_channel[channel])
    
            t_sec = np.array(range(0, raw_eeg_data[:,channel].size)) / sample_rate
            
            ax1 = plt.subplot(321)
            plt.plot(t_sec, raw_eeg_data[:,channel],label=name_channel[channel])
            plt.ylabel('EEG (uV)')
            plt.xlabel('Time (sec)')
            plt.title('Raw')
            plt.xlim(t_sec[0], t_sec[-1])
            
            psd,freqs = mlab.psd(np.squeeze(raw_eeg_data[:,channel]),NFFT=500,Fs=250)    
            ax2 = plt.subplot(322)
            plt.xlim(t_sec[0], t_sec[-1])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.ylim(0,10)
            plt.plot(freqs,psd,label=name_channel[channel])
            plt.title('PSD of Unfiltered')
            
#            ax3= plt.subplot(323)
#            spec_PSDperHz, freqs, t_spec = mlab.specgram(
#                                           np.squeeze(raw_eeg_data[:,channel]),
#                                           NFFT=500,
#                                           window=mlab.window_hanning,
#                                           Fs=sample_rate,
#                                           noverlap=0
#                                           ) 
#            spec_PSDperBin = spec_PSDperHz * sample_rate / float(500)
#            plt.pcolor(t_spec, freqs, 
#                       10*np.log10(spec_PSDperBin))
#            plt.clim(25-5+np.array([-40, 0]))
#            plt.xlim(t_sec[0], t_sec[-1])
#            plt.xlabel('Time (sec)')
#            plt.ylabel('Frequency (Hz)')
#            plt.title('Spectogram of Unfiltered')
            
            

            
        plt.legend(name_channel,loc='upper right', bbox_to_anchor=(0.01, 0.01))
        plt.tight_layout()
        plt.show()

plots(raw_eeg_data)