#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 27 09:21:25 2019

@author: jenisha
"""

import numpy as np
import pandas as pd
import re

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.signal import freqz
from scipy.signal import butter, lfilter

class Preprocessing():
    def __init__(self, path):
        self.path = path
        
    
    def load_data_BCI(self, list_channels):
        """
        
        Load the data from OpenBCI txt file
        and extract some parameters such as 
        - number of channels
        - list of channels
        - raw data
        - timestamp
        
        
        Input:
            channel: lists of channels to use
        
        
        """
        self.list_channels = list_channels
        self.number_channels = len(list_channels)
        
        # load raw data into numpy array
        self.raw_eeg_data = np.loadtxt(self.path, delimiter=',',
                                   skiprows=7,usecols=list_channels)
        
        # extract time stamps
        self.time_stamps = pd.read_csv(self.path, delimiter=',', 
                                       skiprows=7,usecols=[12], header = None)
        
        #TODO: time_length
        
        with open(self.path) as f:
            sample_rate_txt = f.readlines()[2:3]
            self.sample_rate = int(float(re.findall(r'[+-]?\d+\.\d+',str(sample_rate_txt))[0]))
        
    
    def initial_filtering(self, hp_cutoff_Hz = 1.0, notch_freq_Hz  = [60.0, 120.0],
                          order_high_pass=2, order_notch =3):
       """
       Filters the data to remove DC components and center frequencies
       
       Input:
           hp_cutoff_Hz: cutoff frequency for highpass filter
           notch_freq_Hz: cutoff frequencies for notch fitltering
           order_high_pass: order of highpass filter
           order_notch: order of notch filter
        
       
        
       """
       # filter the data to remove DC
       self.nyq = 0.5 * self.sample_rate
       b_high, a_high = butter(order_high_pass, hp_cutoff_Hz/self.nyq, 'highpass')
       self.filtered_eeg_data_dc = lfilter(b_high, a_high, self.raw_eeg_data, 0)
#       self.filtered_eeg_data_dc = np.apply_along_axis(lambda l: lfilter(b_high, a_high ,l),0,
#                                                      self.raw_eeg_data)
       
       # notch filter the data to remove center frequencies
       self.filtered_eeg_data_notch = self.filtered_eeg_data_dc
       for freq_Hz in notch_freq_Hz: 
            bp_stop_Hz = freq_Hz + float(order_notch)*np.array([-1, 1])  # set the stop band
            b_notch, a_notch = butter(order_notch, bp_stop_Hz/self.nyq , 'bandstop')
            self.filtered_eeg_data_notch = lfilter(b_notch, a_notch, self.filtered_eeg_data_notch, 0)
#            self.filtered_eeg_data_notch = np.apply_along_axis(lambda l: lfilter(b_notch, a_notch,l),0,
#                                                               self.filtered_eeg_data_dc)
                   
    
    def convert_to_freq_domain(self, NFFT = 256, FFTstep = 100):
        
        self.FFTstep = FFTstep   # do a new FFT every FFTstep data points
        self.overlap = NFFT - FFTstep  # half-second steps
        
        self.spec_PSDperHz, self.freqs, self.t_spec = mlab.specgram(
                                           np.squeeze(self.filtered_eeg_data_notch),
                                           NFFT=NFFT,
                                           window=mlab.window_hanning,
                                           Fs=self.sample_rate,
                                           noverlap=self.overlap
                                           ) 
    
        self.spec_PSDperBin = self.spec_PSDperHz * self.sample_rate / float(NFFT)
        
        
#        self.spec_PSDperHz,self.spec_PSDperBin , self.freqs, self.t_spec= [], [], [], []
#        for filtered in self.filtered_eeg_data_notch:
#            spec_PSDperHz, freqs, t_spec = mlab.specgram(
#                                           np.squeeze(filtered),
#                                           NFFT=NFFT,
#                                           window=mlab.window_hanning,
#                                           Fs=self.sample_rate,
#                                           noverlap=overlap
#                                           ) 
#            self.spec_PSDperBin = spec_PSDperHz * self.sample_rate / float(NFFT)  # convert to "per bin"
#            self.spec_PSDperHz.append(spec_PSDperHz)
#            self.spec_PSDperBin = spec_PSDperHz * self.sample_rate / float(NFFT)
#            self.spec_PSDperBin.append(spec_PSDperBin )
#            self.freqs.append(freqs)
#            self.t_spec.append(t_spec)
            
        
    def plots(self):
        """
        
        
        """
        
        fig = plt.figure()
        t_sec = np.array(range(0, self.filtered_eeg_data_notch.size)) / self.sample_rate

        ax0 = plt.subplot(311)
        plt.plot(t_sec, self.raw_eeg_data)
        
        
        #plt.ylim(-100, 100)
        plt.ylabel('EEG (uV)')
        plt.xlabel('Time (sec)')
        plt.title('Raw')
        plt.xlim(t_sec[0], t_sec[-1])
        
        ax1 = plt.subplot(312)
        plt.plot(t_sec, self.filtered_eeg_data_notch)
        plt.ylim(-100, 100)
        plt.ylabel('EEG (uV)')
        plt.xlabel('Time (sec)')
        plt.title('Filtered')
        plt.xlim(t_sec[0], t_sec[-1])
        
        ax = plt.subplot(313)
        plt.pcolor(self.t_spec, self.freqs, 10*np.log10(self.spec_PSDperBin))
        plt.clim(25-5+np.array([-40, 0]))
        plt.xlim(t_sec[0], t_sec[-1])
        #plt.ylim([0, self.freqs/2.0])  # show the full frequency content of the signal
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectogram of Filtered')



        plt.tight_layout()
        plt.show()

             
            
    def band_pass_filtering(self, lowcut, highcut, order=5):
        """
        Filters the data using a bandpass filter
        
        
        Input:
            lowcut: lower cutoff frequency
            highcut: higher cutoff frequency
            order: 
        
        
        """ 
        
        
        self.low = lowcut / self.nyq
        self.high = highcut / self.nyq
        b_bandpass, a_bandpass = butter(order, [self.low, self.high], btype='band')
        self.y = lfilter(b, a,self.filtered_eeg_data_notch)
        
        
        
        
  
      
    
fname_ec = '/Users/jenisha/Desktop/NeuroTech-Workshop-Demo/EyesClosedNTXDemo.txt' 
fname_eo = '/Users/jenisha/Desktop/NeuroTech-Workshop-Demo/EyesOpenedNTXDemo.txt' 
test = Preprocessing(fname_ec)
test.load_data_BCI([1])
test.initial_filtering()
test.convert_to_freq_domain()
test.plots()

test2 = Preprocessing(fname_eo)
test2.load_data_BCI([1])
test2.initial_filtering()
test2.convert_to_freq_domain()
test2.plots()




         
         
            
            
        
            
    
            
