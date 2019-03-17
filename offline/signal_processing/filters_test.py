#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:26:48 2019

@author: jenisha
"""

import numpy as np
import pandas as pd
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.signal import butter, ellip, cheby1, cheby2, lfilter, freqs

import numpy.fft as fft

"""
Implementation of the preprocessing as outlined in 
"TrueNorth-enabled Real-time Classification of EEG Data for Brain- TrueNorth-enabled 
Real-time Classification of EEG Data for Brain-"
and
"A Robust Low-Cost EEG Motor Imagery-Based Brain-Computer Interface:"

Input:
- path: where the data is stored
- name_channels: name of each channel

"""


class Kiral_Korek_Preprocessing():
    def __init__(self, path, name_channels=["C1", "C2", "C3", "C4", "Cp1", "Cp2", "Cp3", "Cp4"]):
        self.path = path
        self.sample_rate = 250 #Default sampling rate for OpenBCI
        self.name_channel = name_channels
        

    
    def load_data_BCI(self, list_channels=[2, 7, 1, 8, 4, 5, 3, 6]):
        """
        
        Load the data from OpenBCI txt file

        Input:
            list_channel: lists of channels to use

        
        """
        self.number_channels = len(list_channels)
        self.list_channels = list_channels
        
        # load raw data into numpy array
        self.raw_eeg_data = np.loadtxt(self.path, 
                                       delimiter=',',
                                       skiprows=7,
                                       usecols=list_channels)

        #expand the dimmension if only one channel             
        if self.number_channels == 1:
            self.raw_eeg_data = np.expand_dims(self.raw_eeg_data, 
                                                          axis=1)
            
        
        
    def initial_preprocessing(self, bp_lowcut =1, bp_highcut =70, bp_order=2,
                          notch_freq_Hz  = [60.0, 120.0], notch_order =2):
       """
       Filters the data by applying
       - An SL filter
       - A zero-phase Butterworth bandpass was applied from 1 – 70 Hz. 
       - A 60-120 Hz notch filter to suppress line noise
      
       
       Input:
           - bp_ lowcut: lower cutoff frequency for bandpass filter
           - bp_highcut: higher cutoff frequency for bandpass filter
           - bp_order: order of bandpass filter
           - notch_freq_Hz: cutoff frequencies for notch fitltering
           - notch_order: order of notch filter

        
        """
       self.nyq = 0.5 * self.sample_rate #Nyquist frequency
       self.low = bp_lowcut / self.nyq
       self.high = bp_highcut / self.nyq
       
       #Butter
       #b_bandpass, a_bandpass = butter(bp_order, [bp_lowcut, bp_highcut], btype='band', analog=True)
       
       #Ellip
       #b_bandpass, a_bandpass = ellip(bp_order, 5, 40, [bp_lowcut, bp_highcut], 'bandpass', analog=True)
       
       #Cheby type 1
       #b_bandpass, a_bandpass =cheby1(bp_order, 5, [bp_lowcut, bp_highcut], 'bandpass', analog=True)
       
       #Cheby type 2
       #b_bandpass, a_bandpass =cheby2(bp_order, 5, [bp_lowcut, bp_highcut], 'bandpass', analog=True)
       
       #Firwin filter (replace b_bandpass and a_bandpass in line with coefficients)
       #coefficients =  firwin(2**6-1, [0.5, 30], width=0.05, pass_zero=False, fs = sampling_freq)
       
       
       self.bp_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_bandpass, a_bandpass ,l),0,
                                                      self.sl_filtered_data)
   
       self.notch_filtered_eeg_data = self.bp_filtered_eeg_data
       
       for freq_Hz in notch_freq_Hz: 
            bp_stop_Hz = freq_Hz + float(notch_order)*np.array([-1, 1])  # set the stop band
            b_notch, a_notch = butter(notch_order, bp_stop_Hz/self.nyq , 'bandstop')
            self.notch_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_notch, a_notch,l),0,
                                                              self.notch_filtered_eeg_data)

    def convert_to_freq_domain(self, data, NFFT = 500, FFTstep = 125):
        
        """
        
        Computes a spectogram via an FFT
        
        Input:
            - data: data to draw the spectogram on
            - NFFT: The number of data points used in each block
            - FFTstep: Length of the signal you want to calculate the Fourier transform of.
       
        Return:
            - list_spec_PSDperBin: list of periograms
                - 2D where columns are the periodograms of successive segments
            - list_freqs: The frequencies corresponding to the rows in spectrum
            - list_t_spec: The times corresponding the columns in spectrum
        
       """
        
        FFTstep = FFTstep   # do a new FFT every FFTstep data points
        overlap = NFFT - FFTstep  # half-second steps
    
        list_spec_PSDperHz,list_spec_PSDperBin, list_freqs, list_t_spec= [], [], [], []
        

        for filtered in data.T:
            spec_PSDperHz, freqs, t_spec = mlab.specgram(
                                           np.squeeze(filtered),
                                           NFFT=NFFT,
                                           window=mlab.window_hanning,
                                           Fs=self.sample_rate,
                                           noverlap=overlap
                                           ) 
            spec_PSDperBin = spec_PSDperHz * self.sample_rate / float(NFFT)  # convert to "per bin"
            list_spec_PSDperHz.append(spec_PSDperHz)
            list_spec_PSDperBin.append(spec_PSDperBin)
            list_freqs.append(freqs)
            list_t_spec.append(t_spec)
        
        return (np.array(list_spec_PSDperBin), np.array(list_freqs), np.array(list_t_spec))
    
    
    def plots(self, num_channels=0):
        """
       
        Plot the raw and filtered data of a channel as well as their spectrograms
        
        Input:
            - channel: channel whose data is to plot
        
        """
        self.raw_spec_PSDperBin, self.raw_freqs, self.raw_t_spec = self.convert_to_freq_domain(self.raw_eeg_data)
        
        for i in range(num_channels):  
            fig = plt.figure()
    
            t_sec = np.array(range(0, self.raw_eeg_data[:,channel].size)) / self.sample_rate
            
            ax1 = plt.subplot(221)
            plt.plot(t_sec, self.raw_eeg_data[:,channel])
            plt.ylabel('EEG (uV)')
            plt.xlabel('Time (sec)')
            plt.title('Raw')
            plt.xlim(t_sec[0], t_sec[-1])
            
            ax2 = plt.subplot(222)
            plt.pcolor(self.raw_t_spec[channel], self.raw_freqs[channel], 
                       10*np.log10(self.raw_spec_PSDperBin[channel]))
            plt.clim(25-5+np.array([-40, 0]))
            plt.xlim(t_sec[0], t_sec[-1])
            plt.ylim([0, 60]) 
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectogram of Unfiltered')
            
            self.corrected_spec_PSDperBin, self.corrected_freqs, self.corrected_t_spec = self.convert_to_freq_domain(self.corrected_eeg_data)
            ax3 = plt.subplot(223)
            plt.plot(t_sec, self.corrected_eeg_data[:,channel])
            plt.ylim(-100, 100)
            plt.ylabel('EEG (uV)')
            plt.xlabel('Time (sec)')
            plt.title('Filtered')
            plt.xlim(t_sec[0], t_sec[-1])
            
            ax4 = plt.subplot(224)
            plt.pcolor(self.corrected_t_spec[channel], self.corrected_freqs[channel], 
                       10*np.log10(self.corrected_spec_PSDperBin[channel]))
            plt.clim(25-5+np.array([-40, 0]))
            plt.xlim(t_sec[0], t_sec[-1])
            plt.ylim([0, 60])  
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectogram of Filtered')
    
    
    
            plt.tight_layout()
            plt.show()
            

#%%

\

            
