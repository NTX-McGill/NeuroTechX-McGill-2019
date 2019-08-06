#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:14:12 2019

@author: jenisha
"""



import numpy as np
import pandas as pd
from copy import deepcopy

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.signal import butter, ellip, cheby1, cheby2, lfilter, freqs,iirfilter

import numpy.fft as fft

"""
Visualize eye blinks

Input:
- path: where the data is stored
- name_channels: name of each channel

"""


class Kiral_Korek_Preprocessing():
    def __init__(self, path, 
                 name_channels=["C3", "C1", "Cp3", "Cp1", "Cp2", "Cp4", "C2", "C4"]):
        self.path = path
        self.sample_rate = 250 #Default sampling rate for OpenBCI
        self.name_channel = name_channels

    def load_data_BCI(self, list_channels=[1, 2, 3, 4, 5, 6, 7, 8]):
        """
        list_channels=[2, 7, 1, 8, 4, 5, 3, 6])
        name_channels=["C1", "C2", "C3", "C4", "Cp1", "Cp2", "Cp3", "Cp4"]
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
            
        
        
    def initial_preprocessing(self, bp_lowcut =1, bp_highcut =70, bp_order=5,
                          notch_freq_Hz  = [60.0, 120.0], notch_order=2):
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
       b_bandpass, a_bandpass = butter(bp_order, [self.low , self.high], btype='band', analog=True)
       
       
       
       self.bp_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_bandpass, a_bandpass ,l),0,
                                                      self.raw_eeg_data)

       self.notch_filtered_eeg_data = self.bp_filtered_eeg_data
       
       for freq_Hz in notch_freq_Hz: 
            bp_stop_Hz = freq_Hz + float(notch_order)*np.array([-1, 1])  # set the stop band
            b_notch, a_notch = butter(notch_order, bp_stop_Hz/self.nyq , 'bandstop')
            self.notch_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_notch, a_notch,l),0,
                                                              self.notch_filtered_eeg_data)
       self.corrected_eeg_data = self.notch_filtered_eeg_data
       
       #self.corrected_eeg_data = self.raw_eeg_data


    def convert_to_freq_domain(self, data, NFFT = 250, FFTstep = 1):
        
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
    
            
    

    def plots(self, num_channels=8):
        """
       
        Plot the raw and filtered data 
        
        Input:
            - num_channels: number of channels to plot
        
        """

        fig = plt.figure()
        for channel in range(num_channels):  
            #fig = plt.figure()

            #fig.suptitle(self.name_channel[channel])
    
            self.t_sec = np.array(range(0, self.raw_eeg_data[:,channel].size)) / self.sample_rate
            
            ax1 = plt.subplot(321)
            plt.plot(self.t_sec, self.raw_eeg_data[:,channel],label=self.name_channel[channel])
            plt.ylabel('EEG (uV)')
            plt.xlabel('Time (sec)')
            plt.title('Raw')
            plt.xlim(self.t_sec[0], self.t_sec[-1])
            
            psd,freqs = mlab.psd(np.squeeze(self.raw_eeg_data[:,channel]),NFFT=500,Fs=250)    
            ax2 = plt.subplot(322)
            plt.xlim(self.t_sec[0], self.t_sec[-1])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.ylim(0,20*np.median(psd))
            plt.plot(freqs,psd,label=self.name_channel[channel])
            plt.title('PSD of Unfiltered')
            
            
            raw_spec_PSDperHz, raw_freqs, raw_t_spec =  mlab.specgram(
                                            np.squeeze(raw_eeg_data[:,channel]),
                                           NFFT=NFFT,
                                           window=mlab.window_hanning,
                                           Fs=sample_rate,
                                           noverlap=overlap
                                           ) 

            raw_spec_PSDperBin = raw_spec_PSDperHz * sample_rate / float(NFFT)
            ax2 = plt.subplot(323)
            plt.pcolor(raw_t_spec, raw_freqs, 10*np.log10(raw_spec_PSDperBin))
            plt.clim(25-5+np.array([-40, 0]))
            plt.xlim(t_sec[0], t_sec[-1])
            plt.ylim([0, 100]) 
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectogram of Unfiltered')
            
            
            ax3 = plt.subplot(323)
            plt.plot(self.t_sec, self.corrected_eeg_data[:,channel],label=self.name_channel[channel])
            plt.ylabel('EEG (uV)')
            plt.xlabel('Time (sec)')
            plt.title('Filtered')
            plt.xlim(self.t_sec[0], self.t_sec[-1])
            
            psd,freqs = mlab.psd(np.squeeze(self.corrected_eeg_data[:,channel]),NFFT=500,Fs=250)    
            ax4 = plt.subplot(324)
            plt.xlim(self.t_sec[0], self.t_sec[-1])
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power')
            plt.ylim(0,20*np.median(psd))
            plt.plot(freqs,psd,label=self.name_channel[channel])
            plt.title('PSD of filtered')
            
        plt.legend(self.name_channel,loc='upper right', bbox_to_anchor=(0.01, 0.01))
        plt.tight_layout()
        plt.show()
            
plt.close('all')
path='/Users/jenisha/Desktop/NeuroTechX-McGill-2019/offline/data/March24_013/'
#fname= path +  '1_heartrate_jumpscare_OpenBCI-RAW-2019-03-24_11-17-25.txt'
fname = path + '3_heartrate_horrormvoietrailercompilation_OpenBCI-RAW-2019-03-24_11-31-33.txt'


test4 = Kiral_Korek_Preprocessing(fname, name_channels=["heart"])
test4.load_data_BCI(list_channels=[1])
test4.initial_preprocessing(bp_lowcut =5, bp_highcut =20, bp_order=2)

test4.plots(num_channels=1)     