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
              

       # SL filter
       self.sl_filtered_data = self.raw_eeg_data
       raw_channels = self.raw_eeg_data.transpose()
       sl_channels = deepcopy(raw_channels)

       # Subtract average of adjacent electrodes
       sl_channels[0] -= (raw_channels[1] + raw_channels[2] + raw_channels[4]) / 3 # C1 -= (C2 + C3 + Cp1) / 3
       sl_channels[1] -= (raw_channels[0] + raw_channels[3] + raw_channels[5]) / 3 # C2 -= (C1 + C4 + Cp2) / 3
       sl_channels[2] -= (raw_channels[0] + raw_channels[6]) / 3 # C3 -= (C1 + Cp3) / 2
       sl_channels[3] -= (raw_channels[1] + raw_channels[7]) / 3 # C4 -= (C2 + Cp4) / 2
       sl_channels[4] -= (raw_channels[5] + raw_channels[6] + raw_channels[0]) / 3 # Cp1 -= (Cp2 + Cp3 + C1) / 3
       sl_channels[5] -= (raw_channels[4] + raw_channels[7] + raw_channels[1]) / 3 # Cp2 -= (Cp1 + Cp4 + C2) / 3
       sl_channels[6] -= (raw_channels[4] + raw_channels[2]) / 3 # Cp3 -= (Cp1 + C3) / 2
       sl_channels[7] -= (raw_channels[5] + raw_channels[3]) / 3 # Cp4 -= (Cp2 + C4) / 2

       self.sl_filtered_data = sl_channels.transpose()

       #Bandpass filter
       plt.figure(figsize=(16, 12))
       
       plt.subplot(221)
       b_bandpass, a_bandpass = butter(bp_order, [bp_lowcut, bp_highcut], btype='band', analog=True)
       w, h = freqs(b_bandpass, a_bandpass)
       #plt.plot(w, 20 * np.log10(abs(h)))
       plt.semilogx(w, 20 * np.log10(abs(h)))
       plt.xscale('log')
       plt.xlim([0.1, 1000])
       plt.title('Butterworth filter frequency response ')
       plt.xlabel('Frequency [radians / second]')
       plt.ylabel('Amplitude [dB]')
       plt.margins(0, 0.1)
       plt.grid(which='both', axis='both')
       
       plt.subplot(222)
       b, a = ellip(bp_order, 5, 40, [bp_lowcut, bp_highcut], 'bandpass', analog=True)
       w, h = freqs(b, a)
       plt.semilogx(w, 20 * np.log10(abs(h)))
       plt.xlim([0.1, 1000])
       plt.title('Elliptic filter frequency response (rp=5, rs=40)')
       plt.xlabel('Frequency [radians / second]')
       plt.ylabel('Amplitude [dB]')
       plt.margins(0, 0.1)
       plt.grid(which='both', axis='both')
       
       plt.subplot(223)
       b, a = cheby1(bp_order, 5, [bp_lowcut, bp_highcut], 'bandpass', analog=True)
       w, h = freqs(b, a)
       plt.plot(w, 20 * np.log10(abs(h)))
       plt.xlim([0.1, 1000])
       plt.xscale('log')
       plt.title('Chebyshev Type I frequency response (rp=5)')
       plt.ylabel('Amplitude [dB]')
       plt.margins(0, 0.1)
       plt.grid(which='both', axis='both')
       
       plt.subplot(224)
       b, a = cheby2(bp_order, 5, [bp_lowcut, bp_highcut], 'bandpass', analog=True)
       w, h = freqs(b, a)
       plt.plot(w, 20 * np.log10(abs(h)))
       plt.xlim([0.1, 1000])
       plt.xscale('log')
       plt.title('Chebyshev Type II frequency response (rp=5)')
       plt.ylabel('Amplitude [dB]')
       plt.margins(0, 0.1)
       plt.grid(which='both', axis='both')
       
       
       b_bandpass, a_bandpass = butter(bp_order, [self.low, self.high], btype='band', analog=True)

        

#       plt.figure()
#       plt.plot(b_bandpass,a_bandpass)
       
       self.bp_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_bandpass, a_bandpass ,l),0,
                                                      self.sl_filtered_data)
   
       self.notch_filtered_eeg_data = self.bp_filtered_eeg_data
       
       for freq_Hz in notch_freq_Hz: 
            bp_stop_Hz = freq_Hz + float(notch_order)*np.array([-1, 1])  # set the stop band
            b_notch, a_notch = butter(notch_order, bp_stop_Hz/self.nyq , 'bandstop')
            self.notch_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_notch, a_notch,l),0,
                                                              self.notch_filtered_eeg_data)
       
       
#       mult_std = 6 
#       self.list_std_channel, self.list_mean_channel  = [], []
#       self.corrected_eeg_data = self.notch_filtered_eeg_data
#       for channel in self.corrected_eeg_data.T:
#            self.list_std_channel.append(np.std(channel))
#            self.list_mean_channel.append(np.mean(channel))
#            for val in channel:
#                if val > self.list_mean_channel[-1] + self.list_std_channel[-1] *  mult_std :
#                    print(val)
#                    val = val -  mult_std * self.list_mean_channel[-1]
#                    print(val)
#                elif val <  self.list_mean_channel[-1] - self.list_std_channel[-1] *  mult_std :
#                    val = val +  mult_std * self.list_mean_channel[-1]
#            
          
            
    def epoch_data(self, data,mode="1", window_length = 2, overlap=125):
        """
        Separates the data into several windows
        
        Input:
            - data: data to seperate into windows
            - mode: whether the windows are of same length (mode 1) or different lengths (mode 2)
            - window_length: length of the window in s
            - overlap: overlap in the previous window
        
        """
        array_epochs = []
        i = 0
        self.window_size_hz = int(window_length * self.sample_rate)
        
        if mode == "1":
            while(i  < len(data) ):
                array_epochs.append(data[i:i+self.window_size_hz ])
                i = i + self.window_size_hz 
            

            if i is not len(data) - 1:
                array_epochs.append(data[i:len(data)])

            self.epoch = array_epochs
        
        if mode == "2":
            while i + self.window_size_hz < len(data):
                array_epochs.append(data[i:i + self.window_size_hz])
                i += overlap
            self.num_epochs = i + 1
           

        return np.array(array_epochs)
    
    def epoch_and_remove_outlier(self,mult_std = 6):
        """
        For each channel, Separates the data into several windows
        and sample values exceeding ±mult_std, where mult_std is the standard deviation
           of any given voltage trace, are set to ±mult_std to rectify outliers in voltage.
           
        Input: 
            - mult_std:
        
        
        """
        self.corrected_epoched_eeg_data = []
        #need to use lists because different window sizes
        for filtered in self.notch_filtered_eeg_data.T:
            epoched = self.epoch_data(filtered,mode="1",window_length = 2,overlap=0)
            epoched_corrected = []
            self.a = []
            for epoch in epoched:
                epoch_mean = np.mean(epoch)
                epoch_std = np.std(epoch)
                self.a.append(epoch_std)
                epoched_corrected.append(np.array([i - epoch_std * mult_std if i > epoch_mean + epoch_std * mult_std 
                                          else i + epoch_std * mult_std if i < epoch_mean - epoch_std * mult_std 
                                          else i for i in epoch]))
            self.corrected_epoched_eeg_data.append(epoched_corrected)
            


        
        # For graphing, "de-epoch it
        self.corrected_eeg_data = []
        for channel in self.corrected_epoched_eeg_data:
            de_epoched = [i for epoch in channel for i in epoch]
            self.corrected_eeg_data.append(de_epoched)
  
        self.corrected_eeg_data = np.array(self.corrected_eeg_data).transpose()
        

        
        

    
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
    
    
    def plots(self, channel=0):
        """
       
        Plot the raw and filtered data of a channel as well as their spectrograms
        
        Input:
            - channel: channel whose data is to plot
        
        """
        self.raw_spec_PSDperBin, self.raw_freqs, self.raw_t_spec = self.convert_to_freq_domain(self.raw_eeg_data)
        
        
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
        
    def extract_features(self, mu_band_Hz=[8,12]):
        
        
        # get the mean spectra and convert from PSD to uVrms
        self.corrected_mean_spectra_PSDperBin,self.corrected_mean_uVrmsPerSqrtBin =[],[]
        i = 0
        while i < self.number_channels:
            spectra = self.corrected_spec_PSDperBin[i]
            bool_inds = (self.corrected_freqs[i] > mu_band_Hz[0]) & (self.corrected_freqs[i] < mu_band_Hz[1])
            corrected_mean_spectra_PSDperBin = np.mean(spectra[bool_inds,:], 0)
            self.corrected_mean_spectra_PSDperBin.append(corrected_mean_spectra_PSDperBin)
            self.corrected_mean_uVrmsPerSqrtBin.append(np.sqrt(self.corrected_mean_spectra_PSDperBin))
            i = i + 1
            
        self.features = np.array(self.corrected_mean_uVrmsPerSqrtBin)
        return self.features
            
#%%

path='C:\\Users\\Dylan\OneDrive - McGill University\\BCI_NeuroTech\\GitHub\\NeuroTechX-McGill-2019\\offline\\data\\'
fname_4 = path + 'March 11\\1_JawRest_JawRightClench_10s.txt'
test4 = Kiral_Korek_Preprocessing(fname_4)
test4.load_data_BCI()
test4.initial_preprocessing(bp_lowcut =5, bp_highcut =20, bp_order=2)
test4.epoch_and_remove_outlier()
test4.plots()            
feats = test4.extract_features()

            
