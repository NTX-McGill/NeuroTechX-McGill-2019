#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:47:32 2019

@author: jenisha
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.signal import butter, lfilter

import numpy.fft as fft


class Preprocessing_by_channel():
    def __init__(self, path, name_channels=["C4"], channel_index=[0]):

        """
            Input:
            - path: where the data is stored
            - name_channels: name of each channel
            - list_channel: lists of channels to use
            
        """
        self.path = path
        self.sample_rate = 250 #Default sampling rate for OpenBCI
        self.name_channels = name_channels
        self.channel_index = channel_index
        self.number_channels = len(channel_index)
        
        
        # load raw data into numpy array
        self.raw_eeg_data = np.loadtxt(self.path, 
                                       delimiter=',',
                                       skiprows=7,
                                       usecols=channel_index)

        #expand the dimmension if only one channel             
        if self.number_channels == 1:
            self.raw_eeg_data = np.expand_dims(self.raw_eeg_data, 
                                                          axis=1)
            
    
    
    def epoch_data(self, data, mode="1", window_length = 2, overlap=125):
        """
        Separates the data into several windows
        
        Input:
            - data: data to seperate into windows
            - mode: whether the windows are non-overlapping(mode 1) 
            or overlapping or of same lengths (mode 2)
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
    
    def preprocessing(self, data, bp_lowcut =1, bp_highcut =70, bp_order=2,
                      notch_freq_Hz  = [60.0, 120.0], notch_order =2,
                      mult_std = 6 ):
        """
       Filters the data by applying
       - A zero-phase Butterworth bandpass was applied from 1 – 70 Hz. 
       - A 60-120 Hz notch filter to suppress line noise
      
        For each channel, Separates the data into several windows
        and sample values exceeding ±mult_std, where mult_std is the standard deviation
           of any given voltage trace, are set to ±mult_std to rectify outliers in voltage.
       
       Input:
           - bp_ lowcut: lower cutoff frequency for bandpass filter
           - bp_highcut: higher cutoff frequency for bandpass filter
           - bp_order: order of bandpass filter
           
           - notch_freq_Hz: cutoff frequencies for notch fitltering
           - notch_order: order of notch filter
           
           - mult_std: multiple of deviation of std

        
        """
        self.nyq = 0.5 * self.sample_rate #Nyquist frequency
        self.low = bp_lowcut / self.nyq
        self.high = bp_highcut / self.nyq
        
        #Bandpass filter
        b_bandpass, a_bandpass = butter(bp_order, [self.low, self.high], btype='band')      
        bp_filtered_eeg_data = lfilter(b_bandpass, a_bandpass, data, 0)
        
        #Notch filtering
        notch_filtered_eeg_data = bp_filtered_eeg_data
        for freq_Hz in notch_freq_Hz: 
            bp_stop_Hz = freq_Hz + float(notch_order)*np.array([-1, 1])  # set the stop band
            b_notch, a_notch = butter(notch_order, bp_stop_Hz/self.nyq , 'bandstop')
            notch_filtered_eeg_data = lfilter(b_notch, a_notch,notch_filtered_eeg_data,0)
            
            
        #6 std correction
        epoched = self.epoch_data(notch_filtered_eeg_data,mode="1",window_length = 2,overlap=0)
        corrected_eeg_data = []
        for epoch in epoched:
            epoch_mean = np.mean(epoch)
            epoch_std = np.std(epoch)
            corrected_eeg_data.extend([i - epoch_std * mult_std if i > epoch_mean + epoch_std * mult_std 
                                          else i + epoch_std * mult_std if i < epoch_mean - epoch_std * mult_std 
                                          else i for i in epoch])
    
        return np.array(corrected_eeg_data)
    

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
    
        spec_PSDperHz, freqs, t_spec = mlab.specgram(
                                           np.squeeze(data),
                                           NFFT=NFFT,
                                           window=mlab.window_hanning,
                                           Fs=self.sample_rate,
                                           noverlap=overlap
                                           ) 
        spec_PSDperBin = spec_PSDperHz * self.sample_rate / float(NFFT)  # convert to "per bin"

        return (spec_PSDperBin, freqs,t_spec)
    
    
    
    def extract_features_mu_band(self, data, mu_band_Hz=[8,12]):
        
        
        # get the mean spectra and convert from PSD to uVrms
        self.corrected_mean_spectra_PSDperBin,self.corrected_mean_uVrmsPerSqrtBin =[],[]
        
        bool_inds = (data > mu_band_Hz[0]) & (data < mu_band_Hz[1])
        corrected_mean_spectra_PSDperBin = np.mean(data[bool_inds,:], 0)
        corrected_mean_uVrmsPerSqrtBin = np.sqrt(self.corrected_mean_spectra_PSDperBin)

            
        return (corrected_mean_spectra_PSDperBin , corrected_mean_uVrmsPerSqrtBin)
        
        
    def preprocessed_and_features(self):
                
        
        for i in self.channel_index:
            data_tmp = self.raw_eeg_data[i:]
            preprocessed = self.preprocessing(data_tmp)
            fft = self.convert_to_freq_domain(preprocessed)
            features = self.extract_features_mu_band(fft)
            
