#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 15:26:48 2019

@author: jenisha
"""

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.signal import butter, lfilter

import numpy.fft as fft

"""
Implement the preprocessing as outlined in 
"" TrueNorth-enabled Real-time Classification of EEG Data for Brain- TrueNorth-enabled 
Real-time Classification of EEG Data for Brain-""

Input:
- path: where the data is stored
- list_channels: name of each channel

"""


class Kiral_Korek_Preprocessing():
    def __init__(self, path, name_channel=["C3"]):
        self.path = path
        self.sample_rate = 250 #Default sampling rate for OpenBCI
        self.name_channel = name_channel
        

    
    def load_data_BCI(self, list_channels):
        """
        
        Load the data from OpenBCI txt file

        Input:
            channel: lists of channels to use
            
            
        
        
        """
        self.number_channels = len(list_channels)
        self.list_channels = list_channels
        
        # load raw data into numpy array
        self.raw_eeg_data = np.loadtxt(self.path, 
                                       delimiter=',',
                                       skiprows=7,
                                       usecols=list_channels)

        # extract time stamps
        self.time_stamps = pd.read_csv(self.path, 
                                       delimiter=',', 
                                       skiprows=7,
                                       usecols=[12],
                                       header = None)
                       
        if self.number_channels == 1:
            self.raw_eeg_data = np.expand_dims(self.raw_eeg_data, 
                                                          axis=1)
        
        
    def initial_preprocessing(self, bp_lowcut =1, bp_highcut =70, bp_order=3,
                          notch_freq_Hz  = [60.0, 120.0], notch_order =3):
       """
       Filters the data by applying
       - A zero-phase Butterworth bandpass was applied from 1 – 70 Hz. 
       - A 60-120 Hz notch filter to suppress line noise
      
        In addition, Sample values exceeding ±6std, where std is the standard deviation
           of any given voltage trace, were set to ±6std to rectify outliers in voltage.
       
       Input:
           - bp_ lowcut: lower cutoff frequency for bandpass filter
           - bp_highcut: higher cutoff frequency for bandpass filter
           - notch_freq_Hz: cutoff frequencies for notch fitltering
           - notch_order: order of notch filter

           

        
        
        """
       self.nyq = 0.5 * self.sample_rate
       self.low = bp_lowcut / self.nyq
       self.high = bp_highcut / self.nyq
        
       b_bandpass, a_bandpass = butter(bp_order, [self.low, self.high], btype='band')
       self.bp_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_bandpass, a_bandpass ,l),0,
                                                      self.raw_eeg_data)
       
        
       self.notch_filtered_eeg_data = self.bp_filtered_eeg_data
       
       for freq_Hz in notch_freq_Hz: 
            bp_stop_Hz = freq_Hz + float(notch_order)*np.array([-1, 1])  # set the stop band
            b_notch, a_notch = butter(notch_order, bp_stop_Hz/self.nyq , 'bandstop')
            self.notch_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_notch, a_notch,l),0,
                                                              self.notch_filtered_eeg_data)
       
       
        
#       self.list_std_channel, self.list_mean_channel  = [], []
#       self.corrected_eeg_data = self.notch_filtered_eeg_data
#       for channel in self.corrected_eeg_data.T:
#            self.list_std_channel.append(np.std(channel))
#            self.list_mean_channel.append(np.mean(channel))
#            for val in channel:
#                if val > self.list_mean_channel[-1] + self.list_std_channel[-1] *  mult_std :
#                    val = val -  mult_std * self.list_mean_channel[-1]
#                elif val <  self.list_mean_channel[-1] - self.list_std_channel[-1] *  mult_std :
#                    val = val +  mult_std * self.list_mean_channel[-1]
            
          
           
            
    def epoch_data(self, data,mode="1", window_length = 2, overlap=125):
        """
        Separates the data into several windows
        
        Input:
            - data: data to seperate into windows
            - window_length: length of the window
            - overlap
        
        """
        array_epochs = []
        i = 0
        self.window_size_hz = window_length * self.sample_rate
        
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
            self.num_epoch = i + 1
           

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
        
        for filtered in self.notch_filtered_eeg_data.T:
            epoched = self.epoch_data(filtered,mode="1",overlap=0)
            epoched_corrected = []
            for epoch in epoched:
                epoch_mean = np.mean(epoch)
                epoch_std = np.std(epoch)
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
    
    def extract_features(self):
        """
        Extract mu bands (8-12 Hz) for MI
        
        """
        self.fft = []
        self.PSD = []
        self.equal_epoched_eeg_data = []
        
        for filtered in self.notch_filtered_eeg_data.T:
            self.equal_epoched_eeg_data.append(self.epoch_data(filtered,mode="2"))
   
        for channel in self.equal_epoched_eeg_data:
            #Todo: last window
            fft_tmp = (fft.fft(channel)/self.window_size_hz )
            self.fft.append(fft_tmp )
        
        self.PSD.append(2*np.abs(fft_tmp[0:int(self.window_size_hz/2),:]))
        
        self.f = self.sample_rate/2*np.linspace(0, 1, int(self.window_size_hz/2))
        self.PSD = np.array(self.PSD)
        #8-12
        #self.mean_mu = np.mean(self.PSD[:, np.where((f >= 8) & (f <= 12))], axis = -1)
        
        #self.feature_vector = np.log10(self.mean_mu)

    
    def convert_to_freq_domain(self, data, NFFT = 256, FFTstep = 100):
        
        """
        
        Computes a spectogram
        
        Input:
            - data: data to do spectogram
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
        
        return (list_spec_PSDperBin, list_freqs, list_t_spec)
    
    
    def plots(self, channel=0):
        """
       
        Plot the raw and filtered data of a channel as well as their spectrograms
        
        Input:
            - channel: channel whose data is to plot
        
        """
        self.raw_spec_PSDperBin, self.raw_freqs, self.raw_t_spec = self.convert_to_freq_domain(self.raw_eeg_data)
        
        fig = plt.figure()

        t_sec = np.array(range(0, self.raw_eeg_data.size)) / self.sample_rate
        
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
        plt.ylim([0, 60])  # show the full frequency content of the signal
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
        plt.ylim([0, 60])  # show the full frequency content of the signal
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectogram of Filtered')



        plt.tight_layout()
        #plt.show()
            
    
#fname_20 = '/Users/jenisha/Desktop/NeuroTechX-McGill-2019/offline/data/20s_rest_20s_clench_20sMI.txt'  
#test3 = Kiral_Korek_Preprocessing(fname_20)
#test3.load_data_BCI([1])
#test3.initial_preprocessing()
#test3.epoch_and_remove_outlier()
#test3.plots()


fname_4 = '/Users/jenisha/Desktop/NeuroTechX-McGill-2019/offline/data/March_4/6_SUCCESS_Rest_RightClench_JawClench_ImagineClench_10secs.txt'  
test4 = Kiral_Korek_Preprocessing(fname_4)
test4.load_data_BCI([1])
test4.initial_preprocessing()
test4.epoch_and_remove_outlier()
test4.extract_features()
#test4.plots()            
        
            
            
        
            
    
        
        
        