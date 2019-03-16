
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.signal import butter, lfilter, iirfilter, welch


from scipy.fftpack import fft

import numpy.fft as fft1

def initial_preprocessing(bp_lowcut =1, bp_highcut =70, bp_order=2,
                          notch_freq_Hz  = [60.0, 120.0], notch_order =2):
       """
       Filters the data by applying
       - A zero-phase Butterworth bandpass was applied from 1 – 70 Hz. 
       - A 60-120 Hz notch filter to suppress line noise
      
       
       Input:
           - bp_ lowcut: lower cutoff frequency for bandpass filter
           - bp_highcut: higher cutoff frequency for bandpass filter
           - bp_order: order of bandpass filter
           - notch_freq_Hz: cutoff frequencies for notch fitltering
           - notch_order: order of notch filter
        
        """
       
       nyq = 0.5 * 250 #Nyquist frequency
       low = bp_lowcut / nyq
       high = bp_highcut / nyq
       
       #Bandpass filter
       b_bandpass, a_bandpass = butter(bp_order, [low, high], btype='band')
       bp_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_bandpass, a_bandpass ,l),0,raw_eeg_data)
       notch_filtered_eeg_data = bp_filtered_eeg_data
       low1  = notch_freq_Hz[0]
       high1 = notch_freq_Hz[1]
       low1  = low1/nyq
       high1 = high1/nyq
       b_notch, a_notch = iirfilter(2, [low1, high1], btype='bandstop')
       notch_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_notch, a_notch ,l),0,bp_filtered_eeg_data)

       self.corrected_epoched_eeg_data = []
       #need to use lists because different window sizes
       epoched_corrected = []
       a = []
       
       for epoch in notch_filtered_eeg_data.T:
           epoch_mean = np.mean(epoch)
           epoch_std = np.std(epoch)
           a.append(epoch_std)
           epoched_corrected.append(np.array([epoch_mean + epoch_std * 6 if i > epoch_mean + epoch_std * 6 
                                        else epoch_mean - epoch_std * 6 if i < epoch_mean - epoch_std * 6 
                                        else i for i in epoch]))
           self.corrected_epoched_eeg_data = np.array(epoched_corrected)

       for window in self.corrected_epoched_eeg_data:
            f, Pxx_den = welch(window, 250)

            i = 0
            while i < len(f):
                if f[i]>8:
                     upper_in=i
                     break
                else:
                     i=i+1

            while i < len(f):
                 if f[i]>12:
                      lower_in=i
                      break
                 else:
                      i=i+1
    
            corrected_mean_spectra_PSDperBin = np.mean(Pxx_den[upper_in:lower_in])
            corrected_mean_spectram_PSDperBin = np.mean(Pxx_den)
            ratio=corrected_mean_spectra_PSDperBin/corrected_mean_spectram_PSDperBin
            c3c4=[]
            c3c4.append(ratio)
            
