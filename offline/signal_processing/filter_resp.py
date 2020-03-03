# script to tets out different filter responses

import numpy as np
import pandas as pd
from copy import deepcopy
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
from scipy.signal import butter, ellip, cheby1, cheby2, lfilter, freqs, detrend, firwin, convolve, periodogram
import numpy.fft as fft

class filter_resp():
    
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
            
            
    def resps(self, bp_lowcut =1, bp_highcut =70, bp_order=2, notch_freq_Hz  = [60.0, 120.0], notch_order =2):
        
       
       self.epoch = detrend(self.raw_eeg_data[0:2*self.sample_rate, 2])
       t = np.linspace(0, 2, len(self.epoch))
       plt.plot(t, self.epoch)
       plt.title('2 sec epoch')

       self.nyq = 0.5 * self.sample_rate #Nyquist frequency
       self.low = bp_lowcut / self.nyq
       self.high = bp_highcut / self.nyq
       
       b_bandpass, a_bandpass = butter(bp_order, [self.low, self.high], btype='band', analog=False)

       #Bandpass filter
       plt.figure(figsize=(12, 12))
       
       plt.subplot(821)
       b_bandpass, a_bandpass = butter(bp_order, [self.low, self.high], btype='band', analog=False)
       w, h = freqs(b_bandpass, a_bandpass)
       #plt.plot(w, 20 * np.log10(abs(h)))
       plt.semilogx(w, 20 * np.log10(abs(h)))
       plt.xscale('log')
       #plt.xlim([0.1, 1000])
       plt.title('Butterworth filter frequency response ')
       plt.xlabel('Frequency [radians / second]')
       plt.ylabel('Amplitude [dB]')
       plt.margins(0, 0.1)
       plt.grid(which='both', axis='both')
       plt.subplot(823)
       self.butter_resp = lfilter(b_bandpass, a_bandpass, self.epoch)
       plt.plot(t, self.butter_resp)
       
       plt.subplot(822)
       b, a = ellip(bp_order, 5, 40, [self.low, self.high], 'bandpass', analog=False)
       w, h = freqs(b, a)
       plt.semilogx(w, 20 * np.log10(abs(h)))
       #plt.xlim([0.1, 1000])
       plt.title('Elliptic filter frequency response (rp=5, rs=40)')
       plt.xlabel('Frequency [radians / second]')
       plt.ylabel('Amplitude [dB]')
       plt.margins(0, 0.1)
       plt.grid(which='both', axis='both')
       plt.subplot(824)
       self.ellip_resp = lfilter(b, a, self.epoch)
       plt.plot(t, self.ellip_resp)
       
       plt.subplot(827)
       b, a = cheby1(bp_order, 5, [self.low, self.high], 'bandpass', analog=False)
       w, h = freqs(b, a)
       plt.plot(w, 20 * np.log10(abs(h)))
       #plt.xlim([0.1, 1000])
       plt.xscale('log')
       plt.title('Chebyshev Type I frequency response (rp=5)')
       plt.ylabel('Amplitude [dB]')
       plt.margins(0, 0.1)
       plt.grid(which='both', axis='both')
       plt.subplot(829)
       self.cheby1_resp = lfilter(b, a, self.epoch)
       plt.plot(t, self.cheby1_resp)
       
       plt.subplot(828)
       b, a = cheby2(bp_order, 5, [self.low, self.high], 'bandpass', analog=False)
       w, h = freqs(b, a)
       plt.plot(w, 20 * np.log10(abs(h)))
       #plt.xlim([0.1, 1000])
       plt.xscale('log')
       plt.title('Chebyshev Type II frequency response (rp=5)')
       plt.ylabel('Amplitude [dB]')
       plt.margins(0, 0.1)
       plt.grid(which='both', axis='both')
       plt.subplot(8,2,10)
       self.cheby2_resp = lfilter(b, a, self.epoch)
       plt.plot(t, self.cheby2_resp)
       
       plt.subplot(8,2,13)
       self.b = firwin(2**6+1, [bp_lowcut, bp_highcut], pass_zero=False, scale=True, nyq=self.nyq) 
       self.conv_result = convolve(self.epoch, self.b, mode='valid')
       b_fft = fft.rfft(self.b)
       f = np.linspace(0, self.nyq, len(b_fft))
       plt.plot(f, abs(b_fft))
       self.firwin_resp = lfilter(self.b, [1.0], self.epoch)
       plt.title('Frequency response of FIR designed using window method')
       plt.ylabel('Amplitude [dB]')
       plt.grid(which='both', axis='both')
       #plt.plot(self.butter_resp)
       #plt.plot(self.ellip_resp)
       #plt.plot(self.cheby1_resp)
       #plt.plot(self.cheby2_resp)
       plt.subplot(8,2,15)
       plt.plot(t, self.firwin_resp)
       #plt.plot(self.b)
       
    def filter_spectra(self):
       plt.subplot(4,2,8)
       self.spectrum_raw = np.abs(fft.rfft(self.epoch))
       self.spectrum_butter = np.abs(fft.rfft(self.butter_resp))
       self.spectrum_ellip = np.abs(fft.rfft(self.ellip_resp))
       self.spectrum_cheby1 = np.abs(fft.rfft(self.cheby1_resp))
       self.spectrum_cheby2 = np.abs(fft.rfft(self.cheby2_resp))
       self.spectrum_window = np.abs(fft.rfft(self.firwin_resp))
       
       freqs = np.multiply(np.divide(np.arange(len(self.spectrum_raw)), len(self.spectrum_raw)), self.sample_rate/2.0)
       freq_range = np.arange(50*2)
       
       #plt.plot(freqs[freq_range], self.spectrum_raw[freq_range])
       plt.plot(freqs[freq_range], self.spectrum_butter[freq_range])
       plt.plot(freqs[freq_range], self.spectrum_ellip[freq_range])
       plt.plot(freqs[freq_range], self.spectrum_cheby1[freq_range])
       plt.plot(freqs[freq_range], self.spectrum_cheby2[freq_range])
       plt.plot(freqs[freq_range], self.spectrum_window[freq_range])
       plt.legend(['Butterworth', 'Elliptical', 'Chebyshev 1', 'Chebyshev 2', 'Window method'])       
       plt.title('Frequency spectra of filtered epoch using diff filters')
       plt.xlabel('Frequency (Hz)')
       
# https://scipy-cookbook.readthedocs.io/items/ApplyFIRFilter.html       
       
#%%


fname = r'..\data\March 4\6_SUCCESS_Rest_RightClench_JawClench_ImagineClench_10secs.txt'
filt_resps = filter_resp(fname)
filt_resps.load_data_BCI()
filt_resps.resps(bp_lowcut =5, bp_highcut =20, bp_order=3)
filt_resps.filter_spectra()
       