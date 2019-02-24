#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 12:04:01 2019

@author: marley
"""
import numpy as np
import numpy.fft as fft
from scipy import signal
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

def notch_mains_interference(data):
    notch_freq_Hz = np.array([60.0])  # main + harmonic frequencies
    for freq_Hz in np.nditer(notch_freq_Hz):  # loop over each target freq
        bp_stop_Hz = freq_Hz + 3.0*np.array([-1, 1])  # set the stop band
        b, a = signal.butter(3, bp_stop_Hz/(250 / 2.0), 'bandstop')
        data = signal.lfilter(b, a, data, axis=0)
        print("Notch filter removing: " + str(bp_stop_Hz[0]) + "-" + str(bp_stop_Hz[1]) + " Hz")
    return data
def filter_(arr, lowcut, highcut, order):
   arr = notch_mains_interference(arr)
   nyq = 0.5 * 250
   b, a = signal.butter(1, [lowcut/nyq, highcut/nyq], btype='band')
   for i in range(0, order):
       arr = signal.lfilter(b, a, arr, axis=0)
   return arr
def draw_specgram(ch, fs_Hz, fig, i):
    NFFT = fs_Hz*2
    overlap = NFFT - int(0.25 * fs_Hz)
    spec_PSDperHz, spec_freqs, spec_t = mlab.specgram(np.squeeze(ch),
                                   NFFT=NFFT,
                                   window=mlab.window_hanning,
                                   Fs=fs_Hz,
                                   noverlap=overlap
                                   ) # returns PSD power per Hz
    # convert the units of the spectral data
    
    spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)
    f_lim_Hz = [0, 70]   # frequency limits for plotting
    #plt.figure(figsize=(10,5))
    ax = fig.add_subplot(2,1,i)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25,26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    
fname_ec = '../data/EyesClosedNTXDemo.txt' 
fname_eo = '../data/EyesOpenedNTXDemo.txt' 
data_ec = np.loadtxt(fname_ec,
                  delimiter=',',
                  skiprows=7,
                  usecols=(1))
data_eo = np.loadtxt(fname_eo,
                  delimiter=',',
                  skiprows=7,
                  usecols=(1))

fig = plt.figure()
sampling_freq = 250
data_ec_filtered = filter_(data_ec.T, lowcut=1, highcut=20,order=1) #filter data
data_eo_filtered = filter_(data_eo.T, lowcut=1, highcut=20,order=1) #filter data
draw_specgram(data_eo_filtered, sampling_freq, fig, 1)
draw_specgram(data_ec_filtered, sampling_freq, fig, 2)

fig = plt.figure()
ax = fig.add_subplot(111)
start = 250 # start index (don't start at 0 becaues of filtering artifacts)
length = 1 # length of data to plot in seconds
ax.plot(data_ec[start:start + length*sampling_freq] - np.mean(data_ec[start:start + length*sampling_freq]))
ax.plot(data_ec_filtered[start: start+length*sampling_freq])
plt.show()