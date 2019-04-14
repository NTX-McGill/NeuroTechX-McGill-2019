#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 23 16:08:44 2019

@author: marley
"""
from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf

import numpy as np
import numpy.fft as fft
from scipy import signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def filter_(arr, lowcut, highcut, order):
   nyq = 0.5 * 160
   b, a = signal.butter(1, [lowcut/nyq, highcut/nyq], btype='band')
   for i in range(0, order):
       arr = signal.lfilter(b, a, arr, axis=0)
   return arr

def draw_specgram(ch, fs_Hz, fig, i, title):
    NFFT = fs_Hz*2
    overlap = NFFT - int(0.1 * fs_Hz)
    spec_PSDperHz, spec_freqs, spec_t = mlab.specgram(np.squeeze(ch),
                                   NFFT=NFFT,
                                   window=mlab.window_hanning,
                                   Fs=fs_Hz,
                                   noverlap=overlap
                                   ) # returns PSD power per Hz
    # convert the units of the spectral data
    
    spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)
    f_lim_Hz = [0, 20]   # frequency limits for plotting
    #plt.figure(figsize=(10,5))
    ax = fig.add_subplot(3,1,i)
    plt.title(title)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25,26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.show()
    
def plot_data(raw, channel_name):
    fig = plt.figure()
    ch, times = raw[raw.ch_names.index(channel_name),:]
    stim, times = raw[raw.ch_names.index('STI 014'),:]
    
    rest_indices = np.where(stim[0] == 1)[0]
    left_indices = np.where(stim[0] == 2)[0]
    right_indices = np.where(stim[0] == 3)[0]
    rest = ch[0, rest_indices]
    left = ch[0,left_indices]
    right = ch[0, right_indices]
    rest = rest * 1000000
    left = left * 1000000
    right = right * 1000000
    
    rest = filter_(rest.T, lowcut=7, highcut=20, order=1)
    left = filter_(left.T, lowcut=7, highcut=20,order=1) #filter data
    right = filter_(right.T, lowcut=7, highcut=20,order=1) #filter data
    
    draw_specgram(rest, 160, fig, 1, channel_name + " rest")
    draw_specgram(left, 160, fig, 2, channel_name + " left")
    draw_specgram(right, 160, fig, 3, channel_name + " right")
    plt.subplots_adjust(hspace=0.5)
subject = 1
runs = [4,8,12] 
channel_name = 'C4'
raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
             raw_fnames]
raw = concatenate_raws(raw_files)
raw.rename_channels(lambda x: x.strip('.'))

'''
for idx, el in enumerate(raw._data[64,:]):
    if raw._data[64,idx - 1] != el: 
        print(idx)'''
        

fs_Hz = 160


plot_data(raw, 'C4')
plot_data(raw, 'C3')