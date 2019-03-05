#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:25:05 2019

@author: marley
"""
import numpy as np
import numpy.fft as fft
from scipy import signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def filter_(arr, fs_Hz, lowcut, highcut, order):
   nyq = 0.5 * fs_Hz
   b, a = signal.butter(1, [lowcut/nyq, highcut/nyq], btype='band')
   for i in range(0, order):
       arr = signal.lfilter(b, a, arr, axis=0)
   return arr

def get_start_indices(ch):
    start_indices = [0]
    i = 0
    while i < len(ch):
        if ch[i] > 100:
            start_indices.append(i)
            i += 500
        i += 1
    return start_indices

def get_spectral_content(ch, fs_Hz, shift=0.1):
    NFFT = fs_Hz*2
    overlap = NFFT - int(shift * fs_Hz)
    spec_PSDperHz, spec_freqs, spec_t = mlab.specgram(np.squeeze(ch),
                                   NFFT=NFFT,
                                   window=mlab.window_hanning,
                                   Fs=fs_Hz,
                                   noverlap=overlap
                                   ) # returns PSD power per Hz
    # convert the units of the spectral data
    spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)
    return spec_t, spec_freqs, spec_PSDperBin  # dB re: 1 uV

def plot_specgram(spec_freqs, spec_PSDperBin,title,i=1):
    f_lim_Hz = [0, 20]   # frequency limits for plotting
    #plt.figure(figsize=(10,5))
    spec_t = [i*.1 for i in range(len(spec_PSDperBin[0]))]
    plt.subplot(3,1,i)
    plt.title(title)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25,26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.subplots_adjust(hspace=1)
    
def resize_blocks(specgram, i=1):
    min_length = min([len(el[0]) for el in specgram])
    specgram = np.array([el[:, :min_length] for el in specgram])
    return specgram


fname = 'data/March 4/5_SUCCESS_Rest_RightAndJawClench_10secs.txt' 
#fname = 'data/March 4/6_SUCCESS_Rest_RightClench_JawClench_ImagineClench_10secs.txt' 
#fname = 'data/March 4/7_SUCCESS_Rest_RightClenchImagineJaw_10secs.txt'
sampling_freq = 250
shift = 0.1
channel = (1)
channel_name = 'C4'

data = np.loadtxt(fname,
                  delimiter=',',
                  skiprows=7,
                  usecols=channel)
data = filter_(data.T, sampling_freq, 1, 40, 1)
        
ch = data
start_indices = get_start_indices(ch)

left_specgram = []
rest_specgram = []

t,f,all_spectra = get_spectral_content(ch, sampling_freq, shift)
for i in range(len(start_indices) - 1):
    start = int(start_indices[i]/(sampling_freq * shift))
    end = int(start_indices[i+1]/(sampling_freq * shift))
    d = all_spectra[:,start:end]
    # this trial alternates between rest and left motor imagery
    if i % 2:
        left_specgram.append(d)
    else:
        rest_specgram.append(d)

# resize the blocks so that they're the same length as the minimum length block
rest_specgram = resize_blocks(rest_specgram)
left_specgram = resize_blocks(left_specgram)

# plot average spectrogram of both classes
plt.figure()
av = np.mean(np.array(rest_specgram), axis=0)
plot_specgram(f, av,channel_name + ' rest', 1)
av = np.mean(np.array(left_specgram), axis=0)
plot_specgram(f, av,channel_name + ' left', 2)

'''
for i in range(len(start_indices) - 1):
    start = start_indices[i]
    end = start_indices[i+1]
    t, f, d = get_specgram(ch[start:end], sampling_freq)
    if i < 2:
        plt.figure()
        plot_specgram(f, d, 'a')
    if i % 2:
        rest_specgram.append(d)
    else:
        left_specgram.append(d)
        
        
plt.figure()
plt.plot(data_eo)
plt.figure()
draw_specgram(ch, sampling_freq, fig, 1)
'''