#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 15:09:47 2019

@author: marley
"""
import glob
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def draw_specgram(ch, fs_Hz, num_subplots, i, title):
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
    plt.subplot(num_subplots,1,i)
    plt.title(title)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25,26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    # plt.xlabel('Time (sec)') aint got space for this
    plt.ylabel('Frequency (Hz)')
    plt.subplots_adjust(hspace=1)

filenames = sorted([f for f in glob.glob("*.txt")])
sampling_freq = 250
shift = 0.1
channel = (1)

for idx, fname in enumerate(filenames):
    if not idx % 6:
        plt.figure(figsize=(8,10))
    data = np.loadtxt(fname,
                      delimiter=',',
                      skiprows=7,
                      usecols=channel)
    draw_specgram(data, sampling_freq, 6,idx%6 + 1,fname)