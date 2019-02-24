#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 24 13:12:13 2019

@author: marley
"""

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
    f_lim_Hz = [0, 60]   # frequency limits for plotting
    #plt.figure(figsize=(10,5))
    ax = fig.add_subplot(2,1,i)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25,26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.show()
fname_ec = '../data/EyesClosedNTXDemo.txt' 
fname_eo = '../data/EyesOpenedNTXDemo.txt' 
data_ec = np.loadtxt(fname_ec,
                  delimiter=',',
                  skiprows=7,
                  usecols=(2))
data_eo = np.loadtxt(fname_eo,
                  delimiter=',',
                  skiprows=7,
                  usecols=(2))

fig = plt.figure()
sampling_freq = 250
draw_specgram(data_eo, sampling_freq, fig, 1)
draw_specgram(data_ec, sampling_freq, fig, 2)