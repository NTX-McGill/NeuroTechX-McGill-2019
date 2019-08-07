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

import sys
sys.path.append('../utils')
from metadata import SAMPLING_FREQ, ELECTRODE_C3, ELECTRODE_C4
import file_utils
import preprocessing as prepro

def plot_specgram(spec_freqs, spec_PSDperBin,title,shift,i=1):
    f_lim_Hz = [0, 20]   # frequency limits for plotting
    spec_t = [idx*.1 for idx in range(len(spec_PSDperBin[0]))]
    plt.title(title)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25,26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.subplots_adjust(hspace=1)

fname = '../data/March 4/5_SUCCESS_Rest_RightAndJawClench_10secs.txt' 
#fname = '../data/March 4/6_SUCCESS_Rest_RightClench_JawClench_ImagineClench_10secs.txt' 
fname = '../data/March 4/7_SUCCESS_Rest_RightClenchImagineJaw_10secs.txt'
#fname = '../data/March 4/8_SUCCESS_Left_Right_Rest_10secs_3mins_total.txt'
shift = 0.1
channel = (1)
channel_name = 'C4'
continuous = False

eeg, timestamps = file_utils.load_openbci_raw(fname)
data = eeg[:,ELECTRODE_C3]
data = prepro.filter_signal(data.T, 1, 40, 1)
        
ch = data
start_indices = prepro.get_artifact_indices(ch)

right_specgram = []
left_specgram = []
rest_specgram = []

t,f,all_spectra = prepro.get_spectral_content(ch, SAMPLING_FREQ, shift)
fig = plt.figure()
plt.subplot(3,1,1)
plot_specgram(f, all_spectra, "entire session", shift, 1)

tmin, tmax = 0, 0
plt.figure()
for i in range(len(start_indices) - 1):
    start = int(max(start_indices[i] + tmin * SAMPLING_FREQ, 0))
    end = int(min(start_indices[i+1] + tmax * SAMPLING_FREQ, start_indices[-1]))
    t, f, d = prepro.get_spectral_content(ch[start:end], SAMPLING_FREQ)
    if i < 2:
        # plot two sample epochs for fun
        plot_specgram(f, d, 'a', shift, i + 1)
    if i % 2:
        left_specgram.append(d)
    else:
        rest_specgram.append(d)
#resize the blocks so that they're the same length as either the minimum or maximum length block
rest_specgram = prepro.resize_max(rest_specgram)
left_specgram = prepro.resize_max(left_specgram)

# plot average spectrogram of both classes
plt.figure()
rest_av = np.nanmean(np.array(rest_specgram), axis=0)
left_av = np.nanmean(np.array(left_specgram), axis=0)
plot_specgram(f, rest_av,channel_name + ' rest',shift, 1)
plot_specgram(f, left_av,channel_name + ' left',shift, 2)

# plot average mu trace over time
fig = plt.figure()
mu_indices = np.where(np.logical_and(f>=7, f<=12))
plt.plot(np.mean(rest_av[mu_indices], axis=0))
plt.plot(np.mean(left_av[mu_indices], axis=0))

# get mu trace for each trial
rest_mu = np.mean(rest_specgram[:,mu_indices[0],:], axis=1)
left_mu = np.mean(left_specgram[:,mu_indices[0],:], axis=1)

# plot mu trace for each trial over time
plt.figure()
for spec in rest_mu:
    plt.plot(spec, 'm')
for spec in left_mu:
    plt.plot(spec, 'c')

# histogram of mu levels
colors = ['m', 'c']
labels = ['rest', 'left']
rest_mu_vals = rest_mu[~np.isnan(rest_mu)].flatten()
left_mu_vals = left_mu[~np.isnan(left_mu)].flatten()
plt.figure()
plt.hist([rest_mu_vals, left_mu_vals], bins=[i*.1 for i in range(40)], color=colors, label=labels)
plt.legend()

threshold = 0.4
percent_rest = len(np.where(rest_mu_vals < threshold)[0])/len(rest_mu_vals)
percent_left = len(np.where(left_mu_vals < threshold)[0])/len(left_mu_vals)
print(percent_rest)
print(percent_left)

plt.figure()
i = 0
for spec in np.where(rest_mu < threshold, 1, 0):
    i += 1
    plt.subplot(19, 1, i)
    plt.plot(spec, 'm')
for spec in np.where(left_mu < threshold, 1, 0):
    i += 1
    plt.subplot(19, 1, i)
    plt.plot(spec, 'c')