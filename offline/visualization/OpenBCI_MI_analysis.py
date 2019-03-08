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

def plot_specgram(spec_freqs, spec_PSDperBin,title,shift,i=1):
    f_lim_Hz = [0, 20]   # frequency limits for plotting
    #plt.figure(figsize=(10,5))
    spec_t = [idx*.1 for idx in range(len(spec_PSDperBin[0]))]
    plt.subplot(3,1,i)
    plt.title(title)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25,26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.subplots_adjust(hspace=1)

def resize_min(specgram, i=1):
    min_length = min([len(el[0]) for el in specgram])
    specgram = np.array([el[:, :min_length] for el in specgram])
    return specgram
def resize_max(specgram, fillval=np.nan):
    max_length = max([len(el[0]) for el in specgram])
    return np.array([pad_block(el, max_length,fillval) for el in specgram])
def pad_block(block, max_length, fillval):
    padding = np.full([len(block), max_length-(len(block[0]))], fillval)
    return np.hstack((block,padding))

fname = '../data/March 4/5_SUCCESS_Rest_RightAndJawClench_10secs.txt' 
#fname = '../data/March 4/6_SUCCESS_Rest_RightClench_JawClench_ImagineClench_10secs.txt' 
fname = '../data/March 4/7_SUCCESS_Rest_RightClenchImagineJaw_10secs.txt'
sampling_freq = 250
shift = 0.1
channel = (1)
channel_name = 'C4'
continuous = False

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
fig = plt.figure()
plot_specgram(f, all_spectra, "entire session", shift, 1)

if continuous:
    for i in range(len(start_indices) - 1):
        start = int(start_indices[i]/(sampling_freq * shift))
        end = int(start_indices[i+1]/(sampling_freq * shift))
        d = all_spectra[:,start:end]
        # this trial alternates between rest and left motor imagery
        if i % 2:
            left_specgram.append(d)
        else:
            rest_specgram.append(d)
else:
    #tmin, tmax = -1, 1
    tmin, tmax = 0, 0
    plt.figure()
    for i in range(len(start_indices) - 1):
        start = int(max(start_indices[i] + tmin * sampling_freq, 0))
        end = int(min(start_indices[i+1] + tmax * sampling_freq, start_indices[-1]))
        t, f, d = get_spectral_content(ch[start:end], sampling_freq)
        if i < 2:
            # plot two sample epochs for fun
            plot_specgram(f, d, 'a', shift, i + 1)
        if i % 2:
            left_specgram.append(d)
        else:
            rest_specgram.append(d)
#resize the blocks so that they're the same length as either the minimum or maximum length block
'''rest_specgram = resize_min(rest_specgram)
left_specgram = resize_min(left_specgram)
'''
rest_specgram = resize_max(rest_specgram)
left_specgram = resize_max(left_specgram)

np.save('spec_rest.npy', rest_specgram)
np.save('spec_left.npy', left_specgram)

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
#plt.figure()
#plt.plot(data)

'''
        
        
plt.figure()
plt.plot(data_eo)
plt.figure()
draw_specgram(ch, sampling_freq, fig, 1)
'''
