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
import pandas as pd

def draw_specgram(ch, fs_Hz, fig, num_subplots, i, title):
    NFFT = fs_Hz*2
    overlap = NFFT - int(0.1 * fs_Hz)
    spec_PSDperHz, spec_freqs, spec_t = mlab.specgram(np.squeeze(ch),
                                   NFFT=NFFT,
                                   window=mlab.window_hanning,
                                   Fs=fs_Hz,
                                   noverlap=overlap
                                   ) # returns PSD power per Hz
    # convert the units of the spectral data
    print(len(spec_t))
    spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)
    f_lim_Hz = [0, 20]   # frequency limits for plotting
    #plt.figure(figsize=(10,5))
    plt.subplot(num_subplots,1,i)
    #plt.title(title)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25,26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    # plt.xlabel('Time (sec)') aint got space for this
    plt.ylabel('Frequency (Hz)')
    #plt.subplots_adjust(hspace=1)
    #plt.gca().set_aspect('equal', adjustable='box')
from scipy import signal

def filter_(arr, fs_Hz, lowcut, highcut, order):
   nyq = 0.5 * fs_Hz
   b, a = signal.butter(1, [lowcut/nyq, highcut/nyq], btype='band')
   for i in range(0, order):
       arr = signal.lfilter(b, a, arr, axis=0)
   return arr
filenames = sorted([f for f in glob.glob("*.txt")])
sampling_freq = 250
shift = 0.1
channel = (1,2,3,4,5,6,7,8,13)
fname = '1_rest_left_right_10s_OpenBCI-RAW-2019-03-15_18-09-34.txt'
fname = '2_RestLeftRight_10s_OpenBCI-RAW-2019-03-15_18-27-01.txt'
markers = '2_RestLeftRight_10s_time-stamp-64-2019-2-15-18-31-46.csv'
#fname = '3_restleftright_10s_OpenBCI-RAW-2019-03-15_18-46-11.txt'
#markers = '3_restleftright_10s_time-stamp-65-2019-2-15-18-52-10.csv'
#fname = 'Marley_prolonged_trial.txt'
#fname = 'OpenBCI-RAW-2019-03-11_17-21-55.txt'
#fname = '../March 4/5_SUCCESS_Rest_RightAndJawClench_10secs.txt'
df = pd.read_csv(markers)
start = df['START TIME'].iloc[0]
end = df['START TIME'].iloc[-1]

data = np.loadtxt(fname,
                  delimiter=',',
                  skiprows=7,
                  usecols=channel)
eeg = data[:,:-1]
timestamps = data[:,-1]
indices = np.where(np.logical_and(timestamps>=start, timestamps<=end))
eeg = eeg[indices]

data = filter_(eeg, sampling_freq, 1, 40, 1)

# TODO: multi channel average
fig = plt.figure(figsize=(8,10))
length = 60
data.resize((int(len(data)/(250 * length)) + 1, 250 * length,data.shape[-1]))
num_rows = data.shape[0] * data.shape[-1]
idx = 0
for block in data:
    for channel in block.T:
        idx += 1
        draw_specgram(channel, sampling_freq, fig, num_rows, idx, fname)
plt.savefig('prolonged_trials.png', dpi=500)

'''
for idx, fname in enumerate(filenames):
    if not idx % 6:
        fig = plt.figure(figsize=(8,10))
    data = np.loadtxt(fname,
                      delimiter=',',
                      skiprows=7,
                      usecols=channel)
    draw_specgram(data, sampling_freq, fig, 6,idx%6 + 1,fname)'''
plt.show()