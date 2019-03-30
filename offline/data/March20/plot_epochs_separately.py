#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:13:51 2019

@author: marley
"""

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
from scipy import signal

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

def draw_specgram(spec_freqs, spec_PSDperBin, fig, num_subplots, i, title):
    spec_t = [idx*.1 for idx in range(len(spec_PSDperBin[0]))]
    f_lim_Hz = [0, 20]   # frequency limits for plotting
    #plt.figure(figsize=(10,5))
    plt.subplot(num_subplots,2,i)
    #plt.title(title)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25,26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    # plt.xlabel('Time (sec)') aint got space for this
    plt.ylabel('Frequency (Hz)')
    #plt.subplots_adjust(hspace=1)
    #plt.gca().set_aspect('equal', adjustable='box')

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
fs_Hz = 250
#fname = 'time-test-JingMingImagined10s-2019-3-20-10-12-1.csv'

csv = "time-test-JingMingActual_10s-2019-3-20-10-51-28.csv"
csv = "time-test-JingMingActual-2019-3-20-9-54-59.csv"
csv = "time-test-JingMingActual-2019-3-20-10-5-15.csv"
csv = "time-test-JingMingImagined10s-2019-3-20-10-12-1.csv"
#csv = "time-test-JingMingImaginedREALLYGOOD-2019-3-20-10-21-44.csv"
df = pd.read_csv(csv)

fname = 'OpenBCI-RAW-2019-03-20_10-04-29.txt'

data = np.loadtxt(fname,
                  delimiter=',',
                  skiprows=7,
                  usecols=channel)
eeg = data[:,:-1]
timestamps = data[:,-1]
#indices = np.where(np.logical_and(timestamps>=start, timestamps<=end))
#eeg = eeg[indices]


#a = df.loc[df['Direction'] == 'Right'][['Channel 1','Channel 2']]
'''
a = df[['Channel 1','Channel 2']]
data = np.copy(a.values)
data = filter_(data, sampling_freq,1,40,1)
print(data.shape)

fig = plt.figure(figsize=(20,10))
length = 60
data.resize((int(len(data)/(250 * length)) + 1, 250 * length,data.shape[-1]))'''

prev = 0
prev_direction = df['Direction'][prev]
all_data = {'Right': [], 'Left': [], 'Rest': []}
for idx,el in enumerate(df['Direction']):
    if el != prev_direction:
        start = df['Time'][prev]
        end = df['Time'][idx]
        indices = np.where(np.logical_and(timestamps>=start, timestamps<=end))
        trial = eeg[indices]
        all_data[prev_direction].append(trial)
        print(idx - prev, prev_direction)
        print(len(trial))
        prev = idx
        prev_direction = el
data = all_data['Rest']
set_1 = [0,1,2,3]
set_2 = [4,5,6,7]
for direction, data in all_data.items():
    figname = direction + '_spec_' + csv.split('.')[0]
    fig = plt.figure(figname)
    idx = 0
    left = []
    right = []
    for block in data:
        all_spectra = []
        filtered_block = filter_(block, sampling_freq, 1,40,1)
        for channel in filtered_block.T:
            t,f,d = get_spectral_content(channel, fs_Hz)
            all_spectra.append(d)
        all_spectra = np.array(all_spectra)
        left.append(np.mean(all_spectra[set_1], axis=0))
        right.append(np.mean(all_spectra[set_2], axis=0))
    idx = 0
    num_rows = max(len(left), 5)
    for spec in left:
        idx += 1
        draw_specgram(f, spec, fig, num_rows, idx * 2 - 1, fname)
    idx = 0
    for spec in right:
        idx += 1
        draw_specgram(f, spec, fig, num_rows, idx * 2, fname)
    plt.savefig(figname)
'''
for idx, fname in enumerate(filenames):
    if not idx % 6:
        fig = plt.figure(figsize=(8,10))
    data = np.loadtxt(fname,
                      delimiter=',',
                      skiprows=7,
                      usecols=channel)
    draw_specgram(data, sampling_freq, fig, 6,idx%6 + 1,fname)'''