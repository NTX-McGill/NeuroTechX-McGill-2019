#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 12:06:57 2019

@author: marley
"""
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd

def get_data(csv):
    all_data = {'Right': [], 'Left': [], 'Rest': []}
    print("loading " + csv)
    path_c = csv.split('/')
    fname = "/".join(path_c[:-1] + [csv_map[path_c[-1]]])
    df = pd.read_csv(csv)
    channel = (1,2,3,4,5,6,7,8,13)
    data = np.loadtxt(fname,
                  delimiter=',',
                  skiprows=7,
                  usecols=channel)
    eeg = data[:,:-1]
    timestamps = data[:,-1]
    prev = 0
    prev_direction = df['Direction'][prev]
    data = {'Right': [], 'Left': [], 'Rest': []}
    for idx,el in enumerate(df['Direction']):
        if el != prev_direction or idx == len(df.index) - 1:
            start = df['Time'][prev]
            end = df['Time'][idx]
            indices = np.where(np.logical_and(timestamps>=start, timestamps<=end))
            trial = eeg[indices]
            all_data[prev_direction].append(trial)
            #print(idx - prev, prev_direction)
            #print(len(trial))
            prev = idx
            prev_direction = el
    return all_data

csv_map = {"10_008-2019-3-22-15-8-55.csv": "10_008_OpenBCI-RAW-2019-03-22_15-07-58.txt",
           "9_008-2019-3-22-14-59-0.csv": "8to9_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",
           "8_008-2019-3-22-14-45-53.csv": "8to9_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",
           "7_008-2019-3-22-14-27-46.csv": "4to7_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",  # actual
           "6_008-2019-3-22-14-19-52.csv": "4to7_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",  # actual
           "5_008-2019-3-22-14-10-26.csv": "4to7_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",  # actual
           "5-001-rest25s_left10s_right10s_MI-2019-3-22-16-35-57.csv": "1to5_001_OpenBCI-RAW-2019-03-22_15-56-26.txt",
           "4-001-rest25s_left15s_right15s_MI-2019-3-22-16-27-44.csv": "1to5_001_OpenBCI-RAW-2019-03-22_15-56-26.txt",
           "6-001-rest25s_left15s_right15s_MI-2019-3-22-16-46-17.csv": "6to7_001_OpenBCI-RAW-2019-03-22_16-44-46.txt",  # actual
           "7-001-rest25s_left20s_right20s_MI-2019-3-22-16-54-17.csv": "6to7_001_OpenBCI-RAW-2019-03-22_16-44-46.txt",
           "time-test-JingMingImagined_10s-2019-3-20-10-28-35.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
           "time-test-JingMingImagined_10s-2019-3-20-10-30-26.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
           "time-test-JingMingImagined_10s-2019-3-20-10-35-31.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
           "1_011_Rest20LeftRight20_MI-2019-3-24-16-25-41.csv" : '011_1to3_OpenBCI-RAW-2019-03-24_16-21-59.txt',
           "2_011_Rest20LeftRight20_MI-2019-3-24-16-38-10.csv" : '011_1to3_OpenBCI-RAW-2019-03-24_16-21-59.txt',
           "3_011_Rest20LeftRight10_MI-2019-3-24-16-49-23.csv" : '011_1to3_OpenBCI-RAW-2019-03-24_16-21-59.txt',
           "4_011_Rest20LeftRight10_MI-2019-3-24-16-57-8.csv" : '011_4to6_OpenBCI-RAW-2019-03-24_16-54-15.txt',
           "5_011_Rest20LeftRight20_MI-2019-3-24-17-3-17.csv" : '011_4to6_OpenBCI-RAW-2019-03-24_16-54-15.txt'
           }

# specify csv name below
#csv_name = "data/March24_011/5_011_Rest20LeftRight20_MI-2019-3-24-17-3-17.csv"
#csv_name = "data/March22_008/10_008-2019-3-22-15-8-55.csv"
csv_name = "data/March22_008/8_008-2019-3-22-14-45-53.csv"
train_data = get_data(csv_name)
fig1 = plt.figure("psds")
fig1.clf()
fig2 = plt.figure("separate psds")
fig2.clf()
idx = 1
for direction, data in train_data.items():
    l = np.hstack([trial[:,0] for trial in data])
    r = np.hstack([trial[:,-1] for trial in data])
    psd1, freqs = mlab.psd(np.squeeze(l),
                              NFFT=2048,
                              noverlap=250,
                              Fs=250)
    psd2, freqs = mlab.psd(np.squeeze(r),
                              NFFT=2048,
                              noverlap=250,
                              Fs=250)
    plt.figure("psds")
    plt.subplot(211)
    plt.title("electrode 1")
    plt.plot(freqs,psd1,label=direction,linewidth=0.5)
    plt.ylim([0,25])
    plt.xlim([0,20])
    plt.legend()
    plt.subplot(212)
    plt.title("electrode 8")
    plt.plot(freqs,psd2,label=direction,linewidth=0.5)
    plt.ylim([0,25])
    plt.xlim([0,20])
    plt.legend()
    plt.subplots_adjust(hspace=0.5)
    
    plt.figure("separate psds")
    plt.subplot(3,2,idx)
    plt.title(direction)
    plt.plot(freqs, psd1,linewidth=0.5)
    plt.ylim([0,25])
    plt.xlim([6,20])
    plt.subplot(3,2,idx+1)
    plt.plot(freqs, psd2, linewidth=0.5)
    plt.ylim([0,25])
    plt.xlim([6,20])
    idx += 2
    
plt.show()