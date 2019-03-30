#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 14:47:20 2019

@author: jenisha
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from itertools import chain


from scipy.signal import butter, lfilter, find_peaks, peak_widths,iirfilter, detrend,correlate,periodogram,welch

"""
Functions

"""
def merge_dols(dol1, dol2):
    keys = set(dol1).union(dol2)
    no = []
    return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)
def get_data(csvs):
    all_data = {'Right': [], 'Left': [], 'Rest': []}
    for csv in csvs:
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
        all_data = merge_dols(all_data, data)
    return all_data

def filter_2(raw_eeg_data,bp_lowcut =1, bp_highcut =60, bp_order=2,
            notch_freq_Hz  = [60, 120], notch_order =2):
       nyq = 0.5 * 250 #Nyquist frequency
       low = bp_lowcut / nyq
       high = bp_highcut / nyq
       
       #Butter
       b_bandpass, a_bandpass = butter(bp_order, [low , high], btype='band', analog=True)
       
       bp_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_bandpass, a_bandpass ,l),0,raw_eeg_data)

       notch_filtered_eeg_data = bp_filtered_eeg_data
       
       low1  = notch_freq_Hz[0]
       high1 = notch_freq_Hz[1]
       low1  = low1/nyq
       high1 = high1/nyq
       
       b_notch, a_notch = iirfilter(2, [low1, high1], btype='bandstop')
       notch_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_notch, a_notch ,l),
                                                     0,bp_filtered_eeg_data)
       

       return notch_filtered_eeg_data
   
def filter_(arr, fs_Hz=250, lowcut=0.1, highcut=50, order=1):
   nyq = 0.5 * fs_Hz
   b, a = butter(1, [lowcut/nyq, highcut/nyq], btype='band')
   for i in range(0, order):
       arr = lfilter(b, a, arr, axis=0)
   return arr

def find_psd(data,NFFT=500,overlap=0, num_channels=4):
    psd_all = []
    for channel in range(num_channels):  
#        freqs,psd = welch(np.squeeze(data[:,channel]),fs=250, nfft=500)
        psd, freqs =   mlab.psd(np.squeeze(data[:,channel]),
                                           NFFT=NFFT,
                                           window=mlab.window_hanning,
                                           Fs=sample_rate,
                                           noverlap=overlap
                                           ) 
        psd_all.append(psd)
        
    
    return freqs,psd_all
"""
Testing

"""
sample_rate = 250

csv_hasnain = ["../data/March29_014/5_014_eye_blink-2019-3-29-17-44-33.csv",
               "../data/March29_014/6_014_eye_blink-2019-3-29-17-46-14.csv", 
               "../data/March29_014/7_014_eye_blink-2019-3-29-17-47-56.csv"]

csv_map = {"5_014_eye_blink-2019-3-29-17-44-33.csv":"5-7_014_OpenBCI-RAW-2019-03-29_17-41-53.txt",
           "6_014_eye_blink-2019-3-29-17-46-14.csv": "5-7_014_OpenBCI-RAW-2019-03-29_17-41-53.txt",
           "7_014_eye_blink-2019-3-29-17-47-56.csv": "5-7_014_OpenBCI-RAW-2019-03-29_17-41-53.txt"}

#hasnain_data= get_data(csv_hasnain )
raw_data_hasnain_blink = np.concatenate(hasnain_data['Right'])[:,[0,1,6,7]]
raw_data_hasnain_rest = np.concatenate(hasnain_data['Rest'])[:,[0,1,6,7]]

f1_raw, psd_blink = find_psd(raw_data_hasnain_blink)#(detrended_and_filtered_raw_data2_blink)
f2_raw, psd_rest = find_psd(raw_data_hasnain_rest)#(detrended_and_filtered_raw_data2_rest)

f1_detrend, psd_blink_detrend = find_psd(filter_(raw_data_hasnain_blink))
f2_detrend, psd_rest_detrend = find_psd(filter_(raw_data_hasnain_rest))

f1_detrend_filtered, psd_blink_detrend_filtered = find_psd(filter_2(detrend(raw_data_hasnain_blink)))
f2_detrend_filtered, psd_rest_detrend_filtered = find_psd(filter_2(detrend(raw_data_hasnain_rest)))

plt.figure()
for i in range(len(psd_blink)):
   ax1 = plt.subplot(3,2,1)
   ax2 = plt.subplot(3,2,2)
   ax3 = plt.subplot(3,2,3)
   ax4 = plt.subplot(3,2,4)
   ax5 = plt.subplot(3,2,5)
   ax6 = plt.subplot(3,2,6)
   
   ax1.plot(f1_raw, psd_blink[i])
   ax2.plot(f2_raw, psd_rest[i]) 
   
   ax1.set_ylim(0, 50*np.median(psd_blink))
   ax2.set_ylim(0, 50*np.median(psd_rest))
   ax1.set_xlim(0, 20)
   ax2.set_xlim(0, 20)
   
   ax3.plot(f1_detrend, psd_blink_detrend[i])
   ax4.plot(f2_detrend, psd_rest_detrend[i])
   
   ax3.set_ylim(0, 50*np.median(psd_blink_detrend))
   ax4.set_ylim(0, 50*np.median(psd_rest_detrend))
   ax3.set_xlim(0, 20)
   ax4.set_xlim(0, 20)
   
   ax5.plot(f1_detrend_filtered,psd_blink_detrend_filtered[i])
   ax6.plot(f2_detrend_filtered,psd_rest_detrend_filtered[i])
   
   ax5.set_ylim(0, 50*np.median(psd_blink_detrend_filtered))
   ax6.set_ylim(0, 50*np.median(psd_rest_detrend_filtered))
   ax5.set_xlim(0, 20)
   ax6.set_xlim(0, 20)
   
plt.tight_layout()  
 
ax1.set_title("PSD of blinking")
ax2.set_title("PSD of rest")

ax3.set_title("PSD of filtered blinking")
ax4.set_title("PSD of filetered rest")

ax5.set_title("PSD of detrended and filtered blinking")
ax6.set_title("PSD of detrended and filered rest")

indices = np.where(np.logical_and(f1>=5, f2<=15))
for i in range(4):
    print(i)
    print(np.max(psd_blink[i][indices]))
    print(np.max(psd_rest[i][indices]))
    #print(np.mean(psd_blink_detrend[i][indices]))
    #print(np.mean(psd_rest_detrend[i][indices]))
    print(np.max(psd_blink_detrend_filtered[i][indices]))
    print(np.max(psd_rest_detrend_filtered[i][indices]))
    