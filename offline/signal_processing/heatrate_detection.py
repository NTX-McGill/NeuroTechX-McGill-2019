#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 26 15:48:35 2019

@author: jenisha
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from itertools import chain


from scipy.signal import butter, lfilter, find_peaks, peak_widths

path='/Users/jenisha/Desktop/NeuroTechX-McGill-2019/offline/data/March24_011/'
fname= path +  '6_011_EyeBlink-2019-3-24-17-8-26.csv'
sample_rate = 250
name_channel = ["Channel 1","Channel 2","Channel 7","Channel 8"]
csvs = {""}


df_all = pd.read_csv(fname)
df_state = df_all.loc[:,["Direction"]]
list_states = list(chain.from_iterable(df_state.values.tolist()))
list_states_binary = [0 if state == "Rest" else 1 for state in list_states]



raw_data = np.asarray(df_all.loc[:,name_channel])
csv_map = {"6_011_EyeBlink-2019-3-24-17-8-26.csv": "011_4to6_OpenBCI-RAW-2019-03-24_16-54-15.txt"}
def merge_dols(dol1, dol2):
    keys = set(dol1).union(dol2)
    no = []
    return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)
def get_data(csvs):
    all_data = {'Right': [], 'Left': [], 'Rest': []}
    for csv in csvs:
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

csv = [fname]
all_data = get_data(csv)

def filter_(raw_eeg_data,bp_lowcut =1, bp_highcut =70, bp_order=2,
            notch_freq_Hz  = [60.0, 120.0], notch_order =2):
       nyq = 0.5 * 250 #Nyquist frequency
       low = bp_lowcut / nyq
       high = bp_highcut / nyq
       
       #Butter
       b_bandpass, a_bandpass = butter(bp_order, [low , high], btype='band', analog=True)
       
       bp_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_bandpass, a_bandpass ,l),0,raw_eeg_data)

       notch_filtered_eeg_data = bp_filtered_eeg_data
       
       for freq_Hz in notch_freq_Hz: 
            bp_stop_Hz = freq_Hz + float(notch_order)*np.array([-1, 1])  # set the stop band
            b_notch, a_notch = butter(notch_order, bp_stop_Hz/nyq , 'bandstop')
            notch_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_notch, a_notch,l),0,notch_filtered_eeg_data)
       
        
       return notch_filtered_eeg_data
   

filtered_data = filter_(raw_data)





def plots(raw_eeg_data, corrected_eeg_data, num_channels=1, NFFT=500,overlap=125):
        """
       
        Plot the raw and filtered data 
        
        Input:
            - num_channels: number of channels to plot
        
        """
        
        

        fig = plt.figure()
        for channel in range(num_channels):  
            #fig = plt.figure()

            #fig.suptitle(self.name_channel[channel])
    
            t_sec = np.array(range(0, raw_eeg_data[:,channel].size)) / sample_rate
            
            ax1 = plt.subplot(221)
            peaks, _ = find_peaks(raw_eeg_data[:,channel], distance=150, wlen=250)
            
            plt.plot(t_sec, raw_eeg_data[:,channel],label=name_channel[channel])
            plt.plot(peaks/ sample_rate, raw_eeg_data[:,channel][peaks], "x")
            plt.ylabel('EEG (uV)')
            plt.xlabel('Time (sec)')
            plt.title('Raw')
            plt.xlim(t_sec[0], t_sec[-1])
            ax1.pcolorfast(ax1.get_xlim(), ax1.get_ylim(),
              np.asarray(list_states_binary)[np.newaxis])
            
            raw_spec_PSDperHz, raw_freqs, raw_t_spec =  mlab.specgram(
                                           np.squeeze(raw_eeg_data[:,channel]),
                                           NFFT=NFFT,
                                           window=mlab.window_hanning,
                                           Fs=sample_rate,
                                           noverlap=overlap
                                           ) 
            raw_spec_PSDperBin = raw_spec_PSDperHz * sample_rate / float(NFFT)
            ax2 = plt.subplot(222)
            plt.pcolor(raw_t_spec, raw_freqs, 10*np.log10(raw_spec_PSDperBin))
            plt.clim(25-5+np.array([-40, 0]))
            plt.xlim(t_sec[0], t_sec[-1])
            plt.ylim([0, 100]) 
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectogram of Unfiltered')
            
            
#            psd,freqs = mlab.psd(np.squeeze(raw_eeg_data[:,channel]),NFFT=500,Fs=250)    
#            ax2 = plt.subplot(222)
#            plt.xlim(t_sec[0], t_sec[-1])
#            plt.xlabel('Frequency (Hz)')
#            plt.ylabel('Power')
#            median = np.median(psd)
#            plt.ylim(0,10*median)
#            plt.plot(freqs,psd,label=name_channel[channel])
#            plt.title('PSD of Unfiltered')
            
            
            
            
            ax3 = plt.subplot(223)
            plt.plot(t_sec, corrected_eeg_data[:,channel],label=name_channel[channel])
            plt.ylabel('EEG (uV)')
            plt.xlabel('Time (sec)')
            plt.title('Filtered')
            plt.ylim(-6900,-6600)
            plt.xlim(t_sec[0], t_sec[-1])
            ax3.pcolorfast(ax3.get_xlim(), ax3.get_ylim(),
              np.asarray(list_states_binary)[np.newaxis])


            
            corrected_spec_PSDperHz, corrected_freqs, corrected_t_spec = mlab.specgram(
                                           np.squeeze(raw_eeg_data[:,channel]),
                                           NFFT=NFFT,
                                           window=mlab.window_hanning,
                                           Fs=sample_rate,
                                           noverlap=overlap
                                           ) 
            corrected_spec_PSDperBin = corrected_spec_PSDperHz * sample_rate / float(NFFT)      
            ax4 = plt.subplot(224)
            plt.pcolor(corrected_t_spec, corrected_freqs, 
                   10*np.log10(corrected_spec_PSDperBin))
            plt.clim(25-5+np.array([-40, 0]))
            plt.xlim(t_sec[0], t_sec[-1])
            plt.ylim([0, 100])  
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency (Hz)')
            plt.title('Spectogram of Filtered')
            
#            psd,freqs = mlab.psd(np.squeeze(corrected_eeg_data[:,channel]),NFFT=500,Fs=250)    
#            ax4 = plt.subplot(224)
#            plt.xlim(t_sec[0], t_sec[-1])
#            plt.xlabel('Frequency (Hz)')
#            plt.ylabel('Power')
#            median = np.median(psd)
#            plt.ylim(0,10*median)
#            plt.plot(freqs,psd,label=name_channel[channel])
#            plt.title('PSD of filtered')
            
        plt.legend(name_channel,loc='upper right', bbox_to_anchor=(0.01, 0.01))
        plt.tight_layout()
        plt.show()
   
    

    
plots(raw_data,filtered_data)    



