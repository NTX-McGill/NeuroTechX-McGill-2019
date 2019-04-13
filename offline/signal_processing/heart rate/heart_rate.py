#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 31 21:43:26 2019

@author: jenisha


"""




import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def analyze_ecg_data(path,bound=0, channel=[1], sample_rate=250, 
                     distance=100, prominence=0.6):
    """
    Analyzes ECG data to determine threshold values for peaks
    
    Input:
        path: where the data is located
        bound: where legit data starts
        channel: should be the index of the channel where ECG is recorded
        sample_rate: # of samples collected per second
        
        
        For find_peaks function (values to play with):
            distance: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks
            prominence :Required prominence of peaks
          
    """
    #Assume data is of size number of samples x 1 channel
    
    #Load raw data into numpy array
    raw_ecg_data = np.loadtxt(path, delimiter=',',skiprows=7,usecols=channel)
    
    #Correct ecg data
    corrected_ecg_data = -raw_ecg_data  + np.max(raw_ecg_data)
    
    #Time
    n_samples = raw_ecg_data.shape[0]
    time = np.array(range(0, n_samples)) /sample_rate
    
    #Lower bound
    lower_bound = bound *sample_rate
    ecg_data_bounded  = corrected_ecg_data[lower_bound::]
    time_bounded = time[lower_bound::]

    #Detrend
    fit = np.polyfit(time_bounded,ecg_data_bounded ,1)
    ecg_data_normalized = ecg_data_bounded - (fit[0]*time_bounded+ fit[1])
    
    #Plots
    fig = plt.figure()

    ax1= plt.subplot(2,1,1)
    ax1.plot(time_bounded, ecg_data_normalized)
    peaks, _ = find_peaks(ecg_data_normalized,distance=distance,prominence=prominence)
    ax1.plot(bound + peaks/ sample_rate, ecg_data_normalized[peaks], "x")

    ax2= plt.subplot(2,1,2)
    ax2.plot(time, corrected_ecg_data)
    peaks, _ = find_peaks(corrected_ecg_data,distance=distance,prominence=prominence)
    ax2.plot(peaks/ sample_rate, corrected_ecg_data[peaks], "x")

    


def hearbeat_real_time_test(path,time_epoch = 2,distance=100,prominence=0.6):
    """
    Testing if heart rate calculation works with real-time epoching
    
    Input:
        path: where the data is located
        time_epoch: number of seconds for each epoch
        
    For find_peaks function (values to play with):
            distance: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks
            prominence :Required prominence of peaks
        

    
    """
    raw_ecg_data = np.loadtxt(path, delimiter=',',skiprows=7,usecols=[1])
    
    #Epoch
    epoch = time_epoch  * sample_rate
    
    i = 0
    while(i + epoch < raw_ecg_data.shape[0]):
        window = raw_ecg_data[i:i+epoch]
        #corrected window:
        window = -window  + np.max(window)
        
        #peaks
        peaks = find_peaks(window,distance=distance,prominence=prominence)[0]    
        number_peaks = len(peaks)
        
        # Heart shit starts here:
        
        #RR interval
        rr = np.diff(peaks)
        # RMSSD
        rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))     
        # Mean RR
        mean_rr = np.mean(rr)       
        # SDNN
        sdnn = np.std(rr)    
        # Mean HR
        mean_hr = 60 * 250/mean_rr
        print(number_peaks,rmssd,mean_rr,sdnn,mean_hr )
     
        i = i + epoch 
        
def copy_paste_this_code_in_real_time(window_input, sampling_rate=250, threshold=110, 
                                      distance =100,prominence=0.6 ):
    
        """
    
    Input:
        window_input: should be of form number_of_samples * 1
        sampling_rate: # of samples collected per second
        threshold: which heart beat is considered too high
        
    For find_peaks function (values to play with):
            distance: Required minimal horizontal distance (>= 1) in samples between neighbouring peaks
            prominence :Required prominence of peaks
        

    
    """
    # Assume window_input is n_samples * 1
    
    window_input = -window_input  + np.max(window_input)
    peaks = find_peaks(window_input,distance=distance,prominence=prominence)[0]
   
    #RR interval
    rr = np.diff(peaks)
    # RMSSD
    rmssd = np.sqrt(np.mean(np.square(np.diff(rr))))        
    # Mean RR
    mean_rr = np.mean(rr)      
    # SDNN
    sdnn = np.std(rr)        
    # Mean HR per minute
    mean_hr = 60 * sampling_rate/mean_rr
    
    if mean_hr > threshold:
        print("gorll, u in danga")


path='../data/March24_013/'
fname= path +  '1_heartrate_jumpscare_OpenBCI-RAW-2019-03-24_11-17-25.txt'
#fname = path + '2_heartrate_mildhorrormovietrailer_OpenBCI-RAW-2019-03-24_11-29-30.txt'
#fname = path + '3_heartrate_horrormvoietrailercompilation_OpenBCI-RAW-2019-03-24_11-31-33.txt'

sample_rate = 250
analyze_ecg_data(fname)
hearbeat_real_time_test(fname)




