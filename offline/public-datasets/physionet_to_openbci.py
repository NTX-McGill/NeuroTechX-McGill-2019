#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 23:49:00 2019

@author: roland
"""

'''
TODO:
    Fix the numbers so that they don't 
'''

from mne.datasets import eegbci
from mne.io import concatenate_raws, read_raw_edf
import numpy as np
from scipy.signal import resample
import matplotlib.pyplot as plt

class stolened:
    def __init__(self):
#        self.SUBJECTS = 1
        self.SAMPLING_RATE = 250
#        self.CHANNELS = ["C1", "C2", "C3", "C4", "Cp1", "Cp2", "Cp3", "Cp4"]
        self.CHANNELS = ["C3", "C1", "C2", "C4"]
        self.RUNS = [4,8,12]
        self.out_files = ["test_rest.txt", "test_left.txt", "test_right.txt"]

    def load_nme_data(self):
        raw_fnames = eegbci.load_data(1, self.RUNS)
        
        raw_files = []
        raw_files.extend([read_raw_edf(f, preload=True, stim_channel='auto') for f in raw_fnames])

        raw = concatenate_raws(raw_files)
        raw.rename_channels(lambda x: x.strip('.'))
        
        self.raw = raw
        self.data = raw.get_data()
        self.times = self.raw[-1,:][-1] * 1000
        
    def resample_raw(self):
        self.raw_resampled = self.raw.copy().resample(self.SAMPLING_RATE, npad="auto")
        self.data_resampled = self.raw_resampled.get_data()
        self.times_resampled = self.raw_resampled[-1,:][-1] * 1000
        
    def split_data(self):
        stim = self.raw[self.raw.ch_names.index('STI 014'),:][0]
        
        rest_indices = np.where(stim[0] == 1)[0]
        left_indices = np.where(stim[0] == 2)[0]
        right_indices = np.where(stim[0] == 3)[0]

        self.rest = []
        self.left = []
        self.right = []
        
        for c in self.CHANNELS:
            ch = np.round(self.data_resampled[self.raw_resampled.ch_names.index(c), :] * 1000000, decimals=2)
            self.rest.append(ch[rest_indices])
            self.left.append(ch[left_indices])
            self.right.append(ch[right_indices])
                 
        self.rest_left_right = [self.rest, self.left, self.right]    
        
    def write(self):
#        f = open(self.out_file, "w+")
#        f.write("%OpenBCI Raw EEG Data\n%Number of channels = 8\n%Sample Rate = 250.0 Hz\n%First Column = SampleIndex\n%Last Column = Timestamp \n%Other Columns = EEG data in microvolts followed by Accel Data (in G) interleaved with Aux Data\n")
        
#        C3 = np.round(np.log(self.data_resampled[self.raw_resampled.ch_names.index("C3"), :] * 1000000), decimals=2)
#        C1 = np.round(np.log(self.data_resampled[self.raw_resampled.ch_names.index("C1"), :] * 1000000), decimals=2)
#        C2 = np.round(np.log(self.data_resampled[self.raw_resampled.ch_names.index("C2"), :] * 1000000), decimals=2)
#        C4 = np.round(np.log(self.data_resampled[self.raw_resampled.ch_names.index("C4"), :] * 1000000), decimals=2)
        
#        C3 = np.round(self.data_resampled[self.raw_resampled.ch_names.index("C3"), :] * 100000, decimals=2)
#        C1 = np.round(self.data_resampled[self.raw_resampled.ch_names.index("C1"), :] * 100000, decimals=2)
#        C2 = np.round(self.data_resampled[self.raw_resampled.ch_names.index("C2"), :] * 100000, decimals=2)
#        C4 = np.round(self.data_resampled[self.raw_resampled.ch_names.index("C4"), :] * 100000, decimals=2)
#
#
##        for i in range(len(self.times_resampled)):
#        for i in range(10000):
#            line = []
#            
#            line.append(str(i % 256))
#            line.append(str(C3[i]))
#            line.append(str(C1[i]))
#            line.extend(["0.00" for j in range(4)])
#            line.append(str(C2[i]))
#            line.append(str(C4[i]))
#            line.extend(["0.000" for j in range(3)])
#            line.append("0")
#            line.append(str(int(self.times_resampled[i])))
#            
#            write_line = ', '.join(line) + '\n'
#            f.write(write_line)
#            
#        f.close()
        
        for i in range(3):
            f = open(self.out_files[i], "w+")
            f.write("%OpenBCI Raw EEG Data\n%Number of channels = 8\n%Sample Rate = 250.0 Hz\n%First Column = SampleIndex\n%Last Column = Timestamp \n%Other Columns = EEG data in microvolts followed by Accel Data (in G) interleaved with Aux Data\n")
            
            data = self.rest_left_right[i]
            
            for j in range(len(data[i])):
                line = []
            
                line.append(str(j % 256))
                line.append(str(data[0][j]))
                line.append(str(data[1][j]))
                line.extend(["0.00" for k in range(4)])
                line.append(str(data[2][j]))
                line.append(str(data[3][j]))
                line.extend(["0.000" for k in range(3)])
                line.append("0")
                line.append(str(int(self.times_resampled[j])))
                
                write_line = ', '.join(line) + '\n'
                f.write(write_line)
                
            f.close()
            

    def plot_line_graph(self, times, channel):
        fig = plt.figure()
        ax = plt.axes()
        ax.plot(times, channel)
        plt.show()
        
    def compare(self):
        fig = plt.figure()
        
        plt.title("Original Data")
        ax = plt.axes()
        ax.plot(self.times, self.raw[0,:][0][0])
        plt.xlabel('Time (sec)')
        plt.show()
        
        fig = plt.figure()
        plt.title("Resampled Data")
        ax = plt.axes()
        ax.plot(self.times_resampled, self.raw_resampled[0,:][0][0])
        plt.xlabel('Time (sec)')
        plt.show()

if __name__ == "__main__":
    test = stolened()
    test.load_nme_data()
    test.resample_raw()
    #test.compare()
    test.split_data()
    test.write()

