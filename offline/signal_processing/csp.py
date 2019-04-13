#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 20:49:48 2019

@author: jenisha
"""
import numpy as np
import pandas as pd

from mne.decoding import CSP
from mne import Epochs, pick_types, find_events
from mne.channels import read_layout
from mne.io import concatenate_raws, read_raw_edf
from mne.datasets import eegbci

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

csv_name = "../data/March22_008/10_008-2019-3-22-15-8-55.csv"
train_data = get_data(csv_name)

# cue onset.
tmin, tmax = -1., 4.
event_id = dict(hands=2, feet=3)
subject = 1
runs = [6, 10, 14]  # motor imagery: hands vs feet

raw_fnames = eegbci.load_data(subject, runs)
raw_files = [read_raw_edf(f, preload=True, stim_channel='auto') for f in
             raw_fnames]
raw = concatenate_raws(raw_files)

# strip channel names of "." characters
raw.rename_channels(lambda x: x.strip('.'))

# Apply band-pass filter
raw.filter(7., 30., fir_design='firwin', skip_by_annotation='edge')

events = find_events(raw, shortest_event=0, stim_channel='STI 014')

picks = pick_types(raw.info, meg=False, eeg=True, stim=False, eog=False,
                   exclude='bads')

# Read epochs (train will be done only between 1 and 2s)
# Testing will be done with a running classifier
epochs = Epochs(raw, events, event_id, tmin, tmax, proj=True, picks=picks,
                baseline=None, preload=True)
epochs.drop_channels(['Fc5','Fc3','Fc1','Fcz','Fc2','Fc4','Fc6','C5','Cz','C6','Cp5','Cp3','Cp1','Cpz',
 'Cp2','Cp4', 'Cp6','Fp1','Fpz','Fp2','Af7','Af3','Afz','Af4','Af8', 'F7','F5','F3', 'F1', 'Fz', 'F2',
 'F4', 'F6','F8','Ft7','Ft8','T7','T8','T9','T10','Tp7','Tp8', 'P7','P5','P3','P1','Pz','P2','P4','P6',
 'P8','Po7','Po3','Poz','Po4','Po8','O1','Oz','O2','Iz'])


csp = CSP(n_components=4, reg=None, log=True, norm_trace=False)
l =np.array(train_data["Left"][:-2]).swapaxes(-1,-2)
r = np.array(train_data["Right"][:-1]).swapaxes(-1,-2)
n = np.array(train_data["Rest"][1:-1]).swapaxes(-1,-2).reshape(4,8,2525)

epoch2 = np.concatenate([l2,r2,n2])
csp.fit_transform(epoch2,[1,1,1,1,1,0,0,0,0])
layout = read_layout('EEG1005')
csp.plot_patterns(epochs.info, layout=layout, ch_type='eeg',
                  units='Patterns (AU)', size=1.5)