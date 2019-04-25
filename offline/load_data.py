# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:25:23 2019

@author: Danielle
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import scale

def merge_all_dols(arr):
    all_data = {'Right': [], 'Left': [], 'Rest': []}
    for dol in arr:
        all_data = merge_dols(all_data, dol)
    return all_data


def merge_dols(dol1, dol2):
    keys = set(dol1).union(dol2)
    no = []
    return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)

def get_data(csvs, tmin=0):
    all_data = {'Right': [], 'Left': [], 'Rest': []}
    for csv in csvs:
        print("loading " + csv)
        path_c = csv.split('/')
        fname = "/".join(path_c[:-1] + [csv_map[path_c[-1]]])
        df = pd.read_csv(csv)
        channel = (1, 2, 3, 4, 5, 6, 7, 8, 13)
        data = np.loadtxt(fname,
                          delimiter=',',
                          skiprows=7,
                          usecols=channel)
        eeg = data[:, :-1]
        timestamps = data[:, -1]
        prev = 0
        prev_direction = df['Direction'][prev]
        data = {'Right': [], 'Left': [], 'Rest': []}
        '''
        # This was to see if we can normalize the EEG (didn't work)
        start = df['Time'][0]
        end = df['Time'].iloc[-1]
        indices = np.where(np.logical_and(timestamps >= start, timestamps <= end))
        start, end = indices[0][0] - tmin, indices[0][-1]
        eeg = scale(eeg[start:end+1])
        timestamps = timestamps[start:end+1]
        '''
        
        for idx, el in enumerate(df['Direction']):
            if el != prev_direction or idx == len(df.index) - 1:
                start = df['Time'][prev]
                end = df['Time'][idx]
                indices = np.where(np.logical_and(timestamps >= start, timestamps <= end))
                start, end = indices[0][0] - tmin, indices[0][-1]
                trial = eeg[start:end]
                all_data[prev_direction].append(trial)
                #print(idx - prev, prev_direction)
                # print(len(trial))
                prev = idx
                prev_direction = el
        all_data = merge_dols(all_data, data)
    return all_data
csvs = [
        ["data/March22_008/10_008-2019-3-22-15-8-55.csv", #008
        "data/March22_008/9_008-2019-3-22-14-59-0.csv",
        "data/March22_008/8_008-2019-3-22-14-45-53.csv",
        ],
        #["data/March17/4_RestLeftRight_MI_5s.csv", #005
        #"data/March17/5_RestLeftRight_MI_10s.csv",
        #],
        ["data/March22_001/1-001-rest20s_left10s_right10s_MI-2019-3-22-16-0-32.csv", #001
        "data/March22_001/2-001-rest20s_left10s_right10s_MI-2019-3-22-16-12-17.csv",
        "data/March22_001/3-001-rest20s_left15s_right15s_MI-2019-3-22-16-19-25.csv",
        "data/March22_001/4-001-rest25s_left15s_right15s_MI-2019-3-22-16-27-44.csv",
        "data/March22_001/5-001-rest25s_left10s_right10s_MI-2019-3-22-16-35-57.csv",
        "data/March22_001/7-001-rest25s_left20s_right20s_MI-2019-3-22-16-54-17.csv",
        ],
        ["data/March20/time-test-JingMingImagined_10s-2019-3-20-10-28-35.csv",  #009
        "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-30-26.csv",
        "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-35-31.csv",
        "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-57-45.csv",
        "data/March20/time-test-JingMingImaginedREALLYGOOD-2019-3-20-10-21-44.csv",
        "data/March20/time-test-JingMingImagined10s-2019-3-20-10-12-1.csv",
        ],
        ["data/March24_011/1_011_Rest20LeftRight20_MI-2019-3-24-16-25-41.csv",  #011
        "data/March24_011/2_011_Rest20LeftRight20_MI-2019-3-24-16-38-10.csv",
        "data/March24_011/3_011_Rest20LeftRight10_MI-2019-3-24-16-49-23.csv",
        "data/March24_011/4_011_Rest20LeftRight10_MI-2019-3-24-16-57-8.csv",
        "data/March24_011/5_011_Rest20LeftRight20_MI-2019-3-24-17-3-17.csv",
        ],
        [
        "data/March29_014/1_014_rest_left_right_20s-2019-3-29-16-44-32.csv",
        "data/March29_014/2_014_rest_left_right_20s-2019-3-29-16-54-36.csv",
        "data/March29_014/3_014_AWESOME_rest_left_right_20s-2019-3-29-16-54-36.csv",
        "data/March29_014/4_014_final_run-2019-3-29-17-38-45.csv",
        ],
        ["data/March28_027/1_027_20srest_10sleft_10sright-2019-3-28-16-57-21.csv", #027
         "data/March28_027/2_027_20srest_10sleft_10sright-2019-3-28-17-27-46.csv",
         "data/March28_027/3_027_20srest_10sleft_10sright-2019-3-28-17-31-43.csv",
         "data/March28_027/4_027_20srest_10sleft_10sright-2019-3-28-17-50-46.csv",
         "data/March28_027/6_027_20srest_10sleft_10sright-2019-3-28-18-0-18.csv",
         "data/March28_027/7_027_20srest_10sleft_10sright-2019-3-28-18-14-43.csv",
         "data/March28_027/8_027_20srest_10sleft_10sright-2019-3-28-18-20-21.csv",
         "data/March28_027/9_027_20srest_10sleft_10sright-2019-3-28-18-26-5.csv",
        ],
        ["data/March28_045/1_045-2019-3-28-20-46-25.csv", #045
         "data/March28_045/2_045_25rest_20left_20right_fourtimes_ML-2019-3-28-21-0-56.csv",
         "data/March28_045/3_045_25rest_20left_20right_fourtimes_ML-2019-3-28-21-6-55.csv",
         "data/March28_045/4_045_25rest_20left_20right_fourtimes_ML-2019-3-28-21-14-4.csv",
        ],
        ]

all_csvs = [name for sublist in csvs for name in sublist]

csv_map = { "4_RestLeftRight_MI_5s.csv": "4_RestLeftRight_5s_MI_OpenBCI-RAW-2019-03-17_16-32-53.txt", #005
           "5_RestLeftRight_MI_10s.csv": "5_RestLeftRight_10s_OpenBCI-RAW-2019-03-17_16-37-32.txt",
           "10_008-2019-3-22-15-8-55.csv": "10_008_OpenBCI-RAW-2019-03-22_15-07-58.txt",
           "9_008-2019-3-22-14-59-0.csv": "8to9_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",
           "8_008-2019-3-22-14-45-53.csv": "8to9_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",
           
           "5-001-rest25s_left10s_right10s_MI-2019-3-22-16-35-57.csv": "1to5_001_OpenBCI-RAW-2019-03-22_15-56-26.txt",
           "4-001-rest25s_left15s_right15s_MI-2019-3-22-16-27-44.csv": "1to5_001_OpenBCI-RAW-2019-03-22_15-56-26.txt",
           "7-001-rest25s_left20s_right20s_MI-2019-3-22-16-54-17.csv": "6to7_001_OpenBCI-RAW-2019-03-22_16-44-46.txt",
           "1-001-rest20s_left10s_right10s_MI-2019-3-22-16-0-32.csv": "1to5_001_OpenBCI-RAW-2019-03-22_15-56-26.txt",
           "2-001-rest20s_left10s_right10s_MI-2019-3-22-16-12-17.csv": "1to5_001_OpenBCI-RAW-2019-03-22_15-56-26.txt",
           "3-001-rest20s_left15s_right15s_MI-2019-3-22-16-19-25.csv": "1to5_001_OpenBCI-RAW-2019-03-22_15-56-26.txt",
           
           "time-test-JingMingImagined_10s-2019-3-20-10-28-35.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
           "time-test-JingMingImagined_10s-2019-3-20-10-30-26.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
           "time-test-JingMingImagined_10s-2019-3-20-10-35-31.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
           "time-test-JingMingImagined_10s-2019-3-20-10-57-45.csv": 'OpenBCI-RAW-2019-03-20_10-56-20.txt',
           "time-test-JingMingImaginedREALLYGOOD-2019-3-20-10-21-44.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
           "time-test-JingMingImagined10s-2019-3-20-10-12-1.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
           
           "1_011_Rest20LeftRight20_MI-2019-3-24-16-25-41.csv": '011_1to3_OpenBCI-RAW-2019-03-24_16-21-59.txt',
           "2_011_Rest20LeftRight20_MI-2019-3-24-16-38-10.csv": '011_1to3_OpenBCI-RAW-2019-03-24_16-21-59.txt',
           "3_011_Rest20LeftRight10_MI-2019-3-24-16-49-23.csv": '011_1to3_OpenBCI-RAW-2019-03-24_16-21-59.txt',
           "4_011_Rest20LeftRight10_MI-2019-3-24-16-57-8.csv": '011_4to6_OpenBCI-RAW-2019-03-24_16-54-15.txt',
           "5_011_Rest20LeftRight20_MI-2019-3-24-17-3-17.csv": '011_4to6_OpenBCI-RAW-2019-03-24_16-54-15.txt',
           
           "1_014_rest_left_right_20s-2019-3-29-16-44-32.csv": "1_014_OpenBCI-RAW-2019-03-29_16-40-55.txt",
           "2_014_rest_left_right_20s-2019-3-29-16-54-36.csv": "2_014_OpenBCI-RAW-2019-03-29_16-52-46.txt",
           "3_014_AWESOME_rest_left_right_20s-2019-3-29-16-54-36.csv": "3_014_AWESOME_OpenBCI-RAW-2019-03-29_17-08-21.txt",
           "4_014_final_run-2019-3-29-17-38-45.csv": "4_014_OpenBCI-RAW-2019-03-29_17-28-26.txt",
           
           "1_027_20srest_10sleft_10sright-2019-3-28-16-57-21.csv": "027_OpenBCI-RAW-2019-03-28_15-55-27.txt", 
           "2_027_20srest_10sleft_10sright-2019-3-28-17-27-46.csv": "027_OpenBCI-RAW-2019-03-28_17-26-57.txt",
           "3_027_20srest_10sleft_10sright-2019-3-28-17-31-43.csv": "027_OpenBCI-RAW-2019-03-28_17-26-57.txt",
           "4_027_20srest_10sleft_10sright-2019-3-28-17-50-46.csv": "027_OpenBCI-RAW-2019-03-28_17-49-29.txt",
           "6_027_20srest_10sleft_10sright-2019-3-28-18-0-18.csv": "027_OpenBCI-RAW-2019-03-28_17-53-52.txt",
           "7_027_20srest_10sleft_10sright-2019-3-28-18-14-43.csv": "027_OpenBCI-RAW-2019-03-28_18-12-08.txt",
           "8_027_20srest_10sleft_10sright-2019-3-28-18-20-21.csv": "027_OpenBCI-RAW-2019-03-28_18-12-08.txt",
           "9_027_20srest_10sleft_10sright-2019-3-28-18-26-5.csv": "027_OpenBCI-RAW-2019-03-28_18-12-08.txt",
           
           "1_045-2019-3-28-20-46-25.csv": "1_045OpenBCI-RAW-2019-03-28_19-43-36.txt", 
           "2_045_25rest_20left_20right_fourtimes_ML-2019-3-28-21-0-56.csv": "2to4_045_OpenBCI-RAW-2019-03-28_20-55-19.txt",
           "3_045_25rest_20left_20right_fourtimes_ML-2019-3-28-21-6-55.csv": "2to4_045_OpenBCI-RAW-2019-03-28_20-55-19.txt",
           "4_045_25rest_20left_20right_fourtimes_ML-2019-3-28-21-14-4.csv": "2to4_045_OpenBCI-RAW-2019-03-28_20-55-19.txt",
           }

data_dict = {}
for csv in all_csvs:
    data_dict[csv] = get_data([csv])