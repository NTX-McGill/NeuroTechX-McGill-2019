# -*- coding: utf-8 -*-
"""
Created on Fri Apr 19 17:25:23 2019

@author: Danielle
"""

csvs = [
        ["data/March17/4_RestLeftRight_MI_5s.csv", #005
        "data/March17/5_RestLeftRight_MI_10s.csv",
        ],
        ["data/March22_008/10_008-2019-3-22-15-8-55.csv", #008
        "data/March22_008/9_008-2019-3-22-14-59-0.csv",
        "data/March22_008/8_008-2019-3-22-14-45-53.csv",
        ],
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
         #missing files for subjects 027 and 045 bc not on github
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
           
           "1_014-2019-3-27-17-35-31.csv": "OpenBCI-RAW-2019-03-27_17-34-41.txt", 
           "2_014-2019-3-27-17-38-49.csv": "OpenBCI-RAW-2019-03-27_17-46-27.txt",
           "3_014_10s_10s_L_R-2019-3-27-18-1-11.csv": "3_014_OpenBCI-RAW-2019-03-27_17-53-15.txt",
           "4_014_10s_10s_L_R-2019-3-27-18-21-11.csv": "4_014_OpenBCI-RAW-2019-03-27_18-03-56.txt",
           "1_014_rest_left_right_20s-2019-3-29-16-44-32.csv": "1_014_OpenBCI-RAW-2019-03-29_16-40-55.txt",
           "2_014_rest_left_right_20s-2019-3-29-16-54-36.csv": "2_014_OpenBCI-RAW-2019-03-29_16-52-46.txt",
           "3_014_AWESOME_rest_left_right_20s-2019-3-29-16-54-36.csv": "3_014_AWESOME_OpenBCI-RAW-2019-03-29_17-08-21.txt",
           "4_014_final_run-2019-3-29-17-38-45.csv": "4_014_OpenBCI-RAW-2019-03-29_17-28-26.txt",
           }

data_dict = {}
for csv in all_csvs:
    data_dict[csv] = get_data([csv])