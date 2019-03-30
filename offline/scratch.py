#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 30 13:03:39 2019

@author: orange
"""
import numpy as np
import pandas as pd

csvs = ["data/March22_008/10_008-2019-3-22-15-8-55.csv",
        "data/March22_008/9_008-2019-3-22-14-59-0.csv",
        "data/March22_008/8_008-2019-3-22-14-45-53.csv",    # 
            #"data/March22_008/7_008-2019-3-22-14-27-46.csv",    #actual
            #"data/March22_008/6_008-2019-3-22-14-19-52.csv",    #actual
            #"data/March22_008/5_008-2019-3-22-14-10-26.csv",    #actual
            "data/March22_001/4-001-rest25s_left15s_right15s_MI-2019-3-22-16-27-44.csv",
            "data/March22_001/5-001-rest25s_left10s_right10s_MI-2019-3-22-16-35-57.csv",
            #"data/March22_001/6-001-rest25s_left15s_right15s_MI-2019-3-22-16-46-17.csv",    #actual
            "data/March22_001/7-001-rest25s_left20s_right20s_MI-2019-3-22-16-54-17.csv",
            "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-28-35.csv",   #6
            "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-30-26.csv",
            "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-35-31.csv",
            "data/March24_011/1_011_Rest20LeftRight20_MI-2019-3-24-16-25-41.csv",   #9 to 13
            "data/March24_011/2_011_Rest20LeftRight20_MI-2019-3-24-16-38-10.csv",
            "data/March24_011/3_011_Rest20LeftRight10_MI-2019-3-24-16-49-23.csv",
            "data/March24_011/4_011_Rest20LeftRight10_MI-2019-3-24-16-57-8.csv",
            "data/March24_011/5_011_Rest20LeftRight20_MI-2019-3-24-17-3-17.csv",
            "data/March29_014/1_014_rest_left_right_20s-2019-3-29-16-44-32.csv",   # 14
            "data/March29_014/2_014_rest_left_right_20s-2019-3-29-16-54-36.csv",
            "data/March29_014/3_014_AWESOME_rest_left_right_20s-2019-3-29-16-54-36.csv",
            "data/March29_014/4_014_final_run-2019-3-29-17-38-45.csv",
            #"data/March29_014/5_014_eye_blink-2019-3-29-17-44-33.csv",
            #"data/March29_014/6_014_eye_blink-2019-3-29-17-46-14.csv",
            #"data/March29_014/7_014_eye_blink-2019-3-29-17-47-56.csv",
            ]



