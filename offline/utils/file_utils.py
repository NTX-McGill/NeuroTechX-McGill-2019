#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  5 18:30:48 2019

@author: marley
"""
import pandas as pd
import numpy as np

from metadata import MARKER_DATA, DATA_COLUMNS, LABELS, ALL_FILES

def merge_dols(dol1, dol2):
    keys = set(dol1).union(dol2)
    no = []
    return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)

def load_openbci_raw(path):
    data = np.loadtxt(path,
                      delimiter=',',
                      skiprows=7,
                      usecols=DATA_COLUMNS)
    eeg = data[:, :-1]
    timestamps = data[:, -1]
    return eeg, timestamps


def load_data(csv):
    print("loading " + csv)
    data = {label: [] for label in LABELS} 
    df = pd.read_csv('../' + csv)
    path_arr = csv.split('/')
    folder, fname = path_arr[:-1], path_arr[-1]
    data_path = "/".join(['..'] + folder + [MARKER_DATA[fname]])
    eeg, timestamps = load_openbci_raw(data_path)
    prev = 0
    prev_direction = df['Direction'][prev]
    for idx, el in enumerate(df['Direction']):
        if el != prev_direction or idx == len(df.index) - 1:
            start = df['Time'][prev]
            end = df['Time'][idx]
            indices = np.where(
                np.logical_and(
                    timestamps >= start,
                    timestamps <= end))
            trial = eeg[indices]
            data[prev_direction].append(trial)
            prev = idx
            prev_direction = el
    return data

def load_dataset(csv_set):
    dataset = {label: [] for label in LABELS} 
    for csv in csv_set:
        data = load_data(csv)
        dataset = merge_dols(dataset, data)
    return dataset

def load_all():
    return {fname: load_data(fname) for fname in ALL_FILES}