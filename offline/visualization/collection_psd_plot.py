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

import sys
sys.path.append('../utils')
from metadata import SAMPLING_FREQ, ELECTRODE_C3, ELECTRODE_C4
import file_utils

csv_name = "../data/March22_008/8_008-2019-3-22-14-45-53.csv"
train_data = file_utils.load_data(csv_name)
idx = 1
for direction, data in train_data.items():
    l = np.hstack([trial[:,ELECTRODE_C3] for trial in data])
    r = np.hstack([trial[:,ELECTRODE_C4] for trial in data])
    psd1, freqs = mlab.psd(np.squeeze(l),
                              NFFT=2048,
                              noverlap=250,
                              Fs=SAMPLING_FREQ)
    psd2, freqs = mlab.psd(np.squeeze(r),
                              NFFT=2048,
                              noverlap=250,
                              Fs=SAMPLING_FREQ)
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