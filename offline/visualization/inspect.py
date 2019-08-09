#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 19:13:51 2019
@author: marley
"""
import preprocessing as prepro
import file_utils
from metadata import SAMPLING_FREQ, ELECTRODE_C3, ELECTRODE_C4, ELECTRODE_C1, ELECTRODE_C2
import glob
import numpy as np
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import pandas as pd
from scipy import signal

import sys
sys.path.append('../utils')


def draw_spectral_content(spec_freqs, spec_PSDperBin, title=""):
    spec_t = [idx * .1 for idx in range(len(spec_PSDperBin[0]))]
    f_lim_Hz = [0, 20]   # frequency limits for plotting
    if title:
        plt.title(title)
    plt.pcolormesh(spec_t, spec_freqs, 10 *
                   np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25, 25])
    plt.xlim(spec_t[0], spec_t[-1] + 1)
    plt.ylim(f_lim_Hz)
    # plt.xlabel('Time (sec)') aint got space for this
    plt.ylabel('Frequency (Hz)')
    if title:
        plt.subplots_adjust(hspace=0.6)


def plot_specgram(fig=None, title=""):
    spec_t, spec_freqs, spec_PSDperBin = prepro.get_spectral_content()
    draw_spectral_content(spec_freqs, spec_PSDperBin, title=title)


shift = 0.1
folder = "data/March22_008/"
csv = "8_008-2019-3-22-14-45-53.csv"
all_data = file_utils.load_data(folder + csv)

hemisphere_left = [ELECTRODE_C3, ELECTRODE_C1]
hemisphere_right = [ELECTRODE_C4, ELECTRODE_C2]
num = 1
for direction, data in all_data.items():
    figname = direction + '_spec_' + csv.split('.')[0]
    fig = plt.figure(figname)
    left = []
    right = []
    for block in data:
        all_spectra = []
        filtered_block = prepro.filter_signal(block, 1, 40, 1)
        for channel in filtered_block.T:
            t, f, spec = prepro.get_spectral_content(channel, SAMPLING_FREQ)
            all_spectra.append(spec)
        all_spectra = np.array(all_spectra)
        left.append(np.mean(all_spectra[hemisphere_left], axis=0))
        right.append(np.mean(all_spectra[hemisphere_right], axis=0))
    idx = 0
    num_rows = max(len(left), 5)
    for spec in left:
        idx += 1
        plt.subplot(num_rows, 2, idx * 2 - 1)
        draw_spectral_content(f, spec, fig)
    idx = 0
    for spec in right:
        idx += 1
        plt.subplot(num_rows, 2, idx * 2)
        draw_spectral_content(f, spec, fig)
    plt.savefig(figname)

    fig = plt.figure('av_spec_' + csv.split('.')[0])
    plt.subplot(3, 2, num)
    draw_spectral_content(f, np.mean(prepro.resize_max(left), axis=0),
                          title=direction + ", channel 1-4")
    plt.subplot(3, 2, num + 1)
    draw_spectral_content(f, np.mean(prepro.resize_max(right), axis=0),
                          title=direction + ", channel 5-8")
    num += 2

    figname = direction + '_mu_' + csv.split('.')[0]
    fig = plt.figure(figname)
    idx = 0
    mu_indices = np.where(np.logical_and(f >= 10, f <= 12))
    for spec in left:
        idx += 1
        plt.subplot(num_rows, 1, idx)
        mu = np.mean(spec[mu_indices], axis=0)
        plt.plot(mu, label="0 set")
    idx = 0
    for spec in right:
        idx += 1
        plt.subplot(num_rows, 1, idx)
        mu = np.mean(spec[mu_indices], axis=0)
        plt.plot(mu, label="8 set")
    plt.legend()
