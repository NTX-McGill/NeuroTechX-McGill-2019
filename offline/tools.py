'''
@author: Marley
This file contains a bunch of stuff ready to import 
'''
import numpy as np
import numpy.fft as fft
from scipy import signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sn
import pandas as pd
import warnings
warnings.filterwarnings("ignore")   


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    #print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax
def filter_(arr, fs_Hz, lowcut, highcut, order):
   nyq = 0.5 * fs_Hz
   b, a = signal.butter(1, [lowcut/nyq, highcut/nyq], btype='band')
   for i in range(0, order):
       arr = signal.lfilter(b, a, arr, axis=0)
   return arr

def get_start_indices(ch):
    start_indices = [0]
    i = 0
    while i < len(ch):
        if ch[i] > 100:
            start_indices.append(i)
            i += 500
        i += 1
    return start_indices
def get_psd(ch, fs_Hz, shift=0.1):
    NFFT = fs_Hz*2
    overlap = NFFT - int(shift * fs_Hz)
    psd,freqs = mlab.psd(np.squeeze(ch),
                                   NFFT=NFFT,
                                   window=mlab.window_hanning,
                                   Fs=fs_Hz,
                                   noverlap=overlap
                                   ) # returns PSD power per Hz
    # convert the units of the spectral data
    return psd,freqs # dB re: 1 uV

def get_spectral_content(ch, fs_Hz, shift=0.1):
    NFFT = fs_Hz*2
    #overlap = NFFT - int(shift * fs_Hz)
    spec_PSDperHz, spec_freqs, spec_t = mlab.specgram(np.squeeze(ch),
                                   NFFT=NFFT,
                                   window=mlab.window_hanning,
                                   Fs=fs_Hz,
                                   #noverlap=overlap
                                   ) # returns PSD power per Hz
    # convert the units of the spectral data
    spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)
    return spec_t, spec_freqs, spec_PSDperBin  # dB re: 1 uV

def plot_specgram(spec_freqs, spec_PSDperBin,title,shift,i=1):
    f_lim_Hz = [0, 20]   # frequency limits for plotting
    #plt.figure(figsize=(10,5))
    spec_t = [idx*.1 for idx in range(len(spec_PSDperBin[0]))]
    plt.subplot(3,1,i)
    plt.title(title)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25,26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.subplots_adjust(hspace=1)

def resize_min(specgram, i=1):
    min_length = min([el.shape[0] for el in specgram])
    specgram = np.array([el[:min_length] for el in specgram])
    return specgram
def resize_max(specgram, fillval=np.nan):
    max_length = max([len(el[0]) for el in specgram])
    return np.array([pad_block(el, max_length,fillval) for el in specgram])
def pad_block(block, max_length, fillval):
    padding = np.full([len(block), max_length-(len(block[0]))], fillval)
    return np.hstack((block,padding))
def epoch_data(data, window_length, shift):
    arr = []
    i = 0
    while i + window_length < len(data):
        arr.append(data[i:i+window_length])
        i += shift
    return np.array(arr)
def extract(all_data, window_s, shift, plot_psd=False, keep_trials=False):
    all_psds = {'Right': [], 'Left': [], 'Rest': []}
    all_features = {'Right': [], 'Left': [], 'Rest': []}
    
    idx = 1
    fig1 = plt.figure("psd")
    fig1.clf()
    for direction, data in all_data.items():
        for trial in data:
            if direction == 'Rest':
                trial = trial[int(len(trial)/2):]
            epochs = epoch_data(trial, 250 * window_s, int(shift*250))    # shape n x 500 x 2
            trial_features = []
            for epoch in epochs:
                features, freqs, psd1, psd2 = get_features(epoch.T)
                all_psds[direction].append([psd1,psd2])
                trial_features.append(features)
                if plot_psd:
                    plt.subplot(3,2,idx)
                    plt.plot(freqs, psd1)
                    plt.ylim([0,25])
                    plt.xlim([6,20])
                    plt.subplot(3,2,idx+1)
                    plt.plot(freqs, psd2)
                    plt.ylim([0,25])
                    plt.xlim([6,20])
            if keep_trials:
                all_features[direction].append(np.array(trial_features))
            else:
                all_features[direction].extend(trial_features)
        idx += 2
    return all_psds, all_features, freqs
def to_feature_vec(all_features, rest=False):
    classes = ['Left', 'Right', 'Rest']
    feature_arr = []
    for direction, features in all_features.items():
        features = np.array(features)
        arr = np.hstack((features, np.full([features.shape[0],1], classes.index(direction))))
        feature_arr.append(arr)
    if not rest:
        feature_arr = feature_arr[:-1]
    return np.vstack(feature_arr)
def merge_all_dols(arr):
    all_data = {'Right': [], 'Left': [], 'Rest': []}
    for dol in arr:
        all_data = merge_dols(all_data, dol)
    return all_data
def merge_dols(dol1, dol2):
    keys = set(dol1).union(dol2)
    no = []
    return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)
def get_data(csvs, csv_map, tmin=0):
    all_data = {'Right': [], 'Left': [], 'Rest': []}
    for csv in csvs:
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
                start, end = indices[0][0] - tmin, indices[0][-1]
                trial = eeg[start:end]
                all_data[prev_direction].append(trial)
                #print(idx - prev, prev_direction)
                #print(len(trial))
                prev = idx
                prev_direction = el
        all_data = merge_dols(all_data, data)
    return all_data

def get_features(arr):
    # ch has shape (2, 500)
    channels=[0,1,6,7]
    #channels=[0,7]
    
    psds_per_channel = []
    for ch in arr[channels]:
        psd, freqs = mlab.psd(np.squeeze(ch),
                              NFFT=500,
                              Fs=250)
        psds_per_channel.append(psd)
    psds_per_channel = np.array(psds_per_channel)
    mu_indices = np.where(np.logical_and(freqs>=10, freqs<=12))
    
    #features = np.amax(psds_per_channel[:,mu_indices], axis=-1).flatten()   # max of 10-12hz as feature
    features = np.mean(psds_per_channel[:,mu_indices], axis=-1).flatten()   # mean of 10-12hz as feature
    features = np.array([features[:2].mean(), features[2:].mean()])
    #features = psds_per_channel[:,mu_indices].flatten()                     # all of 10-12hz as feature
    return features, freqs, psds_per_channel[0], psds_per_channel[-1]