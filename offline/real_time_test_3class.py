#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 09:03:30 2019

@author: marley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  4 18:25:05 2019

@author: marley
"""
import numpy as np
import numpy.fft as fft
from scipy import signal
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
import seaborn as sn
import pandas as pd
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.utils import shuffle


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
    min_length = min([len(el[0]) for el in specgram])
    specgram = np.array([el[:, :min_length] for el in specgram])
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
def extract(all_data, window_s, shift, plot_psd=False):
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
            for epoch in epochs:
                features, freqs, psd1, psd2 = get_features(epoch.T)
                all_psds[direction].append([psd1,psd2])
                all_features[direction].append(features)
                if plot_psd:
                    plt.subplot(3,2,idx)
                    plt.plot(freqs, psd1)
                    plt.ylim([0,25])
                    plt.xlim([6,20])
                    plt.subplot(3,2,idx+1)
                    plt.plot(freqs, psd2)
                    plt.ylim([0,25])
                    plt.xlim([6,20])
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
def get_data(csvs):
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
                trial = eeg[indices]
                all_data[prev_direction].append(trial)
                #print(idx - prev, prev_direction)
                #print(len(trial))
                prev = idx
                prev_direction = el
        all_data = merge_dols(all_data, data)
    return all_data

csvs = ["data/March22_008/10_008-2019-3-22-15-8-55.csv",
        "data/March22_008/9_008-2019-3-22-14-59-0.csv",
        "data/March22_008/8_008-2019-3-22-14-45-53.csv",
        "data/March22_008/7_008-2019-3-22-14-27-46.csv",    #actual
        "data/March22_008/6_008-2019-3-22-14-19-52.csv",    #actual
        "data/March22_008/5_008-2019-3-22-14-10-26.csv",    #actual
        "data/March22_001/4-001-rest25s_left15s_right15s_MI-2019-3-22-16-27-44.csv",
        "data/March22_001/5-001-rest25s_left10s_right10s_MI-2019-3-22-16-35-57.csv",
        "data/March22_001/6-001-rest25s_left15s_right15s_MI-2019-3-22-16-46-17.csv",    #actual
        "data/March22_001/7-001-rest25s_left20s_right20s_MI-2019-3-22-16-54-17.csv",
        "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-28-35.csv",   #10
        "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-30-26.csv",
        "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-35-31.csv",
        "data/March24_011/1_011_Rest20LeftRight20_MI-2019-3-24-16-25-41.csv",
        "data/March24_011/2_011_Rest20LeftRight20_MI-2019-3-24-16-38-10.csv",
        "data/March24_011/3_011_Rest20LeftRight10_MI-2019-3-24-16-49-23.csv",
        "data/March24_011/4_011_Rest20LeftRight10_MI-2019-3-24-16-57-8.csv",
        "data/March24_011/5_011_Rest20LeftRight20_MI-2019-3-24-17-3-17.csv",
        ]



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
fs_Hz = 250
sampling_freq = 250
window_s = 2
shift = 0.5
channel_name = 'C4'
Viet = 0
Marley = 0
Andy = 0
cm = 0
plot_psd = 0            # set this to 1 if you want to plot the psds per window
colormap = sn.cubehelix_palette(as_cmap=True)
tmin, tmax = 0,0

# * set load_data to true the first time you run the script
load_data = 0
if load_data:
    data_dict = {}
    for csv in csvs:
        data_dict[csv] = get_data([csv])
""" * modify this to test filtering and new features """    
def get_features(arr):
    # ch has shape (2, 500)
    channels=[0,1,6,7]
    channels=[0,7]
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
    #features = psds_per_channel[:,mu_indices].flatten()                     # all of 10-12hz as feature
    return features, freqs, psds_per_channel[0], psds_per_channel[-1]
""" end """

# * use this to select which files you want to test/train on
train_csvs = [1]          # index of the training files we want to use
test_csvs = [2]             # index of the test files we want to use
train_csvs = [csvs[i] for i in train_csvs]
test_csvs = [csvs[i] for i in test_csvs]
train_data = merge_all_dols([data_dict[csv] for csv in train_csvs])
for window_s in [2]:
    train_psds, train_features, freqs = extract(train_data, window_s, shift, plot_psd)
    data = to_feature_vec(train_features)
    
    X = data[:,:-1]
    Y = data[:,-1]
    validation_size = 0.20
    seed = 7
    #X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    
    # Test options and evaluation metric
    scoring = 'accuracy'
    
    # Spot Check Algorithms
    models = []
    models.append(('LR', LogisticRegression(solver='lbfgs')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB(var_smoothing=0.001)))
    models.append(('SVM', SVC(gamma='scale')))
    # evaluate each model in turn
    results = []
    names = []
    
    X, Y = shuffle(X, Y, random_state=seed)
    print("VALIDATION")
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)
    print("average accuracy: " + "{:2.1f}".format(np.array(results).mean() * 100))
    
    print("TEST")
    test_dict = data_dict[test_csvs[0]]
    _, test_features, _ = extract(test_dict, window_s, shift, plot_psd)
    test_data = to_feature_vec(test_features)
    X_test = test_data[:,:-1]
    Y_test = test_data[:,-1]
    test_results = []
    for name, model in models:
        model.fit(X, Y)
        score = model.score(X_test, Y_test)
        msg = "%s: %f" % (name, score)
        print(msg)
        test_results.append(score)
    print("test accuracy:")
    print("{:2.1f}".format(np.array(test_results).mean() * 100))

    

# Plots
mu_indices = np.where(np.logical_and(freqs>=14, freqs<=20))
fig3 = plt.figure("scatter")
fig3.clf()
log = 0
for direction, psd in train_psds.items():
    psd = np.array(psd).T
    mu = np.mean(psd[mu_indices],axis=0)
    #mu = np.amax(psd[mu_indices], axis=0)
    if log:
        mu = np.log10(mu)
    plt.scatter(mu[0], mu[1], s=2)
plt.axis('scaled')
plt.show()


mean_plt = 1
if mean_plt:
    fig4 = plt.figure('mean')
    fig4.clf()
    for direction, psd in train_psds.items():
        psd = np.array(psd).T
        mu = np.mean(psd[mu_indices],axis=0)
        if log:
            mu = np.log10(mu)
        plt.scatter(np.mean(mu[0]), np.mean(mu[1]), s=2)
    plt.axis('scaled')

plt.figure()   
ax = plt.subplot(121)
plt.title("Mean")
for direction, psd in train_psds.items():
    psd = np.array(psd).T
    mu = np.mean(psd[mu_indices],axis=0)
    #mu = np.amax(psd[mu_indices], axis=0)
    if log:
        mu = np.log10(mu)
    if direction != 'Rest':
        sn.kdeplot(mu[0], mu[1], ax=ax, shade_lowest=False, alpha=0.6)
ax.set(aspect="equal")
ymin, ymax = plt.gca().get_ylim()

ax = plt.subplot(122, sharex=ax, sharey=ax)
plt.title("Max")
for direction, psd in train_psds.items():
    psd = np.array(psd).T
    #mu = np.mean(psd[mu_indices],axis=0)
    mu = np.amax(psd[mu_indices], axis=0)
    if log:
        mu = np.log10(mu)
    if direction != 'Rest':
        sn.kdeplot(mu[0], mu[1], ax=ax, shade_lowest=False, alpha=0.6)
