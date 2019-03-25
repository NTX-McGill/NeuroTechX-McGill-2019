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

    print(cm)

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
def merge_dols(dol1, dol2):
    keys = set(dol1).union(dol2)
    no = []
    return dict((k, dol1.get(k, no) + dol2.get(k, no)) for k in keys)
def get_data(csvs):
    all_data = {'Right': [], 'Left': [], 'Rest': []}
    for csv in csvs:
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
""" BASELINE PREDICTION ALGORITHM FOR MVP """
def predict(ch, threshold):
    # ch has shape (2, 500)
    
    psd1,freqs = mlab.psd(np.squeeze(ch[0]),
                           NFFT=500,
                           Fs=250)
    mu_indices = np.where(np.logical_and(freqs>=10, freqs<=12))
    mu1 = np.mean(psd1[mu_indices])
    
    psd2,freqs = mlab.psd(np.squeeze(ch[7]),
                           NFFT=500,
                           Fs=250)
    mu2 = np.mean(psd2[mu_indices])
    
    #return int(mu1 < threshold), int(mu2 < threshold), freqs, psd1, psd2     # return 1,0 for left, 0,1 for right, 1,1 for both and 0,0 for rest
    #return mu1/np.mean(psd1[8:40]), mu2/np.mean(psd2[8:40]), freqs, psd1, psd2
    return mu1, mu2, freqs, psd1, psd2
""" END """

csv_map = {"10_008-2019-3-22-15-8-55.csv": "10_008_OpenBCI-RAW-2019-03-22_15-07-58.txt",
           "9_008-2019-3-22-14-59-0.csv": "8to9_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",
           "8_008-2019-3-22-14-45-53.csv": "8to9_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",
           "7_008-2019-3-22-14-27-46.csv": "4to7_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",
           "6_008-2019-3-22-14-19-52.csv": "4to7_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",
           "5_008-2019-3-22-14-10-26.csv": "4to7_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",
           "5-001-rest25s_left10s_right10s_MI-2019-3-22-16-35-57.csv": "1to5_001_OpenBCI-RAW-2019-03-22_15-56-26.txt",
           "4-001-rest25s_left15s_right15s_MI-2019-3-22-16-27-44.csv": "1to5_001_OpenBCI-RAW-2019-03-22_15-56-26.txt",
           "6-001-rest25s_left15s_right15s_MI-2019-3-22-16-46-17.csv": "6to7_001_OpenBCI-RAW-2019-03-22_16-44-46.txt",
           "7-001-rest25s_left20s_right20s_MI-2019-3-22-16-54-17.csv": "6to7_001_OpenBCI-RAW-2019-03-22_16-44-46.txt",
           "time-test-JingMingImagined_10s-2019-3-20-10-28-35.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
           "time-test-JingMingImagined_10s-2019-3-20-10-30-26.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
           "time-test-JingMingImagined_10s-2019-3-20-10-35-31.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt'}
fs_Hz = 250
sampling_freq = 250
window_s = 2
shift = 0.1
channel = (1,2)
channel_name = 'C4'
Viet = 0
Marley = 0
Andy = 0
cm = 0
plot_psd = 0
colormap = sn.cubehelix_palette(as_cmap=True)
tmin, tmax = 0,0

left_data = []
rest_data = []

if Viet:
    fname = 'data/March 4/5_SUCCESS_Rest_RightAndJawClench_10secs.txt' 
    #fname = 'data/March 4/6_SUCCESS_Rest_RightClench_JawClench_ImagineClench_10secs.txt' 
    fname = 'data/March 4/7_SUCCESS_Rest_RightClenchImagineJaw_10secs.txt'
    data = np.loadtxt(fname,
                      delimiter=',',
                      skiprows=7,
                      usecols=channel)
    data_ = filter_(data, sampling_freq, 1, 40, 1)
    ch = data_.T[0]     
    start_indices = get_start_indices(ch)
    for i in range(len(start_indices) - 1):
        start = int(max(start_indices[i] + tmin * sampling_freq, 0))
        end = int(min(start_indices[i+1] + tmax * sampling_freq, start_indices[-1]))
        if i % 2:
            left_data.append(data[start:end])
        else:
            rest_data.append(data[start:end])
if Marley:
    fname = 'data/March11_Marley/Marley_prolonged_trial.txt'
    data = np.loadtxt(fname,
                      delimiter=',',
                      skiprows=7,
                      usecols=channel)
    data = filter_(data, sampling_freq, 1, 40, 1)
    rest_indices = [i * 10 * fs_Hz for i in range(7)]
    left_indices = [(i+6) * 10 * fs_Hz for i in range(7)]
    start_indices = rest_indices
    for i in range(len(start_indices) - 1):
        start = int(max(start_indices[i] + tmin * sampling_freq, 0))
        end = int(min(start_indices[i+1] + tmax * sampling_freq, start_indices[-1]))
        rest_data.append(data[start:end])
    start_indices = left_indices
    for i in range(len(start_indices) - 1):
        start = int(max(start_indices[i] + tmin * sampling_freq, 0))
        end = int(min(start_indices[i+1] + tmax * sampling_freq, start_indices[-1]))
        left_data.append(data[start:end])
if Andy:
    fname = 'OpenBCI-RAW-2019-03-18_18-46-51.txt'
    markers = 'time-stamp-67-2019-2-18-18-47-12.csv'
    fname = 'data/March18/OpenBCI-RAW-2019-03-18_19-35-36.txt'
    markers = 'data/March18/time-stamp-68-2019-2-18-19-36-0.csv'
    df = pd.read_csv(markers)
    start = df['START TIME'].iloc[0]
    end = df['START TIME'].iloc[-1] + 60000
    channel = (1,2,13)
    data = np.loadtxt(fname,
                      delimiter=',',
                      skiprows=7,
                      usecols=channel)
    eeg = data[:,:-1]
    timestamps = data[:,-1]
    indices = np.where(np.logical_and(timestamps>=start, timestamps<=end))
    a = eeg[indices]
    rest_indices = [i * 10 * fs_Hz for i in range(7)]
    left_indices = [(i+6) * 10 * fs_Hz for i in range(7)]
    start_indices = rest_indices
    for i in range(len(start_indices) - 1):
        start = int(max(start_indices[i] + tmin * sampling_freq, 0))
        end = int(min(start_indices[i+1] + tmax * sampling_freq, start_indices[-1]))
        rest_data.append(data[start:end])
    start_indices = left_indices
    for i in range(len(start_indices) - 1):
        start = int(max(start_indices[i] + tmin * sampling_freq, 0))
        end = int(min(start_indices[i+1] + tmax * sampling_freq, start_indices[-1]))
        left_data.append(data[start:end])
else:
    csvs = ["data/March22_008/9_008-2019-3-22-14-59-0.csv",
           "data/March22_008/8_008-2019-3-22-14-45-53.csv",
           "data/March22_008/7_008-2019-3-22-14-27-46.csv",
           #"data/March22_008/6_008-2019-3-22-14-19-52.csv",
           #"data/March22_008/5_008-2019-3-22-14-10-26.csv",
           "data/March22_008/10_008-2019-3-22-15-8-55.csv",
           "data/March22_001/5-001-rest25s_left10s_right10s_MI-2019-3-22-16-35-57.csv",
           "data/March22_001/4-001-rest25s_left15s_right15s_MI-2019-3-22-16-27-44.csv",
           #"data/March22_001/6-001-rest25s_left15s_right15s_MI-2019-3-22-16-46-17.csv",
           #"data/March22_001/7-001-rest25s_left20s_right20s_MI-2019-3-22-16-54-17.csv",
           "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-28-35.csv",
           "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-30-26.csv",
           #"data/March20/time-test-JingMingImagined_10s-2019-3-20-10-35-31.csv"
           ]
    all_data = get_data(csvs)
fig1 = plt.figure("psd")
fig1.clf()
classes = ['Left', 'Right', 'Rest']
all_psds = {'Right': [], 'Left': [], 'Rest': []}
all_mu = {'Right': [], 'Left': [], 'Rest': []}
for thresh in range (1):
    threshold = 0.8 + 0.2 * thresh
    left = []
    rest = []
    right = []
    
    idx = 1
    plt.figure("psd")
    for direction, data in all_data.items():
        for trial in data:
            epochs = epoch_data(trial, 250 * window_s, int(shift*250))    # shape n x 500 x 2
            for epoch in epochs:
                mu1, mu2, freqs, psd1, psd2 = predict(epoch.T, threshold)
                all_mu[direction].append([mu1,mu2])
                all_psds[direction].append([psd1,psd2])
                if plot_psd:
                    plt.subplot(3,2,idx)
                    plt.plot(freqs, psd1)
                    plt.ylim([0,25])
                    plt.xlim([6,20])
                    plt.subplot(3,2,idx+1)
                    plt.plot(freqs, psd2)
                    plt.ylim([0,25])
                    plt.xlim([6,20])
    if cm:
        left_true = sum(left)/len(left)
        left_false = 1 - left_true
        rest_false = sum(rest)/len(rest)
        rest_true = 1 - rest_false
        overall = (sum(left) + len(rest) - sum(rest))/len(left+rest)
        print(overall)
        
        
        fig2 = plt.figure("cm", figsize=(10,10))
        fig2.clf()
        
        array = [[left_true,left_false],
             [rest_false,rest_true]]
        df_cm = pd.DataFrame(array, index = [i for i in "LR"],
                          columns = [i for i in "LR"])
        plt.subplot(3,2,thresh+1)
        sn.heatmap(df_cm, cmap =colormap,annot=True)
        plt.title("Threshold: " + "{:1.1f}".format(threshold))
        plt.subplots_adjust(hspace=0.3)
fig3 = plt.figure("scatter")
fig3.clf()
for direction, mu in all_mu.items():
    mu = np.log10(np.array(mu).T)
    plt.scatter(mu[0], mu[1], s=2)
plt.axis('scaled')
plt.show()


mean_plt = 1
if mean_plt:
    plt.figure()
    for direction, mu in all_mu.items():
        mu = np.array(mu).T
        plt.scatter(np.mean(mu[0]), np.mean(mu[1]), s=2)
    plt.axis('scaled')
last = 0
if last:
    plt.scatter(r_saved[0], r_saved[1], s=2, color='blue')
    plt.scatter(l_saved[0], l_saved[1], s=2, color='red')
all_features = []
for direction, psd in all_psds.items():
    mu_indices = np.where(np.logical_and(freqs>=10, freqs<=12))
    features = np.squeeze(np.mean(np.array(psd)[:,:,mu_indices], axis=-1))
    arr = np.hstack((features, np.full([features.shape[0],1], classes.index(direction))))
    all_features.append(arr)
data = np.vstack(all_features[:-1])
X = data[:,:-1]
Y = data[:,-1]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)

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

for name, model in models:
	kfold = model_selection.KFold(n_splits=10, random_state=seed)
	cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring=scoring)
	results.append(cv_results)
	names.append(name)
	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
	print(msg)
        
fig, ax = plt.subplots()
for direction, mu in all_mu.items():
    mu = np.log10(np.array(mu).T)
    sn.kdeplot(mu[0], mu[1], ax=ax, shade_lowest=False, alpha=0.6)
ax.collections[0].set_alpha(0)
ax.collections[10].set_alpha(0)
ax.set(aspect="equal")
"""
m = [[left_true,left_false],
     [rest_false,rest_true]]
plt.matshow(m, cmap='Greys')
left_specgram = []
rest_specgram = []

t,f,all_spectra = get_spectral_content(ch, sampling_freq, shift)
fig = plt.figure()
plot_specgram(f, all_spectra, "entire session", shift, 1)


if continuous:
    for i in range(len(start_indices) - 1):
        start = int(start_indices[i]/(sampling_freq * shift))
        end = int(start_indices[i+1]/(sampling_freq * shift))
        d = all_spectra[:,start:end]
        # this trial alternates between rest and left motor imagery
        if i % 2:
            left_specgram.append(d)
        else:
            rest_specgram.append(d)
elif psd:
    #tmin, tmax = -1, 1
    tmin, tmax = 0, 0
    plt.figure()
    for i in range(len(start_indices) - 1):
        start = int(max(start_indices[i] + tmin * sampling_freq, 0))
        end = int(min(start_indices[i+1] + tmax * sampling_freq, start_indices[-1]))
        psd, f = get_psd(ch[start:end], sampling_freq)
        if i < 2:
            print(i)
            # plot two sample epochs for fun
            #plot_specgram(f, d, 'a', shift, i + 1)
        if i % 2:
            plt.subplot(2,1,2)
            plt.plot(f, psd)
            #left_specgram.append(d)
        else:
            plt.subplot(2,1,1)
            plt.plot(f, psd)
            #rest_specgram.append(d)
else:
    #tmin, tmax = -1, 1
    tmin, tmax = 0, 0
    plt.figure()
    for i in range(len(start_indices) - 1):
        start = int(max(start_indices[i] + tmin * sampling_freq, 0))
        end = int(min(start_indices[i+1] + tmax * sampling_freq, start_indices[-1]))
        t, f, d = get_spectral_content(ch[start:end], sampling_freq)
        if i < 2:
            # plot two sample epochs for fun
            plot_specgram(f, d, 'a', shift, i + 1)
        if i % 2:
            left_specgram.append(d)
        else:
            rest_specgram.append(d)
    #resize the blocks so that they're the same length as either the minimum or maximum length block
    '''rest_specgram = resize_min(rest_specgram)
    left_specgram = resize_min(left_specgram)
    '''
    rest_specgram = resize_max(rest_specgram)
    left_specgram = resize_max(left_specgram)
    
    # plot average spectrogram of both classes
    plt.figure()
    rest_av = np.nanmean(np.array(rest_specgram), axis=0)
    plot_specgram(f, rest_av,channel_name + ' rest',shift, 1)
    left_av = np.nanmean(np.array(left_specgram), axis=0)
    plot_specgram(f, left_av,channel_name + ' left',shift, 2)
"""