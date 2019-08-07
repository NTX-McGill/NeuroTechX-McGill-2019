"""
Created on Sun Mar 17 09:03:30 2019
@author: marley
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
from sklearn.utils import shuffle
from sklearn.preprocessing import RobustScaler
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn import model_selection
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
import seaborn as sn
from sklearn.utils.multiclass import unique_labels
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy.fft as fft
import numpy as np
from scipy import stats
from scipy import signal

warnings.filterwarnings("ignore")

from metadata import MARKER_DATA, LABELS, FILES_BY_SUBJECT, ALL_FILES
import file_utils

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

    # print(cm)

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



def plot_specgram(spec_freqs, spec_PSDperBin, title, shift, i=1):
    f_lim_Hz = [0, 20]   # frequency limits for plotting
    # plt.figure(figsize=(10,5))
    spec_t = [idx*.1 for idx in range(len(spec_PSDperBin[0]))]
    plt.subplot(3, 1, i)
    plt.title(title)
    plt.pcolor(spec_t, spec_freqs, 10*np.log10(spec_PSDperBin))  # dB re: 1 uV
    plt.clim([-25, 26])
    plt.xlim(spec_t[0], spec_t[-1]+1)
    plt.ylim(f_lim_Hz)
    plt.xlabel('Time (sec)')
    plt.ylabel('Frequency (Hz)')
    plt.subplots_adjust(hspace=1)


def epoch_data(data, window_length, shift, maxlen=2500):
    arr = []
    start = 0
    #if maxlen > len(data):
        #start = int((maxlen - len(data))/2)
    i = start
    maxlen = min(len(data), maxlen)
    while i + window_length < start + maxlen:
        arr.append(data[i:i+window_length])
        i += shift
    return np.array(arr)


def extract_features(all_data, window_s, shift, plot_psd=False, keep_trials=False,scale_by=None):
    all_psds = {'Right': [], 'Left': [], 'Rest': []}
    all_features = {'Right': [], 'Left': [], 'Rest': []}

    idx = 1
    fig1 = plt.figure("psd")
    fig1.clf()
    for direction, data in all_data.items():
        for trial in data:
            epochs = epoch_data(trial, int(250 * window_s), int(shift*250))    # shape n x 500 x 2
            trial_features = []
            for epoch in epochs:
                features, freqs, psd1, psd2 = get_features(epoch.T, scale_by=scale_by)
                all_psds[direction].append([psd1, psd2])
                trial_features.append(features)
                if plot_psd:
                    plt.subplot(3, 2, idx)
                    plt.plot(freqs, psd1)
                    plt.ylim([0, 25])
                    plt.xlim([6, 20])
                    plt.subplot(3, 2, idx+1)
                    plt.plot(freqs, psd2)
                    plt.ylim([0, 25])
                    plt.xlim([6, 20])
            if trial_features:
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
        arr = np.hstack((features, np.full([features.shape[0], 1], classes.index(direction))))
        feature_arr.append(arr)
    if not rest or not len(features):
        feature_arr = feature_arr[:-1]
    return np.vstack(feature_arr)


def normalize(features_dict):
    all_features = to_feature_vec(features_dict, rest=True)
    av = list(np.mean(all_features, axis=0))
    mean_coeff = np.array([el/sum(av[:-1]) for el in av[:-1]])
    for direction, features in features_dict.items():
        features = [np.divide(example, mean_coeff) for example in features]
        features_dict[direction] = features
    
def running_mean(x, N):
   cumsum = np.cumsum(np.insert(x, 0, 0)) 
   return (cumsum[N:] - cumsum[:-N]) / N


def get_features(arr, scale_by=None):
    # ch has shape (2, 500)
    channels = [0, 1, 6, 7]
    channels=[0,7]

    psds_per_channel = []
    nfft = 500
    if arr.shape[-1] < 500:
        nfft = 250
    for ch in arr[channels]:
        #freqs,psd = signal.periodogram(np.squeeze(ch),fs=250, nfft=500, detrend='constant')
        psd, freqs = mlab.psd(np.squeeze(ch),NFFT=nfft,Fs=250)
        psds_per_channel.append(psd)
    psds_per_channel = np.array(psds_per_channel)
    mu_indices = np.where(np.logical_and(freqs >= 10, freqs <= 12))
    
    #features = np.amax(psds_per_channel[:,mu_indices], axis=-1).flatten()   # max of 10-12hz as feature
    features = np.mean(psds_per_channel[:, mu_indices], axis=-1).flatten()   # mean of 10-12hz as feature
    if scale_by:
        scale_indices = np.where(np.logical_and(freqs >= scale_by[0], freqs <= scale_by[-1]))
        scales = np.mean(psds_per_channel[:,scale_indices],axis=-1).flatten()
        temp.append(scales)
        features = np.divide(features, scales)
    #features = np.array([features[:2].mean(), features[2:].mean()])
    # features = psds_per_channel[:,mu_indices].flatten()                     # all of 10-12hz as feature
    return features, freqs, psds_per_channel[0], psds_per_channel[-1]


fs_Hz = 250
sampling_freq = 250
shift = 0.25
channel_name = 'C4'
plot_psd = False
colormap = sn.cubehelix_palette(as_cmap=True)
tmin, tmax = 0, 0
normalize_ = 1

dataset = file_utils.load_all()
subjects = [i for i in range(len(FILES_BY_SUBJECT))]             # index of the test files we want to use

window_lengths = [1, 2, 4]
all_results = []
all_test_results = []
validation = False
test = True
seed = 7
normalize_ = True
run_pca = False

# Perform leave-one-subject-out cross-validation for each subject
for subj in subjects:
    test_csvs = FILES_BY_SUBJECT[subj]
    train_csvs = [el for el in ALL_FILES if el not in test_csvs]
    train_data = prepro.merge_all_dols([dataset[csv] for csv in train_csvs])
    
    # print subject name
    print(test_csvs[0].split('/')[1])
    
    scale_by = None
    all_wtest_results = []
    for window_s in window_lengths:
        train_psds, train_features, freqs = extract_features(train_data, window_s, shift, plot_psd, scale_by=scale_by)
        data = to_feature_vec(train_features, rest=False)
        
        X = data[:, :-1]
        Y = data[:, -1]
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
        if run_pca:
            pca = PCA(n_components=2, svd_solver='full')
            pca.fit(X)
            X = pca.transform(X)
        
        
        subj_results = []
        for csv in test_csvs:
            print(csv)
            test_dict = data_dict[csv]
            _, test_features, _ = extract_features(test_dict, window_s, shift, plot_psd, scale_by=scale_by)
            if normalize_:
                normalize(test_features)
            test_data = to_feature_vec(test_features)
            print(np.mean(test_data, axis=0))
            X_test = test_data[:, :-1]
            Y_test = test_data[:, -1]
            if run_pca:
                X_test = pca.transform(X_test)
            print(np.mean(X_test, axis=0))
            
            if validation:
                print("VALIDATION")
                for name, model in models:
                    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
                    cv_results = model_selection.cross_val_score(model, X_test, Y_test, cv=kfold, scoring=scoring)
                    results.append(cv_results)
                    names.append(name)
                    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
                    print(msg)
                print("average accuracy: " + "{:2.1f}".format(np.array(results).mean() * 100))
                all_results.append(np.array(results).mean() * 100)
                print()
                
            if test:
                print("TEST")
                test_results = []
                for name, model in models:
                    model.fit(X, Y)
                    score = model.score(X_test, Y_test)
                    msg = "%s: %f" % (name, score)
                    print(msg)
                    test_results.append(score)
                print("test accuracy:")
                print("{:2.1f}".format(np.array(test_results).mean() * 100))
                subj_results.append(np.array(test_results).mean() * 100)
            
                # X, Y for training
                # For testing: X_test, Y_test
                ''' EDA '''
                print(X.shape, X_test.shape)
                mctr, mcte = np.mean(X, axis=0), np.mean(X_test, axis=0)
                vartr, varte = np.var(X, axis=0), np.var(X_test, axis=0)
                print()
        all_wtest_results.append([np.array(subj_results).mean(),stats.sem(np.array(subj_results))])
    all_test_results.append(all_wtest_results)
    
print(np.array(all_test_results).mean())