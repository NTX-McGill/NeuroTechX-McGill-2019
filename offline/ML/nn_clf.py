import numpy as np
from scipy import signal
import matplotlib.mlab as mlab
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix , accuracy_score
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import train_test_split
import tensorflow as tf
tf.logging.set_verbosity(tf.logging.ERROR)
import warnings
warnings.filterwarnings("ignore")


def filter_(arr, fs_Hz, lowcut, highcut, order):
   nyq = 0.5 * fs_Hz
   b, a = signal.butter(1, [lowcut/nyq, highcut/nyq], btype='band')
   for i in range(0, order):
       arr = signal.lfilter(b, a, arr, axis=0)
   return arr

def get_start_indices(ch):
    start_indices = [100]
    i = 0
    while i < len(ch):
        if ch[i] > 100:
            start_indices.append(i)
            i += 500
        i += 1
    return start_indices

def get_spectral_content(ch, fs_Hz, shift=0.1):
    NFFT = fs_Hz*2
    overlap = NFFT - int(shift * fs_Hz)
    spec_PSDperHz, spec_freqs, spec_t = mlab.specgram(np.squeeze(ch),
                                   NFFT=NFFT,
                                   window=mlab.window_hanning,
                                   Fs=fs_Hz,
                                   noverlap=overlap
                                   ) # returns PSD power per Hz
    # convert the units of the spectral data
    spec_PSDperBin = spec_PSDperHz * fs_Hz / float(NFFT)
    return spec_t, spec_freqs, spec_PSDperBin  # dB re: 1 uV

def merge_spectograms(spec):
    merged = np.array(spec[0])
    for i in range(1, len(spec)):
        merged = np.concatenate((merged, np.array(spec[i])), axis=1)

    return merged


def get_sliding_windows(arr, window_len=10): #10 is about 1 second
    assert arr.shape[0] > window_len

    res = np.array(arr[0:0+window_len])
    for i in range(1, len(arr)-window_len+1):
        res = np.vstack((res, arr[i:i+window_len]))

    return res


def get_temporal_power_features(fname, window_len):
    sampling_freq = 250
    shift = 0.1
    channel = (1)

    data = np.loadtxt(fname, delimiter=',', skiprows=7, usecols=channel)

    data_filtered = filter_(data.T, sampling_freq, 1, 40, 1)

    start_indices = get_start_indices(data_filtered)

    t_all, f,  all_spectra = get_spectral_content(data_filtered, sampling_freq, shift)

    left_specgram = []
    rest_specgram = []

    for i in range(len(start_indices) - 1):
        start = int(max(start_indices[i], 0))
        end = int(min(start_indices[i+1], start_indices[-1]))
        t, f, d = get_spectral_content(data_filtered[start:end], sampling_freq)
        if i % 2:
            left_specgram.append(d)
        else:
            rest_specgram.append(d)

    mu_indices = np.where(np.logical_and(f>=7, f<=12))

    rest_combined = merge_spectograms(rest_specgram)
    left_combined = merge_spectograms(left_specgram)

    # This is basically filtering only mu freq components and taking mean power across mu frequencies
    rest_combined_mu = np.mean(rest_combined[mu_indices[0],:], axis=0)
    left_combined_mu = np.mean(left_combined[mu_indices[0],:], axis=0)

    rest_windows = get_sliding_windows(rest_combined_mu, window_len)
    left_windows = get_sliding_windows(left_combined_mu, window_len)

    labels = np.array([False]*len(rest_windows) + [True]*len(left_windows))

    return np.concatenate((rest_windows, left_windows)), labels


def run_random_forest(X_train, X_test, y_train, y_test):
    print("===================================================================================")
    print("Running RandomForest classifier...")
    clf = RandomForestClassifier()
    clf.fit(X_train, y_train)
    print_results(clf, X_test, y_test)

def get_nn_model():
    model = Sequential()
    model.add(Dense(60, input_dim=window_len, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(30, kernel_initializer='normal', activation='relu'))
    model.add(Dense(10, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, kernel_initializer='normal', activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def run_nn(X_train, X_test, y_train, y_test):
    print("===================================================================================")
    print("Running Neural Network classifier...")
    estimator = KerasClassifier(build_fn=get_nn_model, epochs=10, batch_size=32, verbose=0)
    estimator.fit(X_train, y_train)
    print_results(estimator, X_test, y_test)

def print_results(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    print('Accuracy: {:.4f}'.format(accuracy_score(y_test, y_pred)))
    print('\nConfusion Matrix')
    print(confusion_matrix(y_test, y_pred))
    print('\nClassification Report')
    print(classification_report(y_test, y_pred))


fname = '../data/March 4/5_SUCCESS_Rest_RightAndJawClench_10secs.txt'
fname2 = '../data/March 4/7_SUCCESS_Rest_RightClenchImagineJaw_10secs.txt'
window_len = 10

X_data, y_data = get_temporal_power_features(fname, window_len)
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

run_random_forest(X_train, X_test, y_train, y_test)
run_nn(X_train, X_test, y_train, y_test)

