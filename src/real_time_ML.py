import numpy as np
import matplotlib.mlab as mlab
import socketio
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import butter, lfilter, find_peaks, peak_widths,iirfilter, detrend,correlate,periodogram,welch


sio = socketio.Client()

buffer_data = []
with open('../offline/model.pkl', 'rb') as fid:
    clf = pickle.load(fid)
print("STARTING")

def filter_(raw_eeg_data,bp_lowcut =1, bp_highcut =60, bp_order=5,
            notch_freq_Hz  = [60, 120], notch_order =5):
   nyq = 0.5 * 250 #Nyquist frequency
   low = bp_lowcut / nyq
   high = bp_highcut / nyq
   
   #Butter
   b_bandpass, a_bandpass = butter(bp_order, [low , high], btype='band', analog=True)
   
   bp_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_bandpass, a_bandpass ,l),0,raw_eeg_data)

   notch_filtered_eeg_data = bp_filtered_eeg_data
   
   low1  = notch_freq_Hz[0]
   high1 = notch_freq_Hz[1]
   low1  = low1/nyq
   high1 = high1/nyq
   
   b_notch, a_notch = iirfilter(2, [low1, high1], btype='bandstop')
   notch_filtered_eeg_data = np.apply_along_axis(lambda l: lfilter(b_notch, a_notch ,l),
                                                     0,bp_filtered_eeg_data)
       

   return np.array(notch_filtered_eeg_data)
def predict(ch):
    """
    BASELINE PREDICTION ALGORITHM FOR MVP
    ch has shape (2, 500)
    """
    threshold = 1
    #ch = np.array(ch)
    
    #ch = filter_(ch)
    ch = np.array(ch).T
    
    psd1,freqs = mlab.psd(np.squeeze(ch[0]),
                           NFFT=500,
                           Fs=250)
    mu_indices = np.where(np.logical_and(freqs>=10, freqs<=12))
    mu1 = np.mean(psd1[mu_indices])

    psd2,freqs = mlab.psd(np.squeeze(ch[7]),
                           NFFT=500,
                           Fs=250)
    mu2 = np.mean(psd2[mu_indices])
    blink = psd2[70:100].mean() + psd1[70:100].mean()
    print(blink)
    result = list(clf.predict_proba(np.array([mu1,mu2]).reshape(1,-1))[0])
    print(result)
    if result[0] > 0.6:
        return "L"
    elif result[1] > 0.6:
        return "R"
    else:
        return "F"


@sio.on('timeseries-prediction')
def on_message(data):
    global buffer_data
    buffer_data += data['data'] # concatenate lists

    if len(buffer_data) < 500:
        # lacking data
        response = "F" # go forward otherwise
    else:
        # we have enough data to make a prediction
        to_pop = len(buffer_data) - 500
        buffer_data = buffer_data[to_pop:]
        response = predict(buffer_data)
    print('hi')
    sio.emit('data from ML', {'response': response,
                              'left-prob': 0.4,
                              'right-prob': 0.6,
                              'blink-prob': -1})


sio.connect('http://localhost:3000')
sio.wait()
