import numpy as np
import matplotlib.mlab as mlab
import socketio
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from scipy.signal import butter, lfilter, find_peaks, peak_widths,iirfilter, detrend,correlate,periodogram,welch
import random

sio = socketio.Client()

buffer_data = []
with open('model.pkl', 'rb') as fid:
    clf = pickle.load(fid)
print("STARTING")

# starting state of wheelchair
wheelchair_state = "stop"

# configs
pred_frequency = 0.2
distance_from_artifact_s = 2    # number of seconds away from artifact we require to consider a window a blink
blink_threshold_s = 1           # number of seconds of blink state required before we send 'BLINK', also the number of seconds of blink state before we send 'BLINK' again in intermediate state **must tune
change_threshold_s = 2          # number of seconds after a recent state change we require before changing the state again
decision_s = 3                  # number of seconds in intermediate before we force a decision

# thresholds
distance_from_artifact = int(distance_from_artifact_s/pred_frequency)
blink_threshold = int(blink_threshold_s/pred_frequency)
change_threshold = int(change_threshold_s/pred_frequency)
decision = int(decision_s/pred_frequency)
clear_threshold = int(1/pred_frequency)     # the minimum number of windows within the decision span that must be clear to be considered a 'turn' signal

# counters
last_artifact = 0
num_blinks = 0
last_change = 0

# buffers
prev_states = []
prev_predictions = []

def predict(ch):
    """
    BASELINE PREDICTION ALGORITHM FOR MVP
    ch has shape (2, 500)
    """
    global last_artifact
    global num_blinks
    global last_change
    global prev_predictions
    global prev_states
    
    last_change = min(100, last_change + 1) # increment the distance (in num predictions) away from the last state change
    # the min function is there in case there's overflow
    ch = np.array(ch).T
    psd1, freqs = mlab.psd(np.squeeze(ch[0]),
                                   NFFT=500,
                                   window=mlab.window_hanning,
                                   Fs=250,
                                   noverlap=0
                                   ) 
    psd2, freqs = mlab.psd(np.squeeze(ch[7]),
                                   NFFT=500,
                                   window=mlab.window_hanning,
                                   Fs=250,
                                   noverlap=0
                                   ) 
    blink_index = np.where(np.logical_and(freqs>= 5, freqs <= 6))
    blink_psd = psd1[blink_index].mean() + psd2[blink_index].mean()
    indices = np.where(np.logical_and(freqs>=15, freqs<=45))
    high_psd = psd1[indices].mean() + psd2[indices].mean()
    
    """ compute the brain state and stateful counters here """
    if (high_psd > 30):
        last_artifact = 0   # the most recent artifact was 0 seconds ago
        state = 'artifact'
    else:
        last_artifact = min(100, last_artifact + 1)          # increment distance from artifact
        if (blink_psd > 20):
            if last_artifact >= distance_from_artifact:
                state = 'blink'
            else:
                state = 'fake_blink'
        else:
            state = 'clear'
            
    if (state == 'blink'):  # if we have a blink
        num_blinks += 1
    else:
        num_blinks = 0      # reset num consecutive blinks to 0
    
    print(state, wheelchair_state)
    
    """ now use the brain state and wheelchair state to make decisions """
    if wheelchair_state == "stop" or wheelchair_state == "forward":
        if last_change > change_threshold:      # we don't want the wheelchair to switch back immediately if there's residual blinks
            if num_blinks >= blink_threshold:
                print('BLINK')
                num_blinks = 0
                return 'BLINK'
    
    if wheelchair_state == "intermediate":
        prev_states.append(state)
        if prev_states.count('blink') >= blink_threshold:
            prev_predictions = []
            prev_states = []
            return 'BLINK'
        if last_change > decision:              # we've reached decision deadline
            if prev_states.count('clear') > clear_threshold:    # enough samples are clear, we interpret the user's intention as wanting to turn
                mu_indices = np.where(np.logical_and(freqs>=10, freqs<=12))
                mu1 = psd1[mu_indices].mean()
                mu2 = psd2[mu_indices].mean()
                l, r = list(clf.predict_proba(np.array([mu1,mu2]).reshape(1,-1))[0])
                prev_predictions.append(l)
                l = np.array(prev_predictions).mean()
                r = 1 - l
                # clear buffers
                prev_predictions = []
                prev_states = []
                if l > r:
                    return 'L'
                return 'R'
            prev_predictions = []
            prev_states = []
            return 'BLINK'
        if state == 'clear' and last_change * pred_frequency >= 1:  # we're at least 1 second away from when we stopped
            if last_change * pred_frequency <= 2:                   # we're between 1 and 2 seconds away from when we stopped
                psd1, freqs = mlab.psd(np.squeeze(ch[0,250:]),      # so we use a 1 second window
                                   NFFT=250,
                                   window=mlab.window_hanning,
                                   Fs=250,
                                   noverlap=0
                                   ) 
                psd2, freqs = mlab.psd(np.squeeze(ch[7,250:]),
                                   NFFT=250,
                                   window=mlab.window_hanning,
                                   Fs=250,
                                   noverlap=0
                                   )
            mu_indices = np.where(np.logical_and(freqs>=10, freqs<=12))
            mu1 = psd1[mu_indices].mean()
            mu2 = psd2[mu_indices].mean()
            l, r = list(clf.predict_proba(np.array([mu1,mu2]).reshape(1,-1))[0])
            if l > 0.8:     # only return if we are very confident; otherwise, delay decision
                prev_predictions = []
                prev_states = []
                return 'L'
            if r > 0.8:
                prev_predictions = []
                prev_states = []
                return 'R'
            prev_predictions.append(l)  # append prediction for voting
            


    #print("{:2.1f}, {:2.1f}".format(mean_psd, blink_psd/mean_psd))

@sio.on('to ML (state)')
def change_state(received_state):

    global wheelchair_state
    global last_change
    if (received_state['state'] != wheelchair_state):
        last_change = 0
    wheelchair_state = received_state['state'] # forward, stop, turning, intermediate

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
    sio.emit('data from ML', {'response': response,
                              'left-prob': random.uniform(0, 1),
                              'right-prob': random.uniform(0, 1),
                              'blink-prob': -1})


sio.connect('http://localhost:3000')
sio.wait()
