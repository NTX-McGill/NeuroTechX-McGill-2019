"""
@author: viet
"""
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import numpy as np


def run_model(model, name, X_train, X_test, y_train, y_test):
    'Generically trains a model and does some stuff with it'
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    scr = accuracy_score(y_test, pred)
    print("{0} accuracy score: {1:.4f}".format(name, scr))
    return model


leftloc = '../visualization/spec_left.npy'
restloc = '../visualization/spec_rest.npy' 

left_specgram = np.load(leftloc)
rest_specgram = np.load(restloc)

# Reshape some stuff, we flattening things squishing them into the time dimension
left_specgram, rest_specgram = left_specgram.swapaxes(1,2), rest_specgram.swapaxes(1,2)
left_specgram = left_specgram.reshape(-1, left_specgram.shape[-1])
rest_specgram = rest_specgram.reshape(-1, rest_specgram.shape[-1])

# rest shape = (801, 251), left shape = (756, 251)

# Remove columns so that the time dimension matches up
#rest_specgram = np.delete(rest_specgram, np.s_[left_specgram.shape[0]:], axis=0)
left_specgram = np.delete(left_specgram, np.s_[rest_specgram.shape[0]:], axis=0)
#print(rest_specgram.shape) # (756, 251)

# Trim down frequencies
freqtrshld = 40
rest_specgram = np.delete(rest_specgram, np.s_[freqtrshld:], axis=1)
left_specgram = np.delete(left_specgram, np.s_[freqtrshld:], axis=1)

print(rest_specgram.shape, left_specgram.shape) # (756, 40) for both

# Left specgram contains NaN values. We deal with them
np.nan_to_num(left_specgram, copy=False)
np.nan_to_num(rest_specgram, copy=False)

# 1 is left, 0 is rest
y = np.asarray([1 if i < left_specgram.shape[0] else 0 for i in range(left_specgram.shape[0]*2)])
print(y[755], y[757]), #A little sanity check
X = np.concatenate((left_specgram, rest_specgram), axis=0)
np.random.shuffle(X)    # Shaking things up a lil bit
# Some ml stuff
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

print()
print()
#LDA
LDA = LinearDiscriminantAnalysis()
SVM = SVC(kernel='rbf', gamma='auto')

LDA = run_model(LDA, 'LDA', X_train, X_test, y_train, y_test)
SVM = run_model(SVM, 'SVM', X_train, X_test, y_train, y_test)

print('test')