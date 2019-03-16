'''
@author: viet
Attempt to classify the squiggly data that was provided by Santiago
'''
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.preprocessing import normalize
import matplotlib.pyplot as plt
plt.style.use('classic')

loc = '../signal_processing/'

data = np.load(loc + 'feature_matrix')
data = np.array(data)
assert type(data) == np.ndarray

X = np.log(data[:,:-1]) # normalization, kind of
#X = data[:,-3:-1]/10000
X = normalize(X, axis=0)
y = np.array(data[:, -1], dtype=np.uint8)

'''
EDA
'''
variance = np.var(X, axis=1)
#print(variance[:19])
r_idx, c_idx = np.where(y == 0), np.where(y == 1)
#print(c_idx)
clench = X[c_idx]
rest = X[r_idx]
c_mean, r_mean = np.mean(clench, axis=1), np.mean(rest, axis=1)
print(np.mean(c_mean), np.mean(r_mean))
print(X)
plt.scatter(X[:,0], X[:,1], marker='o', c=y, cmap='viridis', alpha=0.3)
plt.xlabel('per_win_max')
plt.ylabel('per_win_mean')
#plt.show()

'''
ML
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1, test_size=0.2)


clf = AdaBoostClassifier(n_estimators=2000)

clf.fit(X_train, y_train)
pred = clf.predict(X_test)
scr = accuracy_score(y_test, pred)
print(pred)
print('clf accuracy score: {:.4f}'.format(scr))