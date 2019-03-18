'''
@author: viet
Attempt to classify the squiggly data that was provided by Santiago
'''

import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import LSTM, GRU, Dense
from xgboost import XGBClassifier
from sklearn.preprocessing import normalize, MinMaxScaler, RobustScaler
import matplotlib.pyplot as plt
plt.style.use('classic')


def plot_that_bih(X):
    xn = list(range(X.shape[0]))
    for i in range(1):
        plt.plot(xn, X[:, i])

def train_clf(clf, name, X_train, X_test, y_train, y_test):
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    scr = accuracy_score(y_test, pred)
    print('{} accuracy score: {:.4f}'.format(name, scr))

def construct_metafeatures(clf_arr, X_train, X_test, y_train):
    'This method is a load of crap and shouldnt be used ever'
    train, test = list(), list()
    for clf in clf_arr:
        print(clf)
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        test_pred = clf.predict(X_test)
        train.append(train_pred)
        test.append(test_pred)
    return np.array(train).T, np.array(test).T

def Stacking(model,train,y,test,n_fold):
    folds = StratifiedKFold(n_splits=n_fold,random_state=1)
    test_pred = []
    train_pred = []
    for train_indices, val_indices in folds.split(train,y):
        x_train, x_val = train[train_indices], train[val_indices]
        y_train, y_val = y[train_indices], y[val_indices]

        model.fit(X=x_train,y=y_train)
        train_pred.append(model.predict(x_val))
        test_pred.append(model.predict(test))
    return np.array(train_pred).T, np.array(test_pred).T

loc = '../signal_processing/'

data = np.load(loc + 'feature_matrix2')
data = np.array(data)
assert type(data) == np.ndarray

X = X2 = data[:,:-1]

y = np.array(data[:, -1], dtype=np.uint8)

# Don't ask why.
scaler = MinMaxScaler(feature_range=(0,100))
X = scaler.fit_transform(X)
#X = np.exp(X)

np.savetxt('X.csv', X, delimiter=',')
#print(y)

'''
EDA
'''
plt.figure()
plot_that_bih(X)
std = np.var(X, axis=1)
r_idx, c_idx = np.where(y == 0), np.where(y == 1)

clench = X[c_idx]
rest = X[r_idx]
c_mean, r_mean = np.mean(clench, axis=1), np.mean(rest, axis=1)
lmao = np.mean(X[:,-2])
'''
CLEAN DATA
'''
newX = []
newy = []
for row_idx in range(len(X)):
    if X[row_idx][-1] < lmao: 
        newX.append(X[row_idx])
        newy.append(y[row_idx])

X = np.array(newX)
y = np.array(newy)

#Some plotting
plt.figure()
plot_that_bih(X)
#plt.show()
#exit(0)

'''
PCA/ chi2
'''

components = 75

pca = PCA(n_components=components)
pca.fit(X)
X = pca.transform(X)
'''
selector = SelectKBest(chi2, k=components)
selector.fit(X, y)
X = selector.transform(X)
print(X.shape)
'''
#assert X.shape[1] == components
print(X.shape, y.shape)
'''
ML
'''
X_train, X_test, y_train, y_test = train_test_split(X, y, shuffle=True, random_state=2, test_size=0.2)
'''
model = Sequential()
model.add(LSTM(256, input_shape=X.shape))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='nadam', metrics=['accuracy'])

model.fit(X_train, y_train)
'''

clf1 = MLPClassifier(hidden_layer_sizes=(70,), solver='adam', learning_rate='adaptive', max_iter=2000)
clf2 = XGBClassifier(n_estimators=1000, learning_rate=0.1, max_depth=3,
                    objective='binary:logistic',gamma=7)
clf3 = SVC(C=100)
clf4 = RandomForestClassifier(n_estimators=1000)
clf5 = ExtraTreesClassifier(n_estimators=1000)
clf6 = MLPClassifier(hidden_layer_sizes=(90,60,), solver='sgd', learning_rate='adaptive',
                        learning_rate_init=0.035, max_iter=2000, momentum=0.87)
clf7 = LogisticRegression(solver='saga')
clf8 = KNeighborsClassifier(n_neighbors=3)
clf9 = KNeighborsClassifier(n_neighbors=5)
clfA = KNeighborsClassifier(n_neighbors=7)
clfB = KNeighborsClassifier(n_neighbors=9)
clfC = GaussianNB()
clfD = LinearDiscriminantAnalysis()
clfE = AdaBoostClassifier(n_estimators=500)
clfF = XGBClassifier(n_estimators=500, objective='binary:logistic', gamma=7)

'''
new_X_train, new_X_test = construct_metafeatures(
    [clf1, clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9,
    clfA, clfB, clfC, clfD, clfE], 
    X_train, X_test, y_train)
'''

n_fold = 10
train1, test1 = Stacking(clf1, X_train, y_train, X_test, n_fold)
#train2, test2 = Stacking(clf2, X_train, y_train, X_test, n_fold)

print(train1.shape, test1.shape)
exit()
new_X_train = np.concatenate((train1, train2), axis=1)
new_X_test = np.concatenate((test1, test2), axis=1)

clfF.fit(new_X_train, y_train)
pred = clfF.predict(new_X_test)
scr = accuracy_score(y_test, pred)
print('Meta classifier score: {:.4f}'.format(scr))