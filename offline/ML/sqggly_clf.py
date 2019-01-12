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
from sklearn.manifold import TSNE
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.feature_selection import SelectKBest, chi2
from nbsvm import NBSVM
from sklearn.svm import SVC
from xgboost import XGBClassifier
from mlxtend.classifier import StackingCVClassifier
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

def construct_metafeatures(clf_arr, X_train, X_test, y_train, y_test, n_fold):
    'This method is a load of crap and shouldnt be used ever'
    train, test = list(), list()
    why = []
    folds = StratifiedKFold(n_splits=n_fold,random_state=1, shuffle=False)
    for clf in clf_arr:
        print(clf)
        yeet = []
        yeet1 = []
        yeeet = []
        for train_idx, val_idx in folds.split(X_train, y_train):
            
            xtr, xvl = X_train[train_idx], X_train[val_idx]
            ytr, yvl = y_train[train_idx], y_train[val_idx]

            clf.fit(xtr, ytr)
            pred = clf.predict(xvl)
            for element in pred:
                yeet.append(element)
            for element in yvl:
                yeeet.append(element)

        why.append(yeeet)
        train.append(yeet)
        clf.fit(X_train, y_train)
        test_pred = clf.predict(X_test)
        print('Accuracy score: {:.4f}'.format(accuracy_score(y_test, test_pred)))
        test.append(test_pred)  
        print()

    train, test = np.asarray(train), np.asarray(test)
    print(train.shape)
    return train.T, test.T, np.array(why[0])

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
PCA
'''

components = 75

pca = PCA(n_components=components)
pca.fit(X)
X = pca.transform(X)
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

clf1 = MLPClassifier(hidden_layer_sizes=(90,), solver='adam', learning_rate='adaptive', max_iter=2000)
clf2 = AdaBoostClassifier(base_estimator=ExtraTreesClassifier(n_estimators=1000), n_estimators=5000)
clf3 = SVC(C=100, probability=True)
clf4 = RandomForestClassifier(n_estimators=5000, max_depth=10, min_samples_split=10)
clf5 = ExtraTreesClassifier(n_estimators=5000, max_depth=7, min_samples_split=10)
clf6 = MLPClassifier(hidden_layer_sizes=(90,60,), solver='sgd', learning_rate='adaptive',
                        learning_rate_init=0.035, max_iter=2000, momentum=0.87)
#clf7 = NBSVM(alpha=1, C=15, beta=0.8) 
clf8 = KNeighborsClassifier(n_neighbors=3, weights='distance')
clf9 = KNeighborsClassifier(n_neighbors=5)
clfA = KNeighborsClassifier(n_neighbors=7)
clfB = KNeighborsClassifier(n_neighbors=9)
clfC = SVC(C=10, probability=True, gamma='scale')
clfD = SVC(C=5, probability=True, gamma='scale')
clfE = XGBClassifier(n_estimators=1000, learning_rate=0.3, max_depth=1,
        objective='binary:logistic',gamma=7)
clfF = XGBClassifier(n_estimators=1000, objective='binary:logistic', gamma=7)
#clfF = MLPClassifier(hidden_layer_sizes=(20,), max_iter=2000, learning_rate='adaptive')
#clfF = RandomForestClassifier(n_estimators=500)
sclf = StackingCVClassifier(classifiers=[clf1, clf2, clf3, clf4, clf5, clf6, clf8, clf9,
   clfA, clfB, clfC, clfD, clfE], cv=5, meta_classifier=clfF, use_probas=True, verbose=True)

'''
TUNA
'''
#clf = AdaBoostClassifier(base_estimator=ExtraTreesClassifier(n_estimators=1000), n_estimators=5000)
#train_clf(clf, 'adaboost', X_train, X_test, y_train, y_test)
#exit() 

# Final stacking
'''
sclf.fit(X_train, y_train)
pred = sclf.predict(X_test)
scr = accuracy_score(y_test, pred)
print('Meta classifier score: {:.4f}'.format(scr))
'''

n_fold=3
new_X_train, new_X_test, new_y_train = construct_metafeatures(
    [clf2, clf3, clfA, clfC, clfD, clfE], 
    X_train, X_test, y_train, y_test, n_fold)

print(new_X_train.shape, new_X_test.shape)
#print(new_y_train)
clfF.fit(new_X_train, new_y_train)
pred = clfF.predict(new_X_test)
scr = accuracy_score(y_test, pred)
print('Meta classifier score: {:.4f}'.format(scr))
