'''
@author: viet
'''

import numpy as np
from sklearn.utils import shuffle
from tools import *
from sklearn.preprocessing import RobustScaler
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC


fs_Hz = 250
sampling_freq = 250
window_s = 2
shift = 0.1
channel_name = 'C4'
Viet = 0
Marley = 0
Andy = 0
cm = 0
plot_psd = 0            # set this to 1 if you want to plot the psds per window
colormap = sn.cubehelix_palette(as_cmap=True)
tmin, tmax = 0,0

csvs = ["data/March22_008/10_008-2019-3-22-15-8-55.csv",
        "data/March22_008/9_008-2019-3-22-14-59-0.csv",
        "data/March22_008/8_008-2019-3-22-14-45-53.csv",    # 
            #"data/March22_008/7_008-2019-3-22-14-27-46.csv",    #actual
            #"data/March22_008/6_008-2019-3-22-14-19-52.csv",    #actual
            #"data/March22_008/5_008-2019-3-22-14-10-26.csv",    #actual
            "data/March22_001/4-001-rest25s_left15s_right15s_MI-2019-3-22-16-27-44.csv",
            "data/March22_001/5-001-rest25s_left10s_right10s_MI-2019-3-22-16-35-57.csv",
            #"data/March22_001/6-001-rest25s_left15s_right15s_MI-2019-3-22-16-46-17.csv",    #actual
            "data/March22_001/7-001-rest25s_left20s_right20s_MI-2019-3-22-16-54-17.csv",
            "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-28-35.csv",   #6
            "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-30-26.csv",
            "data/March20/time-test-JingMingImagined_10s-2019-3-20-10-35-31.csv",
            "data/March24_011/1_011_Rest20LeftRight20_MI-2019-3-24-16-25-41.csv",   #9 to 13
            "data/March24_011/2_011_Rest20LeftRight20_MI-2019-3-24-16-38-10.csv",
            "data/March24_011/3_011_Rest20LeftRight10_MI-2019-3-24-16-49-23.csv",
            "data/March24_011/4_011_Rest20LeftRight10_MI-2019-3-24-16-57-8.csv",
            "data/March24_011/5_011_Rest20LeftRight20_MI-2019-3-24-17-3-17.csv",
            "data/March29_014/1_014_rest_left_right_20s-2019-3-29-16-44-32.csv",   # 14
            "data/March29_014/2_014_rest_left_right_20s-2019-3-29-16-54-36.csv",
            "data/March29_014/3_014_AWESOME_rest_left_right_20s-2019-3-29-16-54-36.csv",
            "data/March29_014/4_014_final_run-2019-3-29-17-38-45.csv",
            #"data/March29_014/5_014_eye_blink-2019-3-29-17-44-33.csv",
            #"data/March29_014/6_014_eye_blink-2019-3-29-17-46-14.csv",
            #"data/March29_014/7_014_eye_blink-2019-3-29-17-47-56.csv",
            ]



csv_map = {"10_008-2019-3-22-15-8-55.csv": "10_008_OpenBCI-RAW-2019-03-22_15-07-58.txt",
        "9_008-2019-3-22-14-59-0.csv": "8to9_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",
        "8_008-2019-3-22-14-45-53.csv": "8to9_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",
        #"7_008-2019-3-22-14-27-46.csv": "4to7_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",  # actual
        #"6_008-2019-3-22-14-19-52.csv": "4to7_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",  # actual
        #"5_008-2019-3-22-14-10-26.csv": "4to7_008_OpenBCI-RAW-2019-03-22_13-49-24.txt",  # actual
        "5-001-rest25s_left10s_right10s_MI-2019-3-22-16-35-57.csv": "1to5_001_OpenBCI-RAW-2019-03-22_15-56-26.txt",
        "4-001-rest25s_left15s_right15s_MI-2019-3-22-16-27-44.csv": "1to5_001_OpenBCI-RAW-2019-03-22_15-56-26.txt",
        #"6-001-rest25s_left15s_right15s_MI-2019-3-22-16-46-17.csv": "6to7_001_OpenBCI-RAW-2019-03-22_16-44-46.txt",  # actual
        "7-001-rest25s_left20s_right20s_MI-2019-3-22-16-54-17.csv": "6to7_001_OpenBCI-RAW-2019-03-22_16-44-46.txt",
        "time-test-JingMingImagined_10s-2019-3-20-10-28-35.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
        "time-test-JingMingImagined_10s-2019-3-20-10-30-26.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
        "time-test-JingMingImagined_10s-2019-3-20-10-35-31.csv": 'OpenBCI-RAW-2019-03-20_10-04-29.txt',
        "1_011_Rest20LeftRight20_MI-2019-3-24-16-25-41.csv" : '011_1to3_OpenBCI-RAW-2019-03-24_16-21-59.txt',
        "2_011_Rest20LeftRight20_MI-2019-3-24-16-38-10.csv" : '011_1to3_OpenBCI-RAW-2019-03-24_16-21-59.txt',
        "3_011_Rest20LeftRight10_MI-2019-3-24-16-49-23.csv" : '011_1to3_OpenBCI-RAW-2019-03-24_16-21-59.txt',
        "4_011_Rest20LeftRight10_MI-2019-3-24-16-57-8.csv" : '011_4to6_OpenBCI-RAW-2019-03-24_16-54-15.txt',
        "5_011_Rest20LeftRight20_MI-2019-3-24-17-3-17.csv" : '011_4to6_OpenBCI-RAW-2019-03-24_16-54-15.txt',
        "1_014_rest_left_right_20s-2019-3-29-16-44-32.csv": "1_014_OpenBCI-RAW-2019-03-29_16-40-55.txt",
        "2_014_rest_left_right_20s-2019-3-29-16-54-36.csv": "2_014_OpenBCI-RAW-2019-03-29_16-52-46.txt",
        "3_014_AWESOME_rest_left_right_20s-2019-3-29-16-54-36.csv": "3_014_AWESOME_OpenBCI-RAW-2019-03-29_17-08-21.txt",
        "4_014_final_run-2019-3-29-17-38-45.csv": "4_014_OpenBCI-RAW-2019-03-29_17-28-26.txt",
        #"5_014_eye_blink-2019-3-29-17-44-33.csv": "5-7_014_OpenBCI-RAW-2019-03-29_17-41-53.txt",
        #"6_014_eye_blink-2019-3-29-17-46-14.csv": "5-7_014_OpenBCI-RAW-2019-03-29_17-41-53.txt",
        #"7_014_eye_blink-2019-3-29-17-47-56.csv": "5-7_014_OpenBCI-RAW-2019-03-29_17-41-53.txt",
        }


# Load data
load_data = 1
if load_data:
    data_dict = {}
    for csv in csvs:
        data_dict[csv] = get_data([csv], csv_map)


train_csvs = [-1]          # index of the training files we want to use
test_csvs = [2]             # index of the test files we want to use
train_csvs = [csvs[i] for i in train_csvs]
test_csvs = [csvs[i] for i in test_csvs]
print("Training sets: \n" + str(train_csvs))
print("Test sets: \n" + str(test_csvs))
train_data = merge_all_dols([data_dict[csv] for csv in train_csvs])
all_results = []
all_test_results = []
window_lengths = [1,2,4,6,8]
window_lengths = [5]
for window_s in window_lengths:
    train_psds, train_features, freqs = extract(train_data, window_s, shift, plot_psd)
    data = to_feature_vec(train_features, rest=False)
    
    X = data[:,:-1]
    Y = data[:,-1]
    validation_size = 0.20
    seed = 7
    #X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, Y, test_size=validation_size, random_state=seed)
    
    # Test options and evaluation metric
    scoring = 'accuracy'
    
    # Spot Check Algorithms
    # evaluate each model in turn
    models = []
    models.append(('LR', LogisticRegression(solver='lbfgs')))
    models.append(('LDA', LinearDiscriminantAnalysis()))
    models.append(('KNN', KNeighborsClassifier()))
    models.append(('CART', DecisionTreeClassifier()))
    models.append(('NB', GaussianNB(var_smoothing=0.001)))
    models.append(('SVM', SVC(gamma='scale', C=100)))
    results = []
    names = []
    
    # Setup train and test data
    X, Y = shuffle(X, Y, random_state=seed)
    test_dict = data_dict[test_csvs[0]]
    _, test_features, _ = extract(test_dict, window_s, shift, plot_psd)
    test_data = to_feature_vec(test_features)
    X_test = test_data[:,:-1]
    Y_test = test_data[:,-1]
    test_results = []
    
    # Scaling
    scaler = RobustScaler()
    scaler.fit(np.concatenate([X, X_test], axis=0))    
    X, X_test = scaler.transform(X), scaler.transform(X_test)
    
    print("VALIDATION")
    for name, model in models:
    	kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    	cv_results = model_selection.cross_val_score(model, X, Y, cv=kfold, scoring=scoring)
    	results.append(cv_results)
    	names.append(name)
    	msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    	print(msg)
    print("average accuracy: " + "{:2.1f}".format(np.array(results).mean() * 100))
    all_results.append(np.array(results).mean()* 100)
    print()
    
    print("TEST")
    #test_dict = data_dict[test_csvs[0]]
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
    all_test_results.append(np.array(test_results).mean() * 100)
    print()
    
    # Stuff are: X, Y for training
    # For testing: X_test, Y_test
    ''' EDA '''
    print(X.shape, X_test.shape)
    mctr, mcte = np.mean(X, axis=0), np.mean(X_test, axis=0)
    vartr, varte = np.var(X, axis=0), np.var(X_test, axis=0)
    
    # Check some stuff
    for i in range(2):
        print("Column {}: mean train: {:.2f} +- {:.2f} \t mean test: {:.2f} +- {:.2f}".format(i+1, mctr[i], vartr[i], mcte[i], varte[i]))
