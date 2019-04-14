# Machine Learning
## Dependencies
* [Python 3.6](https://www.python.org/download/releases/2.7/) or later
* [Numpy 1.16](http://www.numpy.org/) or later
* [Matplotlib 3.0.3](https://matplotlib.org/) or later
* [Pandas 0.23.0](https://pandas.pydata.org) or later
* [SciPy 1.2.1](https://www.scipy.org/) or later
* [scikit Learn 0.20.3](https://scikit-learn.org/stable/)

## Multi-class classfication problem
The paradigm used to move, turn, and stop the wheelchair consists of alternating between three states: Rest, Stop, Intermediate. Motor imagery classification takes place within the intermediate state, which outputs either a full stop, or a command to turn the wheelchair in the appropriate direction. To switch from one state to another, artifacts, such as jaw-clenches, are used. While artifacts can be detected through thresholding of their mean PSD values, regression is required to classify the motor imagery state of the wheelchair user. The feature used in the regression is the average mu band power, given as the average of the frequencies of interests (8-13Hz) for all time points. 
## Classifier
Different classification algorithms (Logistic Regression, Linear Discriminant Analysis (LDA), Decision Tree, Support Vector Machine, Gaussian Naive Bayes, K-nearest neighbors) were explored. The script can be found in `real_time_test_3class.py`. Because of the omnipresent trade-off between latency and accuracy, am optimal time-window was required to be selected. LDA consistently yielded optimal accuracy of around 82% for a 3 second time-window. The model that was integrated in the real-time script is `model.pk`. 




