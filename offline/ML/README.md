# Machine Learning
## Multi-class classfication problem
The paradigm used to move, turn, and stop the wheelchair consists of alternating between three states: Rest, Stop, Intermediate. Motor imagery classification takes place within the intermediate state, which outputs either a full stop, or a command to turn the wheelchair in the appropriate direction. To switch from one state to another, artifacts, such as jaw-clenches, are used. While artifacts can be detected through thresholding of their mean PSD values, regression is required to classify the motor imagery state of the wheelchair user. The feature used in the regression is the average mu band power, given as the average of the frequencies of interests (8-12Hz) for all time points. 
## Classifier
Different classification algorithms (Logistic Regression, Linear Discriminant Analysis (LDA), Decision Tree, Support Vector Machine, Gaussian Naive Bayes, K-nearest neighbors) were explored. Because of the omnipresent trade-off between latency and accuracy, am optimal time-window was required to be selected. LDA consistently yielded optimal accuracy of around 82% for a 3 second time-window. 




