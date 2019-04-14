# Visualization
Scripts to vizualize the raw EEG data

## Dependencies
* [Python 3.6](https://www.python.org/download/releases/2.7/) or later
* [Numpy 1.16](http://www.numpy.org/) or later
* [Matplotlib 3.0.3](https://matplotlib.org/) or later
* [Pandas 0.23.0](https://pandas.pydata.org) or later
* [SciPy 1.2.1](https://www.scipy.org/) or later
* [MNE](https://martinos.org/mne/stable/documentation.html)

## Power spectral densities
Power spectral densities describe how the power of a signal at a certain time interval (typically a two-second window) is distributed among several frequency components. `collection_psd_plot.py` contains the script to generate power spectral densities using our data.
![alt text](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/blob/master/offline/visualization/psd_whitebg.png)

## Spectrograms
Spectrograms are visual representations of how the spectrum of frequencies of a signal varies with time. Frequency components with more power are colored more prominently. Spectrograms are therefore invaluable tools to visualize certain states. For example, as demomonstrated by the figure below, left motor imagery is characterized by a suppression in the mu band (7-13 Hz; shown more prominently in the figure by a suppresion at 10 Hz). `physionet_epoched_spectrogram.py` contains the script to generate spectrograms using publicly available motor imagery datasets (https://physionet.org/pn4/eegmmidb/). `OpenBCI_MI.py` contains the script to generate spectrograms from our data (which is found [here](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/offline/data))
![alt text](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/blob/master/offline/visualization/average_spectrogram_rest_vs_left.png)


## Prediction visualization
`OpenBCI_MI_analysis.py` contains the script to visualize the predictions outputed by a classification algorithm. Histograms are used to visualize the mean overall power in the mu band. As seen in the following figure, as expected, "left" states are described by a reduction in power in the mu band. 
![alt text](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/blob/master/offline/visualization/histogram_march_4_trial_5.png)

