# Signal processing

## EEG
Problems associated with EEG is that it is noisy, non-stationary, complex and high dimensionality

### Filtering
We explored different Finite Impulse Response (FIR) and Infinite Impulse Response (IIR_ filters). 
Examples of IIR filters include Butterworth, Chebyshev Type I and II, Elliptic.
Because Infinite impulse response (IIR) filters are known to be more optimal than FIR filters for digital signal processing, 
we decided to focus on IIR filters. 
![Filtering techniques](/figures/filters)
We also explored different filter orders. We used a second-order zero-phase Butterworth filter because of the 
passband and stopbands being the maximially flat between all the filters. To remove power line noises, we notch 
filtered our signal at 60 Hz. We then proceeded to bandpass filter between 1 and 40 Hz, which corresponds to the band 

### Epoching
We separated our raw EEG signal into windows of two seconds to obtain an optimal trade-off between accuracy and latency.

### Feature extraction
Spatial features consisted of isolating channels of interest. In our case, imagined right/left hand movements 
are isolated at electrodes localized over the motor cortex areas of the brain ( for example, around locations C3 and C4 
for right and  left hand movements respectively). 
To obtain spectral features, we computed the signal power through the power spectral density (PSD) To do so, 
we explored two different methods, including the periodogram and Welch's method. While the Welch PSD leads to smoother plots, 
the peridogram described the data more accurately. Thus, we decided to use with the periodogram method. 

The spectral features therefore consisted of obtaining mean PSD values at frequencies of interest 
(μ band from about 8 − 12 Hz ).
![Periodogram vs Welch](/figures/welchVperiodogram)

## ECG
The peaks were determined through a peak detection algorithm. The RR intervals were obtained via the distance between the peaks.
The heart rate was subsquently determined from the mean RR intervals. 
![Heart rate](/figures/heart)




