# Milo Wheelchair
Our Milo wheelchair is designed to help patients with locked-in syndrome. While they lose complete control of the majority of voluntary muscles, their cognitive functions are usually unaffected, and they can only communicate through eye movements and blinking(1). 

In recent years, EEG-based brain-computer interfaces (BCIs) have emerged as a method to enable patients with the locked-in syndrome to interact with the external world. 

Electroencephalography (EEG) is a technique to detect neural activity and has been widely used in conjunction with BCIs, due to its non-invasive nature, accessibility, affordability and fine temporal resolution (3). EEG-based BCIs mainly use three signals: P300 wave, steady-state visually evoked potentials (SSVEPs) and motor imagery (MI). In particular, one does not require external stimulus to generate MI, allowing the users to control external devices at free will (4). When users execute real movements or imagine movements (during a motor imagery task), there will be suppression of the mu rhythm (7-13Hz) (5). In addition to the MI, eye blinking signals have been widely used in BCI-based wheelchair control (6). Eye blinking is relatively easy to detect, as it causes large electrooculogram (EOG) potentials in the 13-15Hz range, known as beta rhythm (6). 

With our Milo wheelchair, the patients can move forward or stop simply by blinking their eyes, and can turn left or right simply by thinking about left and right hand movements. 

We also designed a caregiver mobile application, which should be used along with the wheelchair. Individuals with locked-in syndrome greatly rely on their caregivers for daily tasks. The mobile application can send real-time location of the wheelchair users to their caregivers to ensure the safety and facilitate real-time communication of the users. 
To ensure the safety, the wheelchair has a heart-rate-dependent break system. The wheelchair will stop when individuals are in danger (i.e. collision) and their heart rate increases 

## Contents
- what does each folder do
## Setup
- why do we buy the product

   

## Pipeline


## Data Collection
Medical grade Ten20 EEG conductive paste was used to secure four OpenBCI passive gold cup electrodes directly onto the scalp of the participant. The four electrodes used to collect MI data were placed along the sensorimotor cortex of the subject, rear to the frontal lobe and just before the central sulcus that separates the frontal lobe from the parietal lobe (8). Two reference electrodes were placed on the subject's two ears. C1, C2, C3, and C4 channels were used, as recommend for the detection of the optimal $\mu$-rhythm (4). For eye blinking and heart-rate data, two gold cup electrodes were placed beside the left eye and left wrist of the participants respectively. To acquire raw EEG data, a computer device configured to OpenBCI's Cyton Biosensing 32-bit board would be used. 

To simulate an indoor environment, the experiment would be conducted in a room with ambient noise level. The participants would be seated comfortably on a chair or wheelchair facing a laptop screen. Before they are connected to the electrodes, the participants would be asked to remove any earrings, necklaces, glasses and/or to untie their hair to reduce noise during EEG data collection. 

## Signal Processing
Our targeted frequency range is the mu rhythm (8-12Hz) when the subjects are at rest and beta rhythm (13-36Hz) when the subjects blink their eyes. To process real-time data, we sampled at 250 Hz with a time window of two seconds, which is a standardized protocol on EEG data (ref needed).

The signal was first notched-filtered at 60 Hz and 120 Hz to remove the power-line noise across all eight electrodes (9). In order to determine the optimal filter method for the targeted frequency range (mu band (8Hz-12Hz) and beta band (13-36Hz)), five filter designs were compared. Among \textit{Butterworth}, elliptic, \textit{Chebyshev Type I, Chebyshev Type II, }and the Window Method of the FIR filter, the frequency spectra showed that \textit{Chebyshev Type I} filter performed the best. In particular, the performance of the filter improves marginally for mu and beta rhythm detection when \textit{Chebyshev Type I} filter was used. 

After the data pre-processing, we used Power Spectral Density (PSD) to extract the power of the mu band (8-12Hz) and the beta band (13-36 Hz) with respect to frequency. We compared the periodogram method and Welch’s averaging method, and found that Welch’s method gave us a cleaner signal. 

## Machine Learning
The paradigm used to move, turn, and stop the wheelchair consists of alternating between three states: Rest, Stop, Intermediate. Motor imagery classification takes place within the intermediate state, which outputs either a full stop, or a command to turn the wheelchair in the appropriate direction. To switch from one state to another, artifacts, such as jaw-clenches, are used. A sustained artifact signal of X sec will command the wheelchair to move to the next state. 

A linear regression is used to classify, in real-time, the motor imagery state of the wheelchair user. The feature used in the regression is the average mu band power, given as the average of the frequencies of interests (8-12Hz) for all time points. The linear regression then gives a motor imagery state for every given time point. The direction with the most occurrence within a 3 second time-window is the final decision output and is fed to the wheelchair. 
If no motor imagery signals are detected and jaw-clenches are sustained, the wheelchair will go into a stop. Sustaining jaw clenches again will bring the wheelchair to move forward. 

## Neurofeedback Training
Generation of robust motor imagery can be a difficult task for most individuals without prior training, as people tend to imagine visual images of related movements intead of kinesthetic feelings of actions (4). Thus, various MI neurofeedback training methods have been proposed (4). 

The MI neurofeedback training is performed on the production tab of our user dashboard. In the production tab of the dashboard, the user is given an idea of how their motor imagery signals are being processed. The production dashboard displays a measure of the machine learning model's confidence in that a signal is the correct motor imagery signal corresponding to the labeling (i.e. correct Left, correct Right, correct Rest, etc). The dashboard displays a bar graph with the percentage accuracy of the model. 

## Hardware 
The commercially available \textit{Orthofab Oasis 2008} wheelchair was modified and customized to fit the needs of the project. The motor controller of the wheelchair was replaced with two commercial-grade 40A, 12V PWM controller connected to an Arduino Uno. Furthermore, the seat of the wheelchair was reupholstered and custom-built footrests were installed. Four motion sensors were installed around the circumference of the footrest for the implementation of the self-driving feature.

## Caregiver APP
An application capable of sending the wheelchair's location to the caregiver in real-time was designed as a safety measure for wheelchair users. A notification feature is implemented so that the caregiver receives a text via Twilio, a cloud communication platform, when the user of the wheelchair experiences trouble or distress (i.e. obstacles, trauma, high stress, malfunction, etc.). The location information is received through the location services of the user's smartphone. The measure of stress dictating whether to send an alert or not is currently based on heart rate monitoring information. Once the heart rate exceeds a pre-established threshold customized to the user’s resting heart rate, the caregiver is alerted that the user might require assistance.  

## Assisted Driving
Relying on motor imagery for finer navigation is challenging if not impossible. We therefore created an assisted-driving model which serves to refine the finer detail movements involved in straight navigation. 

The model has two primary functions: wall following and object avoidance.

In order to detect whether the user is following a wall, two ultrasonic sensors — one on the left and one on the right — are used to continuously monitor the wheelchair’s position relative to a potential wall. In order to determine if a wall is present, a linear regression model is fit to the last 5 seconds of sensor data collected from each side. A threshold on the standard error determines whether the wheelchair is approaching a wall from the side or is parallel to a wall. If a wall is detected, the optimal distance to the wall is calculated as the median of the data from 5 seconds ago to 1 second ago. If the difference between the current and optimal distances to the wall is large, a slight turn is executed to correct it.

The second function of the assisted-driving paradigm is obstacle avoidance. The two sensors used in wall following are combined with a frontward facing sensor and a sensor pointing at 45º from the vertical towards the ground. As the wheelchair approaches a small obstacle, using information about the chair’s distance from the obstacle, the algorithm determines if there is room to navigate around it. Once the obstacle has been cleared, the wheelchair continues on the straight path that it had initially set out on. If there isn’t room to navigate around the obstacle, the wheelchair comes to a complete stop and the user decides what subsequent action they wish to execute. The system uses the 45º ultrasonic sensor to detect the presence of stairs or steep hills in its upcoming path and stops if the former are detected.



For more information, see our [facebook page](https://www.facebook.com/McGillNeurotech/) or [our website](https://www.mcgillneurotech.com/).


## Partners
We couldn't do it without you guys <3
* DeepMind
* OpenBCI
* Building 21
* McGill University
* Muse
