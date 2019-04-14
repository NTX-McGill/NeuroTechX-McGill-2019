# MILO: The Mind-Controlled Locomotive
MILO helps people navigate without the use of hands or limbs. We think it could be especially useful for people with ALS, locked-in syndrome, or other forms of paralysis.

Our brain-computer interface makes use of electroencephalography (EEG), an affordable, accessible, and non-invasive technique to detect brain activity. Specifically, MILO uses motor imagery signals to turn, by detecting a suppression of the mu rhythm (7-13 Hz) in the sensorimotor cortex (the brain area associated with movement) when users imagine movements. In addition to motor imagery, eye blinking signals and jaw artefacts were used to initiate starts and stops, and to indicate the desire to turn. With MILO, users can toggle between moving forward and stoping by blinking their eyes or clenching their jaw. They can turn left or right by simply thinking about left and right hand movements.

We also designed a web application for caregivers, from which they can view the location of the wheelchair user in real time to ensure their safety. A text message is also sent to the caregiver if the user's heart rate is abnormal or a crash occurs.

## Github Navigation
- [`\offline`](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/offline/) contains raw EEG data and scripts for offline analysis and visualization
	- [`\offline\data`](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/offline/data) contains the raw EEG data recorded from consenting and anonymized participants. Each folder contains the recording for a single anonymized and consenting participant. The data collection paradigms are specified in the README.md of each folder
	- [`\offline\signal_processing`](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/offline/signal_processing) contains scripts for signal processing
	- [`\offline\visualization`](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/offline/visualization) contains scripts to vizualize the data
	- `\offline\ML\real_time_test_3class.py` is the optimized script for feature extraction and classification
- [`\robotics`](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/robotics) contains scripts to interface with the Arduino hardware connected to the wheelchair
- [`\src`](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/src/) contains software for the dashboard, the self-driving features, the caregiver app and the real-time rendering
	- [`\src\dashboard`](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/src/dashboard) contains the dashboard software as well the instructions to set up and launch it
	- [`\src\real-time`](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/src/real-time) contains scripts to classify EEG signals acquired in real-time, to send/receive data from the wheelchair, and for assisted driving
	- [`\src\caregiver-app`](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/src/caregiver-app) contains the web app and text messaging for the caregiver

## Project Pipeline
![Project pipeline](/figures/Fig1%20(1).png)

## Data Collection

![](/figures/Fig2.jpg)

Medical grade Ten20 EEG conductive paste was used to secure four passive gold cup electrodes directly onto the scalp of the user. The four electrodes used to collect motor imagery data were placed along the sensorimotor cortex according to the 10/20 System (channels C1, C2, C3 and C4), as well as two reference electrodes placed on the subject's ear lobes. For heart-rate detection, an electrode was placed on the left wrist. References and the heart-rate electrode were secured using electrical tape. To acquire raw EEG data, a laptop configured to OpenBCI's Cyton Biosensing 8-channel, 32-bit board was used.

To collect training date, users were presented with a cue (right, left or rest) during which they were instructed to imagine moving their right hand or left hand, or to relax. Neurofeedback was provided in the form of bar plots indicating the strength of their motor imagery.

![](/figures/Fig3%20(1).png)

## Signal Processing
Our targeted frequency range is the mu rhythm (8-13 Hz) when the subjects are at rest and beta rhythm (13-36 Hz) when the subjects blink their eyes. To process real-time data, we sampled at 250 Hz with a time window of two seconds.The signal was first notched-filtered at 60 Hz and 120 Hz to remove power-line noise using a Butterworth filter. After data pre-processing, we used Power Spectral Density (PSD) to extract the power of the mu band and the beta band with respect to frequency using Welch’s method.

## Machine Learning

![](/figures/Fig5.png)

The paradigm used to move, turn, and stop the wheelchair consists of alternating between three states: Rest, Stop and Intermediate. motor imagery classification takes place within the intermediate state, which outputs either a full stop, or a command to turn the wheelchair in the appropriate direction. To switch from one state to another, artifacts, such as jaw-clenches or eye blinks, are used. A sustained artifact signal will command the wheelchair to move to the next state.

A linear regression is used to classify the motor imagery state of the user in real-time. The feature used in the regression is the average mu band power, given as the average of the frequencies of interest for all time points. The linear regression then gives a motor imagery state for every given time point. The direction with the most occurrence within a 3 second time-window is the final decision output and is fed to the wheelchair.

If no motor imagery signals are detected and jaw-clenching or eye-blinking is sustained, the wheelchair will go into a stop. Sustaining these artifacts again will bring the wheelchair to move forward again.

## Dashboard
Our dashboard acts as the hub for both data collection and the real time control of the wheelchair.

During data collection, the experimenter can create a queue of thoughts ("left", "right", "rest") for the subject to think about. After each trial, a CSV file is written with the EEG data collected as well as its corresponding though label. Spectrograms are displayed in real time to assist the experimenter. Neurofeedback is also integrated by displaying the mu bands.

During the real time control of the wheelchair, bar graphs and charts display the machine learning confidence. The sensor readings are also shown on the screen.

![](/figures/Fig4.png)

## Caregiver Web App
An application capable of sending the wheelchair's location to the caregiver in real-time was designed as a safety measure for wheelchair users. A notification feature is implemented so that the caregiver receives a text via Twilio, a cloud communication platform, when the user of the wheelchair experiences trouble or distress (i.e. obstacles, trauma, high stress, malfunction, etc.). The location information is received through the location services of the user's smartphone. The measure of stress dictating whether to send an alert or not is currently based on heart rate monitoring information. Once the heart rate exceeds a pre-established threshold customized to the user’s resting heart rate, the caregiver is alerted that the user might require assistance.  

## Hardware
The commercially available Orthofab Oasis 2008 wheelchair was modified and customized to fit the needs of the project. The motor controller of the wheelchair was replaced with two commercial-grade 40A, 12V PWM controllers connected to an Arduino Uno. Furthermore, the seat of the wheelchair was reupholstered and custom-built footrests were installed. Four motion sensors were installed around the circumference of the footrest for the implementation of the assisted-driving feature.

![](/figures/Fig6.png)

## Assisted Driving
Relying on motor imagery for finer navigation is challenging if not impossible. We therefore created an assisted-driving model which serves to refine movements involved in straight navigation. The model has two primary functions: wall following and object avoidance.

In order to detect whether the user is following a wall, two ultrasonic sensors — one on the left and one on the right — are used to continuously monitor the wheelchair’s position relative to a potential wall. In order to determine if a wall is present, a linear regression model is fit to the last 5 seconds of sensor data collected from each side. A threshold on the standard error determines whether the wheelchair is approaching a wall from the side or is parallel to a wall. If a wall is detected, the optimal distance to the wall is calculated as the median of the data 1 to 5 seconds previously. If the difference between the current and optimal distances to the wall is large, a slight turn is executed to correct it.

The second function of the assisted-driving paradigm is obstacle avoidance. The two sensors used in wall following are combined with a frontward facing sensor and a sensor pointing at 45º from the vertical towards the ground. As the wheelchair approaches a small obstacle, using information about the chair’s distance from the obstacle, the algorithm determines if there is room to navigate around it. Once the obstacle has been cleared, the wheelchair continues on the straight path that it had initially set out on. If there isn’t room to navigate around the obstacle, the wheelchair comes to a complete stop and the user decides what subsequent action they wish to execute. The system uses the 45º ultrasonic sensor to detect the presence of stairs or steep hills in its upcoming path and stops if the former are detected.

## Partners
* [wrnch.ai](https://wrnch.ai/)
* [DeepMind](https://deepmind.com/)
* [OpenBCI](https://openbci.com/)
* [Twilio](https://www.twilio.com/)
* [Muse](https://choosemuse.com/)
* [Building 21](https://building21.ca/)
* [McGill University Faculties of Arts](https://www.mcgill.ca/arts/),[Science](https://www.mcgill.ca/science/) and [Engineering](https://www.mcgill.ca/engineering/)
* [McGill University School of Physical and Occupational Therapy](https://www.mcgill.ca/spot/)

A special thank you to [Dr. Georgios Mitsis](https://www.mcgill.ca/bioengineering/people/faculty/georgios-mitsis), our faculty advisor, and to [Dr. Stefanie Blain-Moraes](https://www.mcgill.ca/spot/stefanie-blain-moraes) for lending equipment.

## The Team
We are an interdisciplinary group of dedicated undergraduate students from McGill University and our mission is to raise awareness and interest in neurotechnology.
For more information, see our [facebook page](https://www.facebook.com/McGillNeurotech/) or [our website](https://www.mcgillneurotech.com/).
