# MILO - the mind-controlled locomotive
MILO is designed to help users navigate without the use of hands or limbs. This brain-computer interface (BCI) makes use of electroencephalography (EEG), an affordable, accessible and non-invasive technique to detect brain activity. Specifically, MILO uses motor imagery (MI) signals to turn, by detecting a suppression of the mu rhythm (7-13 Hz) in the sensorimotor cortex (the brain area associated with movement) when users imagine movements. In addition to MI, eye blinking signals and jaw artefacts were used to initiate starts and stops, and to indicate the desire to turn. With MILO, users can move forward or stop by blinking their eyes or clenching their jaw, and can turn left or right by thinking about left and right hand movements. 

We also designed a caregiver mobile application, which sends the real-time location of the wheelchair user to their caregivers to ensure safety and facilitate real-time communication. It also notifies the caregiver if the user's heartrate increases abnormally or a crash occurs.

## Github Navigation 
- [Data collection raw data](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/offline/data)
- [Signal processing code](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/offline/signal_processing)
- [Signal processing data visualization](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/offline/visualization)
- [Machine learning](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/offline/ML)
- [Software](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/offline/training_software)
- [Robotics code](https://github.com/NTX-McGill/NeuroTechX-McGill-2019/tree/master/robotics)

## Setup
- how to run the code
- dependencies and materials

## Pipeline
![Project pipeline](/FiguresFolder/Fig1%20(1).png)

## Data Collection

![](/FiguresFolder/Fig2.jpg)

Medical grade Ten20 EEG conductive paste was used to secure four passive gold cup electrodes directly onto the scalp of the user. The four electrodes used to collect MI data were placed along the sensorimotor cortex according to the 10/20 System (channels C1, C2, C3 and C4), as well as two reference electrodes placed on the subject's ear lobes. For heart-rate detection, an electrode was placed on the left wrist. References and the heart-rate electrode were secured using electrical tape. To acquire raw EEG data, a laptop configured to OpenBCI's Cyton Biosensing 8-channel, 32-bit board was used. 

To collect training date, users were presented with a cue (right, left or rest) during which they were instructed to imagine moving their right hand or left hand, or to relax. Neurofeedback was provided in the form of bar plots indicating the strength of their motor imagery.

![](/FiguresFolder/Fig3%20(1).png)

## Signal Processing
Our targeted frequency range is the mu rhythm (7-13 Hz) when the subjects are at rest and beta rhythm (13-36 Hz) when the subjects blink their eyes. To process real-time data, we sampled at 250 Hz with a time window of two seconds.

The signal was first notched-filtered at 60 Hz and 120 Hz to remove power-line noise using a Chebyshev Type I filter. After data pre-processing, we used Power Spectral Density (PSD) to extract the power of the mu band and the beta band with respect to frequency using Welch’s method.

## Machine Learning

![](/FiguresFolder/Fig5.png)

The paradigm used to move, turn, and stop the wheelchair consists of alternating between three states: Rest, Stop and Intermediate. MI classification takes place within the intermediate state, which outputs either a full stop, or a command to turn the wheelchair in the appropriate direction. To switch from one state to another, artifacts, such as jaw-clenches or eye blinks, are used. A sustained artifact signal will command the wheelchair to move to the next state. 

A linear regression is used to classify the MI state of the user in real-time. The feature used in the regression is the average mu band power, given as the average of the frequencies of interest for all time points. The linear regression then gives a MI state for every given time point. The direction with the most occurrence within a 3 second time-window is the final decision output and is fed to the wheelchair. 

If no MI signals are detected and jaw-clenching or eye-blinking is sustained, the wheelchair will go into a stop. Sustaining these artifacts again will bring the wheelchair to move forward again. 

## Dashboard

## Caregiver App
An application capable of sending the wheelchair's location to the caregiver in real-time was designed as a safety measure for wheelchair users. A notification feature is implemented so that the caregiver receives a text via Twilio, a cloud communication platform, when the user of the wheelchair experiences trouble or distress (i.e. obstacles, trauma, high stress, malfunction, etc.). The location information is received through the location services of the user's smartphone. The measure of stress dictating whether to send an alert or not is currently based on heart rate monitoring information. Once the heart rate exceeds a pre-established threshold customized to the user’s resting heart rate, the caregiver is alerted that the user might require assistance.  

## Hardware 
The commercially available Orthofab Oasis 2008 wheelchair was modified and customized to fit the needs of the project. The motor controller of the wheelchair was replaced with two commercial-grade 40A, 12V PWM controllers connected to an Arduino Uno. Furthermore, the seat of the wheelchair was reupholstered and custom-built footrests were installed. Four motion sensors were installed around the circumference of the footrest for the implementation of the assisted-driving feature.

## Assisted Driving
Relying on MI for finer navigation is challenging if not impossible. We therefore created an assisted-driving model which serves to refine movements involved in straight navigation. The model has two primary functions: wall following and object avoidance.

In order to detect whether the user is following a wall, two ultrasonic sensors — one on the left and one on the right — are used to continuously monitor the wheelchair’s position relative to a potential wall. In order to determine if a wall is present, a linear regression model is fit to the last 5 seconds of sensor data collected from each side. A threshold on the standard error determines whether the wheelchair is approaching a wall from the side or is parallel to a wall. If a wall is detected, the optimal distance to the wall is calculated as the median of the data 1 to 5 seconds previously. If the difference between the current and optimal distances to the wall is large, a slight turn is executed to correct it.

The second function of the assisted-driving paradigm is obstacle avoidance. The two sensors used in wall following are combined with a frontward facing sensor and a sensor pointing at 45º from the vertical towards the ground. As the wheelchair approaches a small obstacle, using information about the chair’s distance from the obstacle, the algorithm determines if there is room to navigate around it. Once the obstacle has been cleared, the wheelchair continues on the straight path that it had initially set out on. If there isn’t room to navigate around the obstacle, the wheelchair comes to a complete stop and the user decides what subsequent action they wish to execute. The system uses the 45º ultrasonic sensor to detect the presence of stairs or steep hills in its upcoming path and stops if the former are detected.

## Partners
We couldn't do it without you guys <3
* [wrnch.ai](https://wrnch.ai/)
* [DeepMind](https://deepmind.com/)
* [OpenBCI](https://openbci.com/)
* [twilio](https://www.twilio.com/)
* [Muse](https://choosemuse.com/)
* [Building 21](https://building21.ca/)
* [McGill University Faculties of Arts](https://www.mcgill.ca/arts/),[Science](https://www.mcgill.ca/science/) and [Engineering](https://www.mcgill.ca/engineering/)
* [McGill University School of Physical and Occupational Therapy](https://www.mcgill.ca/spot/)
* Dr. Georgios Mitsis, our Faculty Advisor
* Dr. Stefanie Blain-Moraes, for lending equipment

## The Team
For more information, see our [facebook page](https://www.facebook.com/McGillNeurotech/) or [our website](https://www.mcgillneurotech.com/).
