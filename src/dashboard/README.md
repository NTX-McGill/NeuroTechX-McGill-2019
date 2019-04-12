# Dashboard

This is a user interface that can be used for training patients and testing the wheelchair.

## Usage

The frontend display receives data from a [Node.js](https://nodejs.org/en/) server titled server.js.

Dependencies:
* "@openbci/cyton": "^2.0.1",
* "colormap": "^2.3.0",
* "csv-writer": "^1.2.0",
* "dgram": "^1.0.1",
* "events": "^3.0.0",
* "express": "^4.16.4",
* "http": "0.0.0",
* "node-osc": "^3.0.0",
* "osc": "^2.2.4",
* "osc-js": "^2.0.3",
* "socket.io": "^2.2.0"


The server gets EEG data from the [OpenBCI_GUI](https://openbci.com/index.php/downloads) using its networking function. To use the OpenBCI's networking, open the GUI and, under networking set stream 1 to FFT with port 12345 and address fft. Set stream 2 to TimeSeries with port 12345 and address /openbci.


### Training

The training dashboard can be used to collect customized, labelled EEG data from patients. It also provdes real-time feedback.

* **Trial Name**: The data from a 'collect' run of the dashboard generates a CSV file with all the data. This allows the user to label the CSV files with a specific name.
* **Spectograms**: This is a spectogram giving real-time, spectral power of a user's EEG signals. It's generated using [p5.js](https://p5js.org/)
* **Queue and Add to Queue**: Here, a custom sequence of commands can be developed. They will be issued sequentially upon pressing "Collect".
* **Movement**: During a trial, this panel displays what the current command is.
* **Neurofeedback**: This gives real time feedback for the power of the Mu frequency band. 

1. Ensure that server.js and the OpenBCI_GUI are running and communicating
2. In a web browser, go to localhost:3000
3. Title the trial and customize the desired queue
4. When ready, press start; the data from the OpenBCI will be recorded into a CSV file saved under ..offline\data with the current date and time. You can cancel the run at anytime by pressing "Stop". 
5. The CSV file will contain a spreadsheet of labelled data (Left, Right, or Rest).

### Production

* **Predicted Movement**: By also running the real_time_ML.py script, real time predictions will be sent to the server which can relay them to the dashboard. 
* **Spectograms**: Same as in training
* **Machine Learning Predictions**: The ML script will provide values between 0 and 1.0 corresponding to the confidence of it's left/right prediction. These are displayed here.
* **Machine Learning History**: This shows a history of the ML's previous confidence for right motor imagery.

To use the production dashboard, ensure that server.js, the OpenBCI_GUI and real_time_ML.py are all running and communicating. By pressing start, the machine-learning predictions will display in real-time!



