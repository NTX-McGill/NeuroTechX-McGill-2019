## Hardware used
1. Orthofab Oasis 2008 Wheelchair
2. 2 commercial-grade 40A, 12V PWM controllers
3. [Arduino Uno] (Arduino Uno)
4. 4 ultrasonic motion sensors


## Arduino programming
The code uploaded to the Arduino Uno can be found in `/wheelchair_code`. After receiving and processing data from 4 motion sensors, the Arduino outputs directives (Forward, Stop, Left, Right) to the 2 controllers


<!--- This code was uploaded to the arduino which is on the wheelchair. It handles the control of the two speed controllers and %receives data from the 4 ultrasonic sensors.
The arduino recieves commamds from the computer running the 'computer code'.
A list of pin-outs are written in comments in the code.
-->


## Interface with computer 
The code that interfaces with the Arduino on the wheelchair and a computer (for Motor Imagery classification and self-driving applications) can be found in `/computer code`. 

### Connection 
To connect a computer to the Arduino, the Arduino IDE is required and the name of the COM port should be added in `/computer code/communication.py`. Depending on the input, the following commands are sent to the Arduino:
* F: for forward movement
* L: for left turn
* R: for right turn
* S: to stop
* D: for sensor readings

<!---
Python Script that runs on the computer and interfaces the ML predictions, Self driving algorithem and the wheelcahir.
To connect the computer to the Arduino, the Arduino IDE must be downloaded and under the tools menu, the name of the COM port is coppied into the code.
The following commands are then sent over:
* F: for fowrard movement
* L: for left turn
* R: for right turn
* S: to stop
* D: for sensor readings
Note that after every command, the Arduino returns the distance readings in the order specified in the comments.
-->
