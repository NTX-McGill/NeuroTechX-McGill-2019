### Wheelchair Code

This code was uploaded to the arduino which is on the wheelchair. It handles the control of the two speed controllers and receives data from the 4 ultrasonic sensors.
The arduino recieves commamds from the computer running the 'computer code'. 

A list of pin-outs are written in comments in the code.

### Computer Code

Python Script that runs on the computer and interfaces the ML predictions, Self driving algorithem and the wheelcahir.

To connect the computer to the Arduino, the Arduino IDE must be downloaded and under the tools menu, the name of the COM port is coppied into the code.

The following commands are then sent over:
* F: for fowrard movement
* L: for left turn
* R: for right turn
* S: to stop
* D: for sensor readings

Note that after every command, the Arduino returns the distance readings in the order specified in the comments.
