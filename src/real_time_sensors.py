import serial
import sys
import time
import socketio

sio = socketio.Client()
WAITBEFOREREADING = 0.1
ser = serial.Serial('/dev/cu.usbmodem14201',baudrate = 9600, timeout = 1) # or COM5

#define the FSM states

sensor_count = 0
sensor_data_dict = {'left': 0,
                    'right': 0,
                    'front': 0,
                    'front-tilt': 0}
while ser.in_waiting:
    sensor_data = int.from_bytes(ser.read(),byteorder='little')
    print(sensor_data)
    if sensor_count == 0:
        sensor_data_dict["left"] = sensor_data
    else if sensor_count == 1:
        sensor_data_dict["right"] = sensor_data
    else if sensor_count == 2:
        sensor_data_dict["front"] = sensor_data
    else:
        sensor_data_dict["front-tilt"] = sensor_data
        sio.emit('from sensors', sensor_data_dict)
        print(sensor_data_dict)
        sensor_data = -1
    sensor_data += 1

"""Need to do async stuff to allow reading while waitin"""
# #Display any information sent back.
# time.sleep(WAITBEFOREREADING)
# while ser.in_waiting:
#     print(ser.read())

sio.connect('http://localhost:3000')
sio.wait()
