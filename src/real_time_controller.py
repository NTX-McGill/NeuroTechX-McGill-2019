import serial
import sys
import time
import socketio

sio = socketio.Client()
WAITBEFOREREADING = 0.1
ser = serial.Serial('/dev/cu.usbmodem14201',baudrate = 9600, timeout = 1) # or COM5

#define the FSM states

sensor_count = 0

@sio.on('to robotics')
def on_message(data):
    instruction = data['response']  # type of motion
    print(instruction); # ADDED
    try:
        ser.write(instruction.encode('utf-8')) #send instruction
    except:
        print("Not a recognized command")

    time.sleep(WAITBEFOREREADING)
    
    while ser.in_waiting:
        sensor_data = int.from_bytes(ser.read(),byteorder='little')
        print(sensor_data)
        sio.emit('from sensors', {'left': sensor_data,
                                  'right': sensor_data,
                                  'front': sensor_data,
                                  'front-tilt': sensor_data})


"""Need to do async stuff to allow reading while waitin"""
# #Display any information sent back.
# time.sleep(WAITBEFOREREADING)
# while ser.in_waiting:
#     print(ser.read())

sio.connect('http://localhost:3000')
sio.wait()
