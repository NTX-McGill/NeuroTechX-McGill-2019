import serial
import sys
import time
import socketio

sio = socketio.Client()

ser = serial.Serial('COM3',baudrate = 9600, timeout = 1)

#define the FSM states

@sio.on('timeseries')
def on_message(data):
    instruction = data['response']  # type of motion
    try:
        ser.write(instruction.encode('utf-8')) #send instruction
    except:
        print("Not a recognized command")
        continue


"""Need to do async stuff to allow reading while waitin"""
# #Display any information sent back.
# time.sleep(WAITBEFOREREADING)
# while ser.in_waiting:
#     print(ser.read())



sio.connect('http://localhost:3000')
sio.wait()
