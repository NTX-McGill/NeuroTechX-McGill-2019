import serial
import sys
import time
import socketio
import math

sio = socketio.Client()
WAITBEFOREREADING = 0.1
ser = serial.Serial('/dev/cu.usbmodem14201',baudrate = 9600, timeout = 1) # or COM5

# get rid of old data
if ser.in_waiting > 0:
    _ = ser.read(ser.in_waiting)

def get_sensor_data():
    sensor_value = int.from_bytes(ser.read(1),byteorder='little')
    if sensor_value == 0:
        return 1000
    return sensor_value/100.0


@sio.on('to robotics')
def on_message(data):
    instruction = data['response']  # type of motion
    # print(instruction); # ADDED
    try:
        ser.write(instruction.encode('utf-8')) #send instruction
    except:
        print("Not a recognized command")

    time.sleep(WAITBEFOREREADING)

    # print(ser.in_waiting)
    while ser.in_waiting > 3: # wait until all 4 bytes
        sensor_data1 = get_sensor_data()
        sensor_data2 = get_sensor_data()
        sensor_data3 = get_sensor_data()
        sensor_data4 = get_sensor_data()
        # print(sensor_data1, sensor_data2, sensor_data3, sensor_data4)
        sensor_data_dict = {'left': sensor_data1,
                            'right': sensor_data2,
                            'front': sensor_data3,
                            'front-tilt': sensor_data4}

        sio.emit('from sensors', sensor_data_dict)
        # print(sensor_data_dict)

        # print(sensor_data)
        # sensor_data_dict[direction] = sensor_data # in meters


"""Need to do async stuff to allow reading while waitin"""
# #Display any information sent back.
# time.sleep(WAITBEFOREREADING)
# while ser.in_waiting:
#     print(ser.read())

sio.connect('http://localhost:3000')
sio.wait()
