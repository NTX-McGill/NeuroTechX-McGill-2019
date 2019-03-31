import numpy as np
import socketio

sio = socketio.Client()

# assume we get real-time data on sensor info: assume we get
buffer = 0.1   # to allow for error/stopping before we actually hit something

collisionThreshold = 0.5 + buffer # chair has a 0.5m braking distance
sensorHeight = .2  # final height = 20 cm
sensorAngle = np.deg2rad(60)    # ALSO CHANGE THIS
rampAngle = np.deg2rad(7.5)

mathvalue = (sensorHeight*np.tan(sensorAngle) - collisionThreshold)*np.sin(rampAngle) # see diagram
stopThreshold = slowThreshold + mathvalue/np.cos(rampAngle + sensorAngle) + buffer # stop if not a ramp

def collisionDetection(left_forw, right_forw, front_forw, front_tilt): # assume these are distances
    l_out, r_out, f_out = 0,0,0
    if left_forw <= collisionThreshold: l_out = 1
    if right_forw <= collisionThreshold: r_out = 1
    if front_forw <= collisionThreshold: f_out = 1
    f_out = max(stairDetection(front_tilt), f_out)
    return (l_out, r_out,  f_out)
#    return detection(left_forw, right_forw, front_forw, collisionThreshold, 1, (0,0,0))

# returns 1 if need to stop
def stairDetection(front_tilt):
    if front_tilt >= stopThreshold:
        return 1;

@sio.on('to safety')
def on_message(sensor_data):
    l = sensor_data['left']
    r = sensor_data['right']
    f = sensor_data['front']
    f_t = sensor_data['front-tilt']

    l_out, r_out, f_out = collisionDetection(l, r, f, f_t)

    sio.emit('from safety', {'left': l_out,
                             'right': r_out,
                             'forward': f_out})

sio.connect('http://localhost:3000')
sio.wait()
