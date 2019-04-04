"""
Autonomous mode of the wheelchair.
"""

import math
import numpy as np
import socketio

sio = socketio.Client()

"""For safety part"""
# assume we get real-time data on sensor info: assume we get
BUFFER = 0.1   # to allow for error/stopping before we actually hit something

COLLISION_THRESHOLD = 1.0 + BUFFER # need 1 m to turn left/right afterwards
COLLISION_THRESHOLD_STAIR = 0.5 + BUFFER
SENSOR_HEIGHT = .2  # final height = 20 cm
SENSOR_ANGLE = math.radians(60)    # ALSO CHANGE THIS
RAMP_ANGLE = math.radians(7.5)
SLOW_THRESHOLD = SENSOR_HEIGHT/math.cos(SENSOR_ANGLE) + BUFFER # slow down if we may be approaching a ramp

mathvalue = (SENSOR_HEIGHT*math.tan(SENSOR_ANGLE) - COLLISION_THRESHOLD_STAIR)*math.sin(RAMP_ANGLE) # see diagram
STOP_THRESHOLD = mathvalue/math.cos(RAMP_ANGLE + SENSOR_ANGLE) + BUFFER + SLOW_THRESHOLD # stop if not a ramp
# STOP_THRESHOLD = 0.4

"""For wall following"""
SAMPLES_PER_SECOND = 5
MIN_SECONDS = 2
MIN_SAMPLES = MIN_SECONDS*SAMPLES_PER_SECOND
distances = {"left": [], "right": []} # past 3 seconds of distance data
MAX_2ND_ORDER_DERIVATIVE = 0.1
MIN_DISTANCE_TO_WALL = 0.5 # meters
MAX_DISTANCE_TO_WALL = 2.5 # meters
MIN_CHANGE_DISTANCE = 0.2

previously_following_wall = True

"""For doorway traversal"""
# --- nada for now

"""For obstacle avoidance"""


def collision_detection(left_forw, right_forw, front_forw, front_tilt): # assume these are distances
    """
    Returns OK (1) or NOT OK (0) for each direction
    """
    l_out, r_out, f_out = 1,1,1
    if left_forw <= COLLISION_THRESHOLD or front_forw <= COLLISION_THRESHOLD/4:
        l_out = 0
    if right_forw <= COLLISION_THRESHOLD or front_forw <= COLLISION_THRESHOLD/4:
        r_out = 0
    if front_forw <= COLLISION_THRESHOLD:
        f_out = 0
    f_out = min(stair_detection(front_tilt), f_out)
    return (l_out, r_out, f_out)

def stair_detection(front_tilt):
    """
    Returns 0 if there are stairs
    """
    if front_tilt >= STOP_THRESHOLD:
        return 0 #
    return 1

def append_distances(left, right):
    global distances
    distances["left"].append(left)
    distances["right"].append(right)

def deriv_2(history):
    """
    Average 2nd order derivative
    """
    return np.abs(np.mean(np.diff(history, n=2)))

def is_wall(history):
    """
    Checks if 2nd order derivative is ~= 0.
    Returns desired_distance (0 if there is a wall)
    """
    if len(history) < MIN_SAMPLES:
        print("Not enough samples to judge")
        return 0
    relevant_history = history[-min(2*MIN_SAMPLES, len(history)):]
    deriv = deriv_2(relevant_history) # average 2nd order derivative

    if deriv < MAX_2ND_ORDER_DERIVATIVE and history[-1] < MAX_DISTANCE_TO_WALL:
        desired_distance = min(np.mean(distances["right"]), MIN_DISTANCE_TO_WALL)
        return desired_distance

    return 0

def wall_follower(l_out, r_out, f_out):
    """
    A wall is something with ~0 2nd order derivative.
    If there are walls to the left and right, we follow the closer wall.
    TO DO: Try changing desired_distance to fixed amount?
    """
    global distances

    if distances["left"][-1] < distances["right"][-1]:
        desired_distance = is_wall(distances["left"])
        if desired_distance:
            print("WALL: THERE IS A WALL ON THE LEFT")
            # left stuff
            if distances["left"][-1] < desired_distance + MIN_CHANGE_DISTANCE and l_out:
                print("WALL: MAKING LEFT ADJUSTMENT")
                return 'L'
            elif distances["left"][-1] + MIN_CHANGE_DISTANCE > desired_distance and r_out:
                print("WALL: MAKING RIGHT ADJUSTMENT")
                return 'R'
            elif f_out:
                return 'F'
            else:
                return 'S'

    desired_distance = is_wall(distances["right"])
    if desired_distance:
        print("WALL: THERE IS A WALL ON THE RIGHT")
        # right stuff
        if distances["right"][-1] < desired_distance + MIN_CHANGE_DISTANCE and l_out:
            print("WALL: MAKING RIGHT ADJUSTMENT")
            return 'R'
        elif distances["right"][-1] + MIN_CHANGE_DISTANCE > desired_distance and r_out:
            print("WALL: MAKING LEFT ADJUSTMENT")
            return 'L'
        elif f_out:
            return 'F'
        else:
            return 'S'  # can be changed to None with obstacle avoider
    if f_out:
        return 'F'
    return 'S'          # can be changed to None with obstacle avoider

def obstacle_avoider():
    return None


@sio.on('to self-driving')
def on_message(sensor_data):
    l = sensor_data['left']
    r = sensor_data['right']
    f = sensor_data['front']
    f_t = sensor_data['front-tilt']

    l_out, r_out, f_out = collision_detection(l, r, f, f_t)
    sio.emit('from self-driving (safety)', {'left': l_out,
                                            'right': r_out,
                                            'forward': f_out})

    if sensor_data['state'] == 'forward':
        append_distances(l, r)
        command = obstacle_avoider()
        if not command:
            command = wall_follower(l_out, r_out, f_out)
            sio.emit('from self-driving (forward)', {'response': command,
                                                     'duration': 500,
                                                     'state': 'forward'})

    # print(command)
    if f_out:
        command = "F"
    else:
        command = "S"
        print(command)

    # if command == "CAN'T GO FORWARD AND NO WALL TO FOLLOW":

    sio.emit('from self-driving', {'response': command,
                                   'duration': 500})
sio.connect('http://localhost:3000')
sio.wait()
