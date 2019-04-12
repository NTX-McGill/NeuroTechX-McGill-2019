"""
Autonomous mode of the wheelchair.
"""

import math
import numpy as np
import socketio
import time
from scipy import stats
sio = socketio.Client()

"""For safety part"""
# assume we get real-time data on sensor info: assume we get
BUFFER = 0.1   # to allow for error/stopping before we actually hit something

SIDE_COLLISION_THRESHOLD = 0.3
COLLISION_THRESHOLD = 0.5 # need 1 m to turn left/right afterwards
COLLISION_THRESHOLD_STAIR = 0.5
SENSOR_HEIGHT = .2  # final height = 20 cm
SENSOR_ANGLE = math.radians(25)    # ALSO CHANGE THIS
RAMP_ANGLE = math.radians(7.5)
SLOW_THRESHOLD = SENSOR_HEIGHT/math.cos(SENSOR_ANGLE) + BUFFER # slow down if we may be approaching a ramp

mathvalue = (SENSOR_HEIGHT*math.tan(SENSOR_ANGLE) - COLLISION_THRESHOLD_STAIR)*math.sin(RAMP_ANGLE) # see diagram
STOP_THRESHOLD = mathvalue/math.cos(RAMP_ANGLE + SENSOR_ANGLE) + BUFFER + SLOW_THRESHOLD # stop if not a ramp
STOP_THRESHOLD = 0.7

"""For wall following"""
SAMPLES_PER_SECOND = 5
MIN_SECONDS = 1
MIN_SAMPLES = MIN_SECONDS*SAMPLES_PER_SECOND
distances = {"left": [], "right": []} # past 3 seconds of distance data
MAX_2ND_ORDER_DERIVATIVE = 0.1
MIN_DISTANCE_TO_WALL = 0.6 # meters
MAX_DISTANCE_TO_WALL = 2.5 # meters
MIN_CHANGE_DISTANCE = 0.3
MAX_STD_ERR = 0.3
MIN_CHANGE_RATIO = 1.4

previous_wall = ""
comparison_time = time.time()

"""For doorway traversal"""
# --- nada for now

"""For obstacle avoidance"""


def collision_detection(left_forw, right_forw, front_forw, front_tilt): # assume these are distances
    """
    Returns OK (1) or NOT OK (0) for each direction
    """
    l_out, r_out, f_out = 1,1,1
    if left_forw <= COLLISION_THRESHOLD or front_forw <= COLLISION_THRESHOLD/2:
        l_out = 0
    if right_forw <= COLLISION_THRESHOLD or front_forw <= COLLISION_THRESHOLD/2:
        r_out = 0
    if (front_forw <= COLLISION_THRESHOLD or left_forw <= SIDE_COLLISION_THRESHOLD
        or right_forw  SIDE_COLLISION_THRESHOLD):
        f_out = 0
    f_out = min(stair_detection(front_tilt), f_out)
    return (l_out, r_out, f_out)

def stair_detection(front_tilt):
    """
    Returns 0 if there are stairs
    """
    return 1
    # if front_tilt >= STOP_THRESHOLD:
    #     print(66)
    #     return 0 #
    # return 1

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
    history = np.array(history)
    relevant_history = history[-min(2*MIN_SAMPLES, len(history)):]
    x = np.arange(len(relevant_history))
    print(relevant_history[relevant_history == 1000])
    if len(relevant_history[relevant_history == 1000])/len(relevant_history) <= 0.4:
        bad_indeces = np.where(relevant_history==1000)
        relevant_history = np.delete(relevant_history, bad_indeces)
        x = np.delete(x, bad_indeces)
    else:
        return 0
    print(relevant_history)
    slope, _, r_value, p_value, std_err = stats.linregress(x, relevant_history)

    deriv = deriv_2(relevant_history) # average 2nd order derivative


    if (std_err < MAX_STD_ERR or deriv < MAX_2ND_ORDER_DERIVATIVE) and history[-1] < MAX_DISTANCE_TO_WALL:
        desired_distance = max(np.array2str(np.median(history)), MIN_DISTANCE_TO_WALL) # not average cuz 1000 m
        return desired_distance

    return 0

def wall_adjustment(direction, desired_distance, current_distance,
                     l_out, r_out, f_out):
    print('desired distance: {}, real distance: {}'.format(desired_distance,
                                                np.array2str(current_distance)))
    print("WALL: THERE IS A WALL ON THE {}".format(direction))
    if direction == "LEFT":
        if current_distance/desired_distance > MIN_CHANGE_RATIO and l_out:
            print("WALL: MAKING LEFT ADJUSTMENT")
            return 'L'
        elif desired_distance/current_distance > MIN_CHANGE_RATIO and r_out:
            print("WALL: MAKING RIGHT ADJUSTMENT")
            return 'R'
        elif current_distance < MIN_DISTANCE_TO_WALL and r_out:
            print('WALL: TOO CLOSE TO WALL. MAKING RIGHT ADJUSTMENT')
            return 'R'
        # if distances["left"][-1] + MIN_CHANGE_DISTANCE < desired_distance and r_out:
        #     print("WALL: MAKING RIGHT ADJUSTMENT")
        #     return 'R'
        # elif distances["left"][-1] > desired_distance + MIN_CHANGE_DISTANCE  and l_out:
        #     print("WALL: MAKING LEFT ADJUSTMENT")
        #     return 'L'
    elif direction == "RIGHT":
        if distances["right"][-1]/desired_distance > MIN_CHANGE_RATIO and r_out:
            print("WALL: MAKING RIGHT ADJUSTMENT")
            return 'R'
        elif desired_distance/distances["right"][-1] > MIN_CHANGE_RATIO  and l_out:
            print("WALL: MAKING LEFT ADJUSTMENT")
            return 'L'
        elif distances["right"][-1] < MIN_DISTANCE_TO_WALL and l_out:
            print('WALL: TOO CLOSE TO WALL. MAKING LEFT ADJUSTMENT')
            return 'L'
    return ''



def wall_follower(l_out, r_out, f_out):
    """
    A wall is something with ~0 2nd order derivative.
    If there are walls to the left and right, we follow the closer wall.
    TO DO: Try changing desired_distance to fixed amount?
    """
    global distances
    global previous_wall
    global comparison_time
    current_time = time.time()

    if not f_out:
        return 'S'

    if current_time - comparison_time > 1: # check every 1 second
        comparison_time = current_time

        left_desired_distance = is_wall(distances["left"])
        right_desired_distance = is_wall(distances["right"])

        if previous_wall == "left" and left_desired_distance:
            return wall_adjustment("LEFT", left_desired_distance, distances["left"][-1],
                                   l_out, r_out, f_out)
        elif previous_wall == "right" and right_desired_distance:
            return wall_adjustment("RIGHT", right_desired_distance, distances["right"][-1],
                                   l_out, r_out, f_out)
        elif distances["left"][-1] < distances["right"][-1] and left_desired_distance:
            return wall_adjustment("LEFT", left_desired_distance, distances["left"][-1],
                                   l_out, r_out, f_out)
        elif distances["right"][-1] >= distances["left"][-1] and right_desired_distance:
            return wall_adjustment("RIGHT", right_desired_distance, distances["right"][-1],
                                   l_out, r_out, f_out)
        elif left_desired_distance:
            return wall_adjustment("LEFT", left_desired_distance, distances["left"][-1],
                                   l_out, r_out, f_out)
        elif right_desired_distance:
            return wall_adjustment("RIGHT", right_desired_distance, distances["right"][-1],
                                   l_out, r_out, f_out)
    return 'F'

def obstacle_avoider():
    return None

def clear_distances():
    global distances
    global previous_wall
    distances = {"left": [], "right": []}
    previous_wall = ""

@sio.on('to self-driving (clear)')
def on_clear(data):
    clear_distances()
    command = 'F'
    sio.emit('from self-driving (forward)', {'response': command,
                                             'duration': 500,
                                             'state': 'forward'})

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
            print(command)
            if command == '':
                if f_out:
                    command = 'F'
                else:
                    command = 'S'
            sio.emit('from self-driving (forward)', {'response': command,
                                                     'duration': 500,
                                                     'state': 'forward'})
    else:
        clear_distances()

    # print(command)
    if f_out:
        command = "F"
    else:
        command = "S"

    sio.emit('from self-driving', {'response': command,
                                   'duration': 500})
sio.connect('http://localhost:3000')
sio.wait()
