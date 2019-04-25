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
COLLISION_THRESHOLD = 0.8 # need 1 m to turn left/right afterwards
# COLLISION_THRESHOLD_STAIR = 0.5
# SENSOR_HEIGHT = .2  # final height = 20 cm
# SENSOR_ANGLE = math.radians(25)    # ALSO CHANGE THIS
# RAMP_ANGLE = math.radians(7.5)
# SLOW_THRESHOLD = SENSOR_HEIGHT/math.cos(SENSOR_ANGLE) + BUFFER # slow down if we may be approaching a ramp

# mathvalue = (SENSOR_HEIGHT*math.tan(SENSOR_ANGLE) - COLLISION_THRESHOLD_STAIR)*math.sin(RAMP_ANGLE) # see diagram
# STOP_THRESHOLD = mathvalue/math.cos(RAMP_ANGLE + SENSOR_ANGLE) + BUFFER + SLOW_THRESHOLD # stop if not a ramp
# STOP_THRESHOLD = 0.7

"""For wall following"""
SAMPLES_PER_SECOND = 2
MIN_SECONDS = 2
MIN_SAMPLES = MIN_SECONDS*SAMPLES_PER_SECOND
MAX_SAMPLES = MIN_SAMPLES*5
distances = {"left": [], "right": [], "front": []} # past 3 seconds of distance data
MAX_2ND_ORDER_DERIVATIVE = 0.1
MIN_DISTANCE_TO_WALL = 1 # meters
MAX_DISTANCE_TO_WALL = 2.5 # meters
MIN_CHANGE_DISTANCE = 0.2
MAX_STD_ERR = 0.2
MIN_CHANGE_RATIO = 1.3
MAX_OUTLIERS = 0.4

previous_wall = ""
comparison_time = time.time()

"""For doorway traversal"""
# --- nada for now

"""For obstacle avoidance"""
# mode = "no obstacle"
# MIN_OBSTACLE_FRONT = 0.8
# MIN_OBSTACLE_SIDE = 1.2 # 0.8*cos(45º)
# turning_start_time = 0
# turning_time = 0
# MAX_TURNING_TIME = 2*1000 # miliseconds

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
        or right_forw <= SIDE_COLLISION_THRESHOLD):
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

def append_distances(left, right, front):
    global distances
    distances["left"].append(left)
    distances["right"].append(right)
    distances["front"].append(front)

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
    relevant_history = history[-min(MAX_SAMPLES, len(history)):]
    relevant_history = relevant_history[:int(2*len(relevant_history)/3)]
    x = np.arange(len(relevant_history))
    print(relevant_history[relevant_history == 1000])
    if len(relevant_history[relevant_history == 1000])/len(relevant_history) <= MAX_OUTLIERS:
        bad_indeces = np.where(relevant_history==1000)
        relevant_history = np.delete(relevant_history, bad_indeces)
        x = np.delete(x, bad_indeces)
    else:
        return 0
    print(relevant_history)
    slope, _, r_value, p_value, std_err = stats.linregress(x, relevant_history)

    deriv = deriv_2(relevant_history) # average 2nd order derivative


    if (std_err < MAX_STD_ERR or deriv < MAX_2ND_ORDER_DERIVATIVE):
        desired_distance = min(np.array([1.7]),max(np.median(history), np.array([MIN_DISTANCE_TO_WALL]))) # not average cuz 1000 m
        return desired_distance

    return 0

def wall_adjustment(direction, desired_distance, current_distance,
                     l_out, r_out, f_out):
    """
    Fine adjustments to follow wall (if a wall exists)
    """
    print('desired distance: {}, real distance: {}'.format(np.array2string(desired_distance),
                                                           current_distance))
    print("WALL: THERE IS A WALL ON THE {}".format(direction))

    if direction == "LEFT":
        # if current_distance/desired_distance > MIN_CHANGE_RATIO and l_out:
        #     print("WALL: MAKING LEFT ADJUSTMENT")
        #     return 'L'
        # elif desired_distance/current_distance > MIN_CHANGE_RATIO and r_out:
        #     print("WALL: MAKING RIGHT ADJUSTMENT")
        #     return 'R'
        if current_distance + MIN_CHANGE_DISTANCE < desired_distance and r_out:
            print("WALL: MAKING RIGHT ADJUSTMENT")
            return 'R'
        elif current_distance > desired_distance + MIN_CHANGE_DISTANCE  and l_out:
            print("WALL: MAKING LEFT ADJUSTMENT")
            return 'L'
        elif current_distance < MIN_DISTANCE_TO_WALL and r_out:
            print('WALL: TOO CLOSE TO WALL. MAKING RIGHT ADJUSTMENT')
            return 'R'
    elif direction == "RIGHT":
        if current_distance + MIN_CHANGE_DISTANCE < desired_distance and l_out:
            print("WALL: MAKING LEFT ADJUSTMENT")
            return 'L'
        elif current_distance > desired_distance + MIN_CHANGE_DISTANCE  and r_out:
            print("WALL: MAKING RIGHT ADJUSTMENT")
            return 'R'
        # if distances["right"][-1]/desired_distance > MIN_CHANGE_RATIO and r_out:
        #     print("WALL: MAKING RIGHT ADJUSTMENT")
        #     return 'R'
        # elif desired_distance/distances["right"][-1] > MIN_CHANGE_RATIO  and l_out:
        #     print("WALL: MAKING LEFT ADJUSTMENT")
        #     return 'L'
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

    if current_time - comparison_time > 0: # check every 1 second
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
    """
    Avoids small obstacles.
    An obstacle is "small" if the left/right sensors are clear but the front
    is not.
    5 modes: "no obstacle", "avoiding-right", "avoiding-left", "re-adjusting-left", "re-adjusting-right"
    """
    return None
    # global distances
    # global mode
    # global turning_start_time
    # global turning_record_time
    # global turning_time
    # if mode == "no obstacle":
    #     if distances["front"][-1] < MIN_OBSTACLE_FRONT: # ahh there's an obstacle!
    #         if distances["right"][-1] > distances["left"][-1] and distances["right"][-1] > MIN_OBSTACLE_SIDE:
    #             mode = "avoiding-right"
    #             turning_start_time = time.time()
    #         elif distances["left"][-1] >= distances["right"][-1] and distances["left"][-1] > MIN_OBSTACLE_SIDE:
    #             mode = "avoiding-left"
    #             turning_start_time = time.time()
    #     else:
    #         return None
    # if mode == "avoiding-left":
    #     if distances["front"][-1] <= MIN_OBSTACLE_FRONT:
    #         return 'L'
    #     else:
    #         mode = "re-adjusting-left"
    #         turning_time = time.time() - turning_start_time
    #         turning_start_time = time.time()
    # elif mode == "avoiding-right":
    #     if distances["front"][-1] <= MIN_OBSTACLE_FRONT:
    #         return 'R'
    #     else:
    #         mode = "re-adjusting-right"
    #         turning_time = time.time() - turning_start_time
    #         turning_start_time = time.time()
    # if mode == "passing-obstacle-left":
    #     continue
    # elif mode == "passing-obstacle-right":
    #     continue
    # if mode == "re-adjusting-left":
    #     if distances["right"][-1] >= 1.2 and turning:
    #         return 'R'
    # elif mode == "re-adjusting-right":
    #     if distances["left"][-1] >= 1.2:
    #         return 'L'

def clear_distances():
    global distances
    global previous_wall
    global mode
    distances = {"left": [], "right": [], "front": []}
    previous_wall = ""
    mode = "no obstacle"

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
    print("l: ",l_out, "r: ", r_out, "f: ", f_out)
    sio.emit('from self-driving (safety)', {'left': l_out,
                                            'right': r_out,
                                            'forward': f_out})
    # if sensor_data['state'] == 'forward':
    #     append_distances(l, r, f)
    #     command = obstacle_avoider()
    #     if not command:
    #         # command = wall_follower(l_out, r_out, f_out)
    #         print(command)
    #         # if command == '':
    #         if f_out:
    #             command = 'F'
    #         else:
    #             command = 'S'
    #     sio.emit('from self-driving (forward)', {'response': command,
    #                                              'duration': 1000,
    #                                              'state': 'forward'})
    # else:
    #     clear_distances()

sio.connect('http://localhost:3000')
sio.wait()
