"""
Grabs OpenBCI GUI data and saves it to CSV
Listens on localhost, port 12345
Listens at addresses openbci-fft, openbci-time
"""

from pythonosc import dispatcher
from pythonosc import osc_server

import csv
import datetime
import time

def write_time_data(*args):
    test_number = args[1][0]
    time_data = list(args[2:])
    time = datetime.datetime.now()
    with open('settings.txt', 'r') as f:
        settings = f.read()
    print(settings)
    if settings!= 'wait':
        with open('../data/sample-tests/time_test{}.csv'.format(test_number), 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([settings] + [time] + time_data)

def write_fft_data(*args):
    test_number = args[1][0]
    channel = int(args[2])
    frequency_data = list(args[3:])
    time = datetime.datetime.now()
    with open('settings.txt', 'r') as f:
        settings = f.read()
    print(settings)
    if settings!= 'wait':
        with open('../data/sample-tests/fft_test{}_channel{}.csv'.format(test_number, channel), 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([settings] + [time] + frequency_data)

dispatcher = dispatcher.Dispatcher()

def get_data(test_number=0):
    """
    Main function in program
    """
    global dispatcher
    dispatcher.map("/openbci-fft", write_fft_data, test_number)
    dispatcher.map("/openbci-time", write_time_data, test_number)

    ip = "127.0.0.1"
    port = 12345

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
get_data()
