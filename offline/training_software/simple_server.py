from pythonosc import dispatcher
from pythonosc import osc_server

import csv

def time_series(*args):
    print(args)

def fft(*args):
    channel = int(args[1])
    frequency_data = args[2:]
    alpha_mu = frequency_data[7:12] # mu waves at 8-12 Hz
    average = sum(alpha_mu)/len(alpha_mu)
    # averages[channel].append()
    # print(averages)
    batch = (channel-1) % 4 # split channels into 2
    memory_array.append(average)

    if i == 100: # save every 100
        # write to csv
        i = 0
        memory_average = sum(memory_array)/len(memory_array)
        memory_array = []
        with open('batch-{}.csv'.format(batch), 'a') as f:
            writer = csv.writer(f, delimiter=',')
            writer.writerow([str(memory_average)])
    i += 1

if __name__ == "__main__":
    averages = {i: [] for i in range(1,9)} # 8 channels
    i = 1
    memory_array = []

    ip = "127.0.0.1"
    port = 12345

    dispatcher = dispatcher.Dispatcher()
    # dispatcher.map("/openbci", default_handler)
    dispatcher.map("/openbcifft", fft)

    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Serving on {}".format(server.server_address))
    server.serve_forever()
