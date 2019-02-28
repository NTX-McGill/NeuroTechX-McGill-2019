from __future__ import print_function
import sys

sys.path.append('..')  # help python find cyton.py relative to scripts folder
from openbci import cyton as bci
import logging
import time
import csv

def write_data(sample):
    # os.system('clear')
    to_print = True

    if to_print:
        print("%f\t" % (sample.id))
        print("{}\t".format(sample.channel_data))
        print("{}\t".format(sample.aux_data))
        print("\n")
    else:
        with open('data.csv', 'a') as f:
            writer = csv.writer(f, delimiter='\t')
            row = [int(sample.id)] + sample.channel_data + sample.aux_data
            writer.writerow(row)

if __name__ == '__main__':
    # port = '/dev/tty.OpenBCI-DN008VTF'
    port = '/dev/tty.usbserial-DM00QA1Z'
    # port = '/dev/tty.OpenBCI-DN0096XA'
    baud = 115200
    logging.basicConfig(filename="test.log", format='%(asctime)s - %(levelname)s : %(message)s', level=logging.DEBUG)
    logging.info('---------LOG START-------------')
    board = bci.OpenBCICyton(port=port, scaled_output=False, log=False)
    print("Board Instantiated")
    board.ser.write(b'v')
    time.sleep(10)
    board.start_streaming(write_data)
