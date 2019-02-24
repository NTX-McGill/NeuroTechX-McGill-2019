from __future__ import print_function
import sys

sys.path.append('..')  # help python find cyton.py relative to scripts folder
from openbci import cyton as bci
import logging
import time


def printData(sample):
    # os.system('clear')
    print("----------------")
    print("%f" % (sample.id))
    print(sample.channel_data)
    # acceloremeter
    # print(sample.aux_data)
    print("----------------")


if __name__ == '__main__':
    port = '/dev/tty.usbserial-DM00QA1Z'
    baud = 115200
    logging.basicConfig(filename="test.log", format='%(asctime)s - %(levelname)s : %(message)s', level=logging.DEBUG)
    logging.info('---------LOG START-------------')
    board = bci.OpenBCICyton(port=port, scaled_output=False, log=True)
    print("Board Instantiated")
    board.ser.write(str.encode('jess test'))
    time.sleep(1)
    board.start_streaming(printData)
    board.print_bytes_in()
