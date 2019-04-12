import serial
import sys
import time
ser = serial.Serial('COM5',baudrate = 9600, timeout = 1)
WAITBEFOREREADING = 0.1 #Waits this number of second before reading inputs

#define the FSM states
#Index is the type of motion wanted. 0 = forward, 1 = stop/slow, 2 = left, 3 = right
states =	{
  "0": "F",
  "1": "S",
  "2": "L",
  "3": "R",
  "4": "D"
}

def main(argv):
    #main loop.
    while True:
        instruction = input() #Input to send to arduino
        try:
            ser.write(states[instruction].encode('utf-8')) #send instruction
        except:
            print("Not a recognized command")
            continue

        #Display any information sent back.
        time.sleep(WAITBEFOREREADING)
        while ser.in_waiting:
            print(int.from_bytes(ser.read(),byteorder='little'))
        print()


if __name__ == "__main__":
    main(sys.argv)
