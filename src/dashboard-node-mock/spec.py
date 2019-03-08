import socketio
import time
import struct
import numpy as np
import matplotlib.animation as animation
import matplotlib.pyplot as plt

sio = socketio.Client()

fig = plt.figure()
plt.yscale('log')
arr = []
specgram = []

@sio.on('fft')
def fft(data):
    fig.clear()
    time = data['time']
    data = data['eeg']['data'][0] # 0th chanel
    mean = np.mean(data[6:13])
    arr.append(mean)
    if len(arr) > 50:
        arr.pop(0)

    plt.subplot(311)
    plt.ylim(0.1,15)
    plt.bar(0,mean)

    plt.subplot(312)
    plt.ylim(0.1,15)
    plt.plot(arr)

    PSD = np.log10(np.abs(data[:60]) + 1)
    specgram.append(PSD)
    if len(specgram) > 50:
        plt.subplot(313)
        plt.pcolor([i for i in range(len(specgram))],[i for i in range(len(specgram[0]))], np.array(specgram).T)
        specgram.pop(0)
    plt.show()


sio.connect('http://localhost:3000')
sio.wait()
