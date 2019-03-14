import numpy as np
import math
import cmath
import matplotlib.pyplot as plt
import matplotlib.animation as ani
from mpl_toolkits.mplot3d import axes3d as plt3

"""
Morlet wavelets are finite sections of a complex sinusoidal wave modulated
by a Gaussian function. As such, they have two main characteristics: (1) the
frequency they represent (which is dictated by the sinusoid) and (2) the amount
of sinusoidal wave cycles they contain (which is dictated by the Gaussian's
standard deviation).

Therefore, Wavelet objects have the following attributes:
    (1) a static sampling rate 'sRate' determined by the EEG hardware (i.e.
    fixed to 250 Hz for this project),
    (2) a static time range 'tRange' which determines the amount of sample
    points used to represent the wavelet in a discrete series,
    (3) an 'order' which represents the order of the wavelet, and
    (4) a 'wavelet' which contains the discrete time series of the wavelet.
"""


class Wavelet:

    # Constant sRate unless the recording settings are modified.
    sRate = 250

    # Modifiable tRange. -0.5 to 0.5 seconds is perfect for mid-range
    # frequencies within the mu band and higher. In live recordings, this value
    # would account for part the lag between the acquisition and the processing
    # of a signal feature.
    tRange = 0.5

    nyq = math.floor(sRate/2)

    # I set the order to the most optimal values to maintain a fairly good
    # frequency resolution over the 0-125Hz frequency range. Lower frequencies
    # require small orders (i.e. about 3), while higher frequencies require
    # greater orders (i.e. 9 to 12) to maintain a reasonable time-frequency
    # resolution balance.
    o = np.flip(-1*(np.floor(np.logspace(np.log10(1), np.log10(11), nyq))-13))
    # Note that 'o' is an array of numbers. If using only one Wavelet, you can
    # manually set the order to an arbitrary fixed value in the constructor
    # below.

    def __init__(self, freq):
        # Modifiable order value.
        # IMPORTANT: when changing the order, please validate your wavelets
        # with the function wavelet_plots() to see if it holds the intended
        # spectral content.
        order = self.o[freq]

        self.freq = freq
        self.t = np.arange(-self.tRange, self.tRange, 1/self.sRate)

        # 's' is the standard deviation of the Gaussian, which dictates how
        # many sinusoid cycles will be contained in the wavelet.
        s = order/(2*math.pi*freq)

        # Mathematical definition of complex sine wave and a Gaussian modulator
        sineWave = cmath.e**(1j*freq*2*math.pi*self.t)
        gauss = cmath.e**((-self.t**2)/(2*s**2))

        # Multiplication per term, not matrix multiplication.
        self.wavelet = sineWave*gauss

    def filt(self, data):
        # This method convolves a dataset with an instance
        fData = np.convolve(self.wavelet, data, mode='same')
        power = fData*np.conj(fData)
        return fData, power

    def wavelet_plots(self):
        fig = plt.figure()
        frex = np.linspace(0, self.nyq, np.floor(len(self.t)/2)+1)
        grid = plt.GridSpec(3, 3)
        plt.subplot(grid[0, :])
        plt.grid(True)
        plt.plot(self.t, self.wavelet.real, 'r',
                 self.t, self.wavelet.imag, 'b')
        plt.legend(('real', 'imag'))
        plt.xlabel('Time')
        plt.ylabel('Weights')
        plt.title(str(self.freq)+' Hz Wavelet')

        ax = fig.add_subplot(grid[1:, :2], projection='3d')
        ax.plot(self.t, self.wavelet.real, self.wavelet.imag)
        ax.set_ylabel('Real Weights')
        ax.set_zlabel('Imaginary Weights')
        ax.set_xlabel('Time')
        plt.title(str(self.freq)+' Hz Wavelet')

        # FFT for spectral content.
        x = np.fft.rfft(self.wavelet.real)
        y = abs(x)
        y = y/max(y)
        ax = plt.subplot(grid[1:, 2])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlim([0, 125])
        plt.plot(frex, y)
        plt.plot([self.freq, self.freq], [-0.01, 1.01], 'k:')
        plt.ylabel('Normalized Power')
        plt.xlabel('Frequency (Hz)')
        plt.title(str(self.freq)+' Hz Wavelet: Spectral Content')
        x = np.arange(0, self.nyq, 20)
        if (self.freq % 20 != 0):
            x = np.append(x, self.freq)
            x.sort()
        plt.xticks(x)
        plt.grid(True)

    # Irrelevant method when using only one or a few Wavelets
    def wavelet_plots2(self):
        frex = np.linspace(0, self.nyq, np.floor(len(self.t)/2)+1)
        if plt.fignum_exists(1) == False:
            plt.figure()
        else:
            plt.gcf()
        grid = plt.GridSpec(3, 3)
        plt.subplot(grid[0, :])
        plt.grid(True)
        plt.plot(self.t, self.wavelet.real, 'r',
                 self.t, self.wavelet.imag, 'b')
        plt.legend(('real', 'imag'))
        plt.xlabel('Time')
        plt.ylabel('Weights')
        plt.title(str(self.freq)+' Hz Wavelet')

        x = np.fft.rfft(self.wavelet.real)
        y = abs(x)
        y = y/max(y)
        ax = plt.subplot(grid[1:, 0:])
        ax.set_ylim([-0.01, 1.01])
        ax.set_xlim([0, 125])
        plt.plot(frex, y)
        plt.plot([self.freq, self.freq], [-0.01, 1.01], 'k:')
        plt.ylabel('Normalized Power')
        plt.xlabel('Frequency (Hz)')
        plt.title(str(self.freq)+' Hz Wavelet: Spectral Content')
        x = np.arange(0, self.nyq, 20)
        if (self.freq % 20 != 0):
            x = np.append(x, self.freq)
            x.sort()
        plt.xticks(x)
        plt.grid(True)

    # Irrelevant method when using only one or a few Wavelets
    def wavelet_ani(freq):
        w1 = Wavelet(freq)
        w1.wavelet_plots2()


# fig = plt.figure(1)
# anim = ani.FuncAnimation(fig, Wavelet.wavelet_ani, frames=125,
#                         interval=20)
w1 = Wavelet(14)  # Uncomment this and next line for example plots
w1.wavelet_plots()
