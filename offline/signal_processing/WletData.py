import numpy as np
import matplotlib.pyplot as plt

"""
WletData objects are meant to group together raw data series with their
respective Morlet wavelet-filtered data and power data. They contain
information on a single frequency band, whose bandwidth depends on the order
of the wavelet object used to create it.

As such, these objects have three main attributes:
    (1) a 'raw' dataset time series,
    (2) a 'filtered' time series, and
    (3) a 'power' data time series.
"""


class WletData:

    sRate = 250

    # Cropping on the edges of the data to eliminate edge artifacts. 2 seconds
    # seems to be good enough. Modifiable.
    crop = 2*sRate

    def __init__(self, freq, path):
        # The constructor takes two inputs: the frequency at which the data is
        # to be filtered, and the path of the data file.
        self.freq = freq
        self.raw = np.loadtxt(path,
                              delimiter=',',
                              skiprows=7,
                              usecols=1)
        w1 = Wavelet(freq)
        self.filtered, self.power = w1.filt(self.raw)
        self.tr = np.linspace(0, len(self.raw)/self.sRate, len(self.raw))
        self.tf = self.tr[self.crop:-self.crop]
        self.filtered = self.filtered[self.crop:-self.crop]
        self.power = self.power[self.crop:-self.crop]

    def plots(self):
        # Plots the raw data, the filtered data and the power data.
        plt.subplot(311)
        plt.plot(self.tr, self.raw)
        plt.grid(True)
        plt.title('Raw Data')
        plt.ylabel('Voltage ' + r'($\mu$V)')
        plt.subplot(312)
        plt.plot(self.tf, self.filtered)
        plt.title('Data Filtered at ' + str(self.freq)+' Hz')
        plt.ylabel('Voltage ' + r'($\mu$V)')
        plt.grid(True)
        plt.subplot(313)
        plt.plot(self.tf, self.power)
        plt.title('Power at ' + str(self.freq)+' Hz')
        plt.ylabel('Power ' + r'($\mu$V $^2$)')
        plt.xlabel('Time (sec)')
        plt.grid(True)

# To run the example, use a path to a data file that the constructor can use.
# For now, it only uses data from the first channel in that data file.
w = WletData(14, r'D:\Santiago\University\NeurotechX\6_SUCCESS_Rest_RightClench_JawClench_ImagineClench_10secs.txt')
w.plots()
