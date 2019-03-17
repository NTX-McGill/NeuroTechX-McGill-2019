import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D as plt3
from wavelet import Wavelet
<<<<<<< HEAD
=======

>>>>>>> b31bac4ab29b602f238bd1d9512e776bb3a387f5
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

    def __init__(self, chList, freq, path):
        # The constructor takes three inputs: the channels of interest, the
        # frequency at which the data is to be filtered, and the path of the
        # data file.
        self.freq = freq
        self.raw = np.loadtxt(path,
                              delimiter=',',
                              skiprows=7,
                              usecols=chList)
        w1 = Wavelet(freq)
        self.filtered, self.power = w1.filt(self.raw)
        self.tr = np.linspace(0, len(self.raw)/self.sRate, len(self.raw))
        self.tf = self.tr[self.crop:-self.crop]
        self.filtered = self.filtered[self.crop:-self.crop]
        self.power = self.power[self.crop:-self.crop]

    def feat_mat(self, length, stride):
        # Adjust nFeat depending on how many values will be appended to the end
        # of each row (e.g. if interested in the max, mean and time derivative
        # of each window, set nFeat to 3 and make sure these values are defined
        # in the for loop below).
        nWin = int(np.floor(((len(self.power)-length)/stride)+1))
        data = self.power
        nFeat = 2
        fMat = np.ma.zeros((nWin, length+nFeat+1))
        # Manually set label ranges
        boundaries = np.array([20, 101, 205, 303, 408, 504, 603, 705, 802, 903,
                               1007, 1105, 1206, 1306, 1406, 1507, 1606, 1705,
                               1805, 1000000000000000])
        boundaries = boundaries*self.sRate/10
        boundaries = boundaries-self.crop
        
        counter = 0
        for i in range(nWin):
            start = i*stride
            end = start+length
            win = data[start:end]
            mean = np.mean(win)
            mx = np.argmax(win)
            if start >= boundaries[counter]:
                if (counter % 2) == 0:
                    tag = 0
                else:
                    tag = 1
                counter += 1
            else:
                if (counter % 2) == 0:
                    tag = 1
                else:
                    tag = 0
            arr = np.append(win, [mean, win[mx], tag])
            fMat[i, :] = arr
        return fMat

    def example_plots(self):
        # Plots the raw data, the filtered data and the power data.
        ch = 1
        raw = self.raw[:, ch-1]
        filtered = self.filtered[:, ch-1]
        power = self.power[:, ch-1]
        ax = plt.subplot(311)
        plt.plot(self.tr, raw)
        plt.grid(True)
        plt.title('Raw Data')
        plt.ylabel('Voltage ' + r'($\mu$V)')
        ax = plt.subplot(312)
        plt.plot(self.tf, filtered)
        plt.title('Data Filtered at ' + str(self.freq)+' Hz')
        plt.ylabel('Voltage ' + r'($\mu$V)')
        plt.grid(True)
        ax = plt.subplot(313)
        plt.plot(self.tf, power)
        plt.title('Power at ' + str(self.freq)+' Hz')
        plt.ylabel('Power ' + r'($\mu$V $^2$)')
        plt.xlabel('Time (sec)')
        plt.grid(True)
        fig = plt.figure()
        ax = fig.gca(projection='3d')
        ax.plot(self.tf, filtered.real, filtered.imag)
        ax.set_ylabel('Real')
        ax.set_zlabel('Imaginary')
        ax.set_xlabel('Time (sec)')
        plt.title('Analytical Signal at ' + str(self.freq) + ' Hz')

    def all_power_plots(self):
        data = self.power
        plt.figure()
        try:
            nCh = np.ma.size(data, 1)
            for i in range(nCh):
                plt.subplot(nCh, 1, i+1)
                plt.plot(self.tf, data[:, i])
                plt.grid(True)
            if (i == 0):
                plt.title('Power at ' + str(self.freq)+' Hz')
                plt.ylabel('Ch ' + str(i+1) + 'Power ' + r'($\mu$V $^2$)')
            if (i == nCh-1):
                plt.xlabel('Time (sec)')
        except (ValueError, IndexError) as e:
            nCh = 1
            plt.plot(self.tf, data)
            plt.grid(True)
            plt.title('Power at ' + str(self.freq)+' Hz')
            plt.ylabel('Ch ' + str(1) + 'Power ' + r'($\mu$V $^2$)')
            plt.xlabel('Time (sec)')

# To run the example, use a path to a data file that the constructor can use.
# For now, it only uses data from the first channel in that data file.
try:
    del(w)
except NameError:
    pass
<<<<<<< HEAD
#w = WletData(1, 10, r'D:\Santiago\University\NeurotechX\6_SUCCESS_Rest_RightClench_JawClench_ImagineClench_10secs.txt')
w = WletData(1, 12, '../data/March 4/6_SUCCESS_Rest_RightClench_JawClench_ImagineClench_10secs.txt')
w.all_power_plots()
plt.show()
=======

path='/Users/jenisha/Desktop/NeuroTechX-McGill-2019/offline/data/March_11/'
fname= path +  '1_JawRest_JawRightClench_10s.txt'
w = WletData([1,8], 10, fname)
w.all_power_plots()



        
        

>>>>>>> b31bac4ab29b602f238bd1d9512e776bb3a387f5
# z is the feature matrix! It is saved as a masked array, to import it use
# np.load('feature_matrix')
z = w.feat_mat(125, 62)
z.dump('feature_matrix2')
