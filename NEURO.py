import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.mlab as mlab

from scipy.signal import butter, lfilter, iirfilter, welch


from scipy.fftpack import fft

import numpy.fft as fft1

"""
Implementation of the preprocessing as outlined in 
"TrueNorth-enabled Real-time Classification of EEG Data for Brain- TrueNorth-enabled 
Real-time Classification of EEG Data for Brain-"
and
"A Robust Low-Cost EEG Motor Imagery-Based Brain-Computer Interface:"
Input:
- path: where the data is stored
- name_channels: name of each channel
"""
f1 = open("C:\\Users\\Yuan\\Documents\\Clench.txt","w")  


class Kiral_Korek_Preprocessing():
    def __init__(self, path, name_channels=["C3"]):
        self.path = path
        self.sample_rate = 250 #Default sampling rate for OpenBCI
        self.name_channel = name_channels
        

    
    def load_data_BCI(self, list_channels=[1]):
        """
        
        Load the data from OpenBCI txt file
        Input:
            list_channel: lists of channels to use
        
        """
        # load raw data into numpy array
        self.raw_eeg_data = np.loadtxt(self.path, 
                                       delimiter=',',
                                       skiprows=7,
                                       usecols=1)

            
        
        
    def initial_preprocessing(self, bp_lowcut =1, bp_highcut =70, bp_order=2,
                          notch_freq_Hz  = [60.0, 120.0], notch_order =2):
       """
       Filters the data by applying
       - A zero-phase Butterworth bandpass was applied from 1 – 70 Hz. 
       - A 60-120 Hz notch filter to suppress line noise
      
       
       Input:
           - bp_ lowcut: lower cutoff frequency for bandpass filter
           - bp_highcut: higher cutoff frequency for bandpass filter
           - bp_order: order of bandpass filter
           - notch_freq_Hz: cutoff frequencies for notch fitltering
           - notch_order: order of notch filter
        
        """
       self.nyq = 0.5 * self.sample_rate #Nyquist frequency
       self.low = bp_lowcut / self.nyq
       self.high = bp_highcut / self.nyq
       
       #Bandpass filter
       b_bandpass, a_bandpass = butter(bp_order, [self.low, self.high], btype='band')
       self.bp_filtered_eeg_data = lfilter(b_bandpass, a_bandpass, self.raw_eeg_data)
       self.notch_filtered_eeg_data = self.bp_filtered_eeg_data
       self.low1  = notch_freq_Hz[0]
       self.high1 = notch_freq_Hz[1]
       self.low1  = self.low1/self.nyq
       self.high1 = self.high1/self.nyq
       b_notch, a_notch = iirfilter(2, [self.low1, self.high1], btype='bandstop')
       self.notch_filtered_eeg_data = lfilter(b_notch, a_notch, self.notch_filtered_eeg_data)
       
       
#       mult_std = 6 
#       self.list_std_channel, self.list_mean_channel  = [], []
#       self.corrected_eeg_data = self.notch_filtered_eeg_data
#       for channel in self.corrected_eeg_data.T:
#            self.list_std_channel.append(np.std(channel))
#            self.list_mean_channel.append(np.mean(channel))
#            for val in channel:
#                if val > self.list_mean_channel[-1] + self.list_std_channel[-1] *  mult_std :
#                    print(val)
#                    val = val -  mult_std * self.list_mean_channel[-1]
#                    print(val)
#                elif val <  self.list_mean_channel[-1] - self.list_std_channel[-1] *  mult_std :
#                    val = val +  mult_std * self.list_mean_channel[-1]
#            
          
           
            
    def epoch_data(self, data, mode, window_length = 6):
        """
        Separates the data into several windows
        
        Input:
            - data: data to seperate into windows
            - mode: whether the windows are of same length (mode 1) or different lengths (mode 2)
            - window_length: length of the window in s
            - overlap: overlap in the previous window
        
        """
        if mode == "1":
            i=500
        elif mode == "0":
            i=3000
        
        array_epochs = []
        self.window_size_hz = int(window_length * self.sample_rate)

        for j in range(8):
            array_epochs.append(data[i:i+self.window_size_hz ])
            i = i + 5000
        

            self.epoch = array_epochs
           

        return np.array(array_epochs)
    
    def epoch_and_remove_outlier(self,mult_std = 6):
        """
        For each channel, Separates the data into several windows
        and sample values exceeding ±mult_std, where mult_std is the standard deviation
           of any given voltage trace, are set to ±mult_std to rectify outliers in voltage.
           
        Input: 
            - mult_std:
        
        
        """
        self.corrected_epoched_eeg_data = []
        #need to use lists because different window sizes
        epoched = self.epoch_data(self.notch_filtered_eeg_data,"1")
        epoched_corrected = []
        self.a = []
        for epoch in epoched:
            epoch_mean = np.mean(epoch)
            epoch_std = np.std(epoch)
            self.a.append(epoch_std)
            epoched_corrected.append(np.array([epoch_mean + epoch_std * mult_std if i > epoch_mean + epoch_std * mult_std 
                                        else epoch_mean - epoch_std * mult_std if i < epoch_mean - epoch_std * mult_std 
                                        else i for i in epoch]))
        self.corrected_epoched_eeg_data = np.array(epoched_corrected)


        '''

        b_bandpas, a_bandpas = butter(2, [8/self.nyq, 12/self.nyq], btype='band')
        b_bandpas1, a_bandpas1 = butter(2, [13/self.nyq, 30/self.nyq], btype='band')

        for window in self.corrected_epoched_eeg_data:
            
            bp_segment_eeg_data = lfilter(b_bandpas, a_bandpas, window)
            bp_segment_eeg_datab = lfilter(b_bandpas1, a_bandpas1, window)

            avg = np.mean(bp_segment_eeg_data**2)
            f, Pxx_den = welch(bp_segment_eeg_data, 250)

            avgb = np.mean(bp_segment_eeg_datab**2)
            fb, Pxx_denb = welch(bp_segment_eeg_datab, 250)

            f1.write(str(avg)+" "+str(np.mean(Pxx_den))+"   "+str(avgb)+"   "+str(np.mean(Pxx_denb))+"\n")
        '''

        
        
        for window in self.corrected_epoched_eeg_data:
            f, Pxx_den = welch(window, 250)

            i = 0
            while i < len(f):
                if f[i]>8:
                     upper_in=i
                     break
                else:
                     i=i+1

            while i < len(f):
                 if f[i]>12:
                      lower_in=i
                      break
                 else:
                      i=i+1
            while i < len(f):
                 if f[i]>13:
                      lower_inb=i
                      break
                 else:
                      i=i+1

            while i < len(f):
                 if f[i]>30:
                      upper_inb=i
                      break
                 else:
                      i=i+1

    
            corrected_mean_spectra_PSDperBin = np.mean(Pxx_den[upper_in:lower_in])
            corrected_mean_spectram_PSDperBin = np.mean(Pxx_den)
            corrected_mean_spectra_PSDperBinb = np.mean(Pxx_den[lower_inb:upper_inb])
            ratio=corrected_mean_spectra_PSDperBin
            ratiob=corrected_mean_spectra_PSDperBinb
            f1.write(str(ratio)+" "+str(ratiob)+"\n")

        
        period = 1/250
        
        plt.figure(1)

        x = np.linspace(0.0, 1500*period, 1500)
        y = epoched_corrected
        yf = fft(y[0])
        xf = np.linspace(0.0, 1.0/(2.0*period), 1500/2) 
        plt.xlabel("Frequency (Hz)")
        plt.ylabel("Gain")
        plt.plot(xf, 2.0/1500 * np.abs(yf[0:750])) 

        plt.grid()
        plt.show()
            

'''
        
        # For graphing, "de-epoch it
        self.corrected_eeg_data = []
        for channel in self.corrected_epoched_eeg_data:
            de_epoched = [i for epoch in channel for i in epoch]
            self.corrected_eeg_data.append(de_epoched)
  
        self.corrected_eeg_data = np.array(self.corrected_eeg_data).transpose()
        

        '''

    
'''
    
    def convert_to_freq_domain(self, data, NFFT = 500, FFTstep = 125):
        
        """
        
        Computes a spectogram via an FFT
        
        Input:
            - data: data to draw the spectogram on
            - NFFT: The number of data points used in each block
            - FFTstep: Length of the signal you want to calculate the Fourier transform of.
       
        Return:
            - list_spec_PSDperBin: list of periograms
                - 2D where columns are the periodograms of successive segments
            - list_freqs: The frequencies corresponding to the rows in spectrum
            - list_t_spec: The times corresponding the columns in spectrum
        
       """
        
        FFTstep = FFTstep   # do a new FFT every FFTstep data points
        overlap = NFFT - FFTstep  # half-second steps
    
        list_spec_PSDperHz,list_spec_PSDperBin, list_freqs, list_t_spec= [], [], [], []
        

        for filtered in data.T:
            spec_PSDperHz, freqs, t_spec = mlab.specgram(
                                           np.squeeze(filtered),
                                           NFFT=NFFT,
                                           window=mlab.window_hanning,
                                           Fs=self.sample_rate,
                                           noverlap=overlap
                                           ) 
            spec_PSDperBin = spec_PSDperHz * self.sample_rate / float(NFFT)  # convert to "per bin"
            list_spec_PSDperHz.append(spec_PSDperHz)
            list_spec_PSDperBin.append(spec_PSDperBin)
            list_freqs.append(freqs)
            list_t_spec.append(t_spec)
        
        return (np.array(list_spec_PSDperBin), np.array(list_freqs), np.array(list_t_spec))
'''
'''
    def plots(self, channel=0):
        """
       
        Plot the raw and filtered data of a channel as well as their spectrograms
        
        Input:
            - channel: channel whose data is to plot
        
        """
        self.raw_spec_PSDperBin, self.raw_freqs, self.raw_t_spec = self.convert_to_freq_domain(self.raw_eeg_data)
        
        
        fig = plt.figure()

        t_sec = np.array(range(0, self.raw_eeg_data.size)) / self.sample_rate
        
        ax1 = plt.subplot(221)
        plt.plot(t_sec, self.raw_eeg_data[:,channel])
        plt.ylabel('EEG (uV)')
        plt.xlabel('Time (sec)')
        plt.title('Raw')
        plt.xlim(t_sec[0], t_sec[-1])
        
        ax2 = plt.subplot(222)
        plt.pcolor(self.raw_t_spec[channel], self.raw_freqs[channel], 
                   10*np.log10(self.raw_spec_PSDperBin[channel]))
        plt.clim(25-5+np.array([-40, 0]))
        plt.xlim(t_sec[0], t_sec[-1])
        plt.ylim([0, 60]) 
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectogram of Unfiltered')
        
        
        self.corrected_spec_PSDperBin, self.corrected_freqs, self.corrected_t_spec = self.convert_to_freq_domain(self.corrected_eeg_data)
        ax3 = plt.subplot(223)
        plt.plot(t_sec, self.corrected_eeg_data[:,channel])
        plt.ylim(-100, 100)
        plt.ylabel('EEG (uV)')
        plt.xlabel('Time (sec)')
        plt.title('Filtered')
        plt.xlim(t_sec[0], t_sec[-1])
        
        ax4 = plt.subplot(224)
        plt.pcolor(self.corrected_t_spec[channel], self.corrected_freqs[channel], 
                   10*np.log10(self.corrected_spec_PSDperBin[channel]))
        plt.clim(25-5+np.array([-40, 0]))
        plt.xlim(t_sec[0], t_sec[-1])
        plt.ylim([0, 60])  
        plt.xlabel('Time (sec)')
        plt.ylabel('Frequency (Hz)')
        plt.title('Spectogram of Filtered')



        plt.tight_layout()
        plt.show()
        
    def extract_features(self, mu_band_Hz=[8,12]):
        
       
        # get the mean spectra and convert from PSD to uVrms
        self.corrected_mean_spectra_PSDperBin,self.corrected_mean_uVrmsPerSqrtBin =[],[]
        i = 0
        while i < self.number_channels:
            spectra = self.corrected_spec_PSDperBin[i]
            bool_inds = (self.corrected_freqs[i] > mu_band_Hz[0]) & (self.corrected_freqs[i] < mu_band_Hz[1])
            corrected_mean_spectra_PSDperBin = np.mean(spectra[bool_inds,:], 0)
            self.corrected_mean_spectra_PSDperBin.append(corrected_mean_spectra_PSDperBin)
            self.corrected_mean_uVrmsPerSqrtBin.append(np.sqrt(self.corrected_mean_spectra_PSDperBin))
            i = i + 1
            
        self.features = np.array(self.corrected_mean_uVrmsPerSqrtBin)
        
            
    
'''

fname_4 = "C:\\Users\\Yuan\\Documents\\New Text Document.txt"
test4 = Kiral_Korek_Preprocessing(fname_4)
test4.load_data_BCI()
test4.initial_preprocessing()
test4.epoch_and_remove_outlier()
f1.close()
#test4.plots()            
#test4.extract_features()


