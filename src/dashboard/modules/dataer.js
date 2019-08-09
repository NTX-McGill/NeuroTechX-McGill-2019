module.exports = class Dataer {
    constructor(top) {
        this.top = top;
        this.FFT_WINDOW_LENGTH = 5; // we send the fft once for every n packets we get, can tune according to the resolution and time length you want to see
        this.fftChannel1Counter = 0;
        this.fftChannel2Counter = 0;
        
        this.numSamplesReceivedForSampleRate = 0;
        this.lastSampleRateSendTime = this.top.getTime();
        this.SAMPLE_RATE_POLL_FREQUENCY = 1000; // ms
    }
    handleFFTData(data) {
        if (data[1] == 1) {     // channel 1
			this.fftChannel1Counter += 1;
			if (this.fftChannel1Counter % this.FFT_WINDOW_LENGTH == 0) {
				this.top.getSocketer().sendFFTChannel1(data);
			}
		}
		else if (data[1] == 8) {     // channel 2
			this.fftChannel2Counter += 1;
			if (this.fftChannel2Counter % this.FFT_WINDOW_LENGTH == 0) {
				this.top.getSocketer().sendFFTChannel2(data);
			}
		}
    }
    handleTimeSeriesData(data) {
        this.top.getSocketer().sendTimeSeriesToClient(data);
        this.updateSampleRate();

        if (this.top.getMode() == 'training') {
            this.top.getTraining().receiveTimeSeriesData(data);
        }
        else if (this.top.getMode() == 'production') {
            this.top.getProduction().receiveTimeSeriesData(data);
        }

    }
    updateSampleRate() {
        this.numSamplesReceivedForSampleRate++;
        let time = this.top.getTime();

        if (time - this.lastSampleRateSendTime > this.SAMPLE_RATE_POLL_FREQUENCY) {
            this.top.getSocketer().sendSampleRate(this.numSamplesReceivedForSampleRate);
            this.numSamplesReceivedForSampleRate = 0;
            this.lastSampleRateSendTime = time;
        }
    }
}