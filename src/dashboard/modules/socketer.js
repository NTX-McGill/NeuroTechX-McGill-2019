module.exports = class Socketer {
	constructor(io) {
		this.io = io;
	}
	sendFFTChannel1(data) {
		this.io.sockets.emit('fft-test', {'data': data.slice(1)});
	}
	sendFFTChannel2(data) {
		this.io.sockets.emit('fft-test2', {'data': data.slice(1)});
	}
	sendTimeSeries(data) {
		this.io.sockets.emit('timeseries', {'time': new Date().getTime(), 'eeg': data.slice(1)});
	}
	sendSampleRate(numSamples) {
		this.io.sockets.emit('sample rate', {'sample rate': numSamples});
	}
	sendTimeSeriesToML(data) {
		this.io.sockets.emit('timeseries-prediction', {'data': data});
	}
	sendStateToML(state) {
		this.io.sockets.emit('to ML (state)', {'state': state});
	}
	initializeAssistedDriving() {
		this.io.sockets.emit('to self-driving (clear)', {}); // request data
	}
	requestSensorData() {
		this.io.sockets.emit('to robotics', {response: 'D'}); // request data
	}
	sendSensorData(data) {
		this.io.sockets.emit('to self-driving', data);
	}
	sendMessageToWheelchair(message) {
		this.io.sockets.emit('to robotics', {'response': message});
	}
	sendDataForMLVisualization(data) {
		this.io.sockets.emit('ML graphs', data);
	}
};