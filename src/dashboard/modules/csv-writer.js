/* Write data collected to CSV file */

module.exports = class CSVWriter {
	constructor() {
		this.createCSVWriter = require('csv-writer').createObjectCsvWriter;

		// Formatting header of time CSV
		this.timeHeader = [{id: 'time', title: 'TIME'},
			{id: 'channel1', title: 'CHANNEL 1'},
			{id: 'channel2', title: 'CHANNEL 2'},
			{id: 'channel3', title: 'CHANNEL 3'},
			{id: 'channel4', title: 'CHANNEL 4'},
			{id: 'channel5', title: 'CHANNEL 5'},
			{id: 'channel6', title: 'CHANNEL 6'},
			{id: 'channel7', title: 'CHANNEL 7'},
			{id: 'channel8', title: 'CHANNEL 8'},
			{id: 'direction', title: 'STATE'}];
        
		// Setting up array for actually storing the time data where each index has
		// the header data (time, channels 1-8)
		this.timeHeaderToWrite = {
			time: 'Time',
			channel1: 'Channel 1',
			channel2: 'Channel 2',
			channel3: 'Channel 3',
			channel4: 'Channel 4',
			channel5: 'Channel 5',
			channel6: 'Channel 6',
			channel7: 'Channel 7',
			channel8: 'Channel 8',
			direction: 'Direction',
		};
		this.csvTimeWriter = null;
        
		this.timeSamples = [this.timeHeaderToWrite];
		this.activeSensors = [];
		this.motorImageryDirection = 'none';
	}

	setupCsvWriters(trialName) {
		let date = new Date();
		// Formatting date as YYYY-MM-DD-hr-min-sec
		let day = date.getFullYear() + '-' + (date.getMonth()+1) + '-' +
                       date.getDate() + '-' + date.getHours() + '-' +
                       date.getMinutes() + '-' + date.getSeconds();
    
		this.csvTimeWriter = this.createCSVWriter({
			path: __dirname + '/../data/' + trialName + '-'
                              + day + '.csv',
			//File name of CSV for time test
			header: this.timeHeader,
			append: true
		});
	}
    
	setActiveSensors(activeSensors) {
		this.activeSensors = activeSensors;
	}

	setDirection(direction) {
		this.motorImageryDirection = direction;
	}

	appendSample(data, type) {
		/*
        When we're collecting data (collecing = True), this function is run for every
        sample. It writes samples to a CSV file, recording the collected data and the time.
        Takes in 'data' object which has 'time' and 'data' attributes, and type (fft or time)
        */

		let channelData = [];
		for (let i = 0; i < 8; i++) {
			if (this.activeSensors[i]) {
				// Only get data for active channels
				channelData.push(data[i]);
			}
			else {
				channelData.push(null);
			}
		}
    
		if (type == 'time') {
			let timeSampleToPush = {time: new Date().getTime(),
				channel1: channelData[0],
				channel2: channelData[1],
				channel3: channelData[2],
				channel4: channelData[3],
				channel5: channelData[4],
				channel6: channelData[5],
				channel7: channelData[6],
				channel8: channelData[7],
				direction: this.direction,
			};

			this.timeSamples.push(timeSampleToPush);
		}
	}

	endTest(save) {
		/*
        This function runs when a trial is finishing. If data is meant to be saved,
        test number increments by one and testNumber is reset. timeSamples and
        fftSamples are reset as well, to just the headers.Takes boolean argument.
        */
		if (save){
			// time data is written to CSV
			this.csvTimeWriter.writeRecords(this.timeSamples).then(() => {
				console.log('Added some time samples');
			});
		}
		else {
			console.log('User terminated trial. No data saved.');
		}
    
		// Both global variables are reset
		this.timeSamples = [];
	}
};