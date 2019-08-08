/* eslint-disable no-mixed-spaces-and-tabs */
const express = require('express');
const expressApp = express();
const server = expressApp.listen(3000);
const io = require('socket.io').listen(server);
const osc = require('node-osc');
const path = require('path');

const CSVWriter = require('./modules/csv-writer.js');
const Socketer = require('./modules/socketer.js');

let oscServer = new osc.Server(12345, '127.0.0.1');

let mode = 'training';

const writer = new CSVWriter();
const socketer = new Socketer(io);

/*
Training Parameters
*/
//Other values will only be updated if collecting is true!
let collecting = false; //Will determine if collecting and sending to file currently.
let collectionTimer = null;
let lastSampleRateSendTime = getTimeValue();
let numSamplesForSampleRate = 0;

/*
Production Parameters
*/
const SEND_RATE = .2; // seconds
const EXPECTED_SAMPLE_RATE = 250; // samples per second
const SAMPLES_TO_SEND = EXPECTED_SAMPLE_RATE * SEND_RATE;
let toSendToML = [];
let state = 'stop'; // forward, turning, stop
const TURN_TIME = 3000; // milliseconds
let canGo = {left: 1,
	right: 1,
	forward: 1};
let lastRequestSensorDataTime = getTimeValue();

/*
Express
*/
expressApp.use(express.static(__dirname + '/app'));
expressApp.get('/', (req, res) => {
	res.sendFile(path.join(__dirname + '/app/index.html'));
});
expressApp.get('/production', (req, res) => {
	res.sendFile(path.join(__dirname + '/app/production.html'));
});

console.log('Listening on Port 3000!');


function getTimeValue() {
	/* Gets the current time (milliseconds since 1 January 1970) */
	return new Date().getTime();
}

let counterSpect1 = 0;
let counterSpect2 = 0;
oscServer.on('message', function (data) {
	let time = getTimeValue();

	// we send the fft once for every n packets we get, can tune according to the resolution and time length you want to see
	const FFT_WINDOW_LENGTH = 5; // 5 packets
	
	if (data[0] == 'fft'){
		if (data[1] == 1) {     // channel 1
			counterSpect1 += 1;
			if (counterSpect1 % FFT_WINDOW_LENGTH == 0) {
				socketer.sendFFTChannel1(data);
			}
		}
		else if (data[1] == 8) {     // channel 2
			counterSpect2 += 1;
			if (counterSpect2 % FFT_WINDOW_LENGTH == 0) {
				socketer.sendFFTChannel2(data);
			}
		}
	}
	// TODO: why is there fft in the /openbci address?
	else if (data[0] == '/openbci' && data.length < 10){
		if (collecting) {
			writer.appendSample(data.slice(1), 'time');
		}
		socketer.sendTimeSeries(data);
		//This data is used to make the graphs

		numSamplesForSampleRate++;

		if ((time - lastSampleRateSendTime) > 1000) { // check every second
			socketer.sendSampleRate(numSamplesForSampleRate);
			lastSampleRateSendTime = time;
			numSamplesForSampleRate = 0;
		}

		if (mode == 'production'){
			toSendToML.push(data.slice(1));
			if (toSendToML.length >= SAMPLES_TO_SEND){
				socketer.sendTimeSeriesToML(toSendToML);
				toSendToML = [];
			}
		}
	}
});

/*USER CONTROL OF COLLECTING BOOLEAN WITH SOCKET IO*/


// Socket IO:
io.on('connection', function(socket){
	console.log('A user connected socket');

	socket.on('stop', function(){
		clearInterval(collectionTimer);
		collecting = false;
		writer.endTest(false);
	});

	// Training Mode
	socket.on('collectQueue', function(clientRequest){
		mode = 'training';
		let collectQueue = clientRequest['queue'];
		let trialName = clientRequest['trialName'];
		let loop = clientRequest['loop'];
		console.log(collectQueue);
		console.log('This is trial: ' + trialName);

		writer.setActiveSensors(clientRequest['sensors']);

		let totalTime = 0;
		let times = [];
		collectQueue.forEach(function(command){
			totalTime+=command[1];
			times.push(totalTime);
		});

		writer.setDirection(collectQueue[0][0]);
		writer.setupCsvWriters();
		collecting = true;

		let j = 0;
		let time = 1;
		collectionTimer = setInterval(function(){
			if (time < totalTime) {
				if (time >= times[j]){
					// move onto next commmand
					writer.endTest(true);
					j += 1;
					writer.setDirection(collectQueue[j][0]); // setup new one!
				}
			}
			else {
				collecting = false;
				writer.endTest(true);

				console.log('Trial over.');
				if (loop == true){
					time = 0;
					collecting = true;
					j = 0;
					writer.setDirection(collectQueue[0][0]);
				}
				else {
					clearInterval(collectionTimer);
				}
			}
			time++;
		}, 1000);

	});



	// Production

	socketer.initializeAssistedDriving();

	// request data every 400 ms
	collectionTimer = setInterval(function(){
		let currentTime = getTimeValue();
		if (currentTime - lastRequestSensorDataTime > 500) {
			lastRequestSensorDataTime = currentTime;
			socketer.requestSensorData();
		}
	}, 400);

	socket.on('from sensors', function(data){
		// Let's see what security says 🚨
		console.log(data);
		data.state = state;
		socketer.sendSensorData(data);
	});

	socket.on('from self-driving (forward)', function(data){
		// data: {response, TURN_TIME, stop-permanent}
		if (state == 'forward') {
	    console.log('response from SD: ' + data['response']);
	    if (data['response'] != null) {
	        io.sockets.emit('to robotics', {'response': data['response']});
	    }
	    if (data['response'] == 'L' | data['response'] == 'R') {
	      // stopTime = 0;
	      // turn for data['TURN_TIME'] milliseconds
	      setTimeout(function(){
	        if (canGo.forward) {
	            io.sockets.emit('to robotics', {'response': 'F'});
	        } else {
	          io.sockets.emit('to robotics', {'response': 'S'});
	        }
	      }, data['duration']);
	    }
	    else if (data['response'] == 'S') {
	      // stopTime++;
	      // if (stopTime > MAX_STOP_TIME) {
	      //   state = "stop"
	      //
	      //   console.log("BACK TO STOP MODE")
				//   // stopTime = 0;
				socketer.sendStateToML(state);
	      state = 'stop';
	      // }
	    }
	  }
	  // else {
	  //   stopTime = 0;
	  // }
	});
	socket.on('from self-driving (safety)', function(data) {
		canGo = {left: data['left'],
			right: data['right'],
			forward: data['forward']};
		console.log(canGo);
		if (canGo.forward == 0 && state =='forward') {
			state = 'stop';
			console.log('CAN\'T GO FORWARD. STOPPING');
			socketer.sendMessageToWheelchair('S');
		}
		else if (canGo.left == 0 && state=='turning-L') {
			if (canGo.forward == 1) {
				state = 'forward';
				console.log('CAN\'T TURN LEFT. GOING FORWARD');
				socketer.sendMessageToWheelchair('F');
			}
			else {
				state = 'stop';
				console.log('CAN\'T TURN LEFT. CAN\'T GO FORWARD. STOPPING');
				socketer.sendMessageToWheelchair('S');
			}
		}
		else if (canGo.right == 0 && state=='turning-R') {
			if (canGo.forward == 1) {
				state = 'forward';
				console.log('CAN\'T TURN RIGHT. GOING FORWARD');
				socketer.sendMessageToWheelchair('F');
			}
			else {
				state = 'stop';
				console.log('CAN\'T TURN RIGHT. CAN\'T GO FORWARD. STOPPING');
				socketer.sendMessageToWheelchair('S');
			}
		}
		socketer.sendStateToML(state);
	});

	socket.on('data from ML', function(data){
		if (data.response != null) {
			console.log(data);
		}
    
		if (state == 'stop') {
			if (data['response'] == 'BLINK') {
				// go forward
				if (canGo.forward == 1) {
					socketer.sendMessageToWheelchair('F');
					state = 'forward';
				}
				else {
					console.log('CAN\'T GO FORWARD');
				}
			}
		} else if (state == 'forward') {
			if (data['response'] == 'BLINK') {
				// stop
				socketer.sendMessageToWheelchair('S');
				state = 'intermediate';
			}
		} else if (state == 'intermediate') {
			if (data['response'] == 'BLINK') {
				state = 'stop';
			} else if (data['response'] == 'L') {
				if (canGo.left == 1) {
					socketer.sendMessageToWheelchair('L');
					state='turning-' + data['response'];
					setTimeout(function(){
						if (canGo.forward == 1) {
							state='forward';
							socketer.sendStateToML(state);
							socketer.sendMessageToWheelchair('F');
						}
						else {
							state='stop';
							socketer.sendStateToML(state);
							socketer.sendMessageToWheelchair('S');
						}
					}, TURN_TIME);
				}
			} else if (data['response'] == 'R') {
				if (canGo.right == 1) {
					io.sockets.emit('to robotics', {'response': 'R'});
					state='turning-' + data['response'];
					setTimeout(function(){
						if (canGo.forward == 1) {
							state='forward';
							socketer.sendStateToML(state);
							socketer.sendMessageToWheelchair('F');
						}
						else {
							state='stop';
							socketer.sendStateToML(state);
							socketer.sendMessageToWheelchair('S');
						}
					}, TURN_TIME);
				}
			}
		}
		socketer.sendStateToML(state);
		socketer.sendDataForMLVisualization(data);
		console.log(data['response']);
	});

	socket.on('production', function(data){
		toSendToML = [];
		if (data['on'] == true) {
			mode = 'production';
			console.log(mode);
		}

		else {
			mode = 'training';
			console.log(mode);
		}
	});
});
