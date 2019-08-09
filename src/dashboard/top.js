const osc = require('node-osc');
const path = require('path');
const express = require('express');
const CSVWriter = require('./modules/csv-writer.js');
const Socketer = require('./modules/socketer.js');
const Dataer = require('./modules/dataer.js');
const Training = require('./modules/training.js');
const Production = require('./modules/production.js');

module.exports = class Top {
    constructor() {
        this.expressApp = express();
        this.server = this.expressApp.listen(3000);
        this.io = require('socket.io').listen(this.server);

        const OSC_PORT = 12345;
        this.oscServer = new osc.Server(OSC_PORT, '127.0.0.1');
        this.writer = new CSVWriter(this);
        this.socketer = new Socketer(this);
        this.training = new Training(this);
        this.dataer = new Dataer(this);
        this.production = new Production(this);
        
        this.setMode('training');
        this.setUpExpressApp();

        this.oscServer.on('message', data => this.handleMessage(data));
        this.io.on('connection', socket => this.handleSocketIOConnection(socket));
    }
    getTime() {
        return new Date().getTime();
    }
    setMode(mode) {
        this.mode = mode;
        if (mode == 'training') {
            this.production.turnOff();
            this.training.turnOn();
        } else if (mode == 'production') {
            this.training.turnOff();
            this.production.turnOn();
        }
    }
    setUpExpressApp() {
        this.expressApp.use(express.static(__dirname + '/app'));
        this.expressApp.get('/', (req, res) => {
            res.sendFile(path.join(__dirname + '/app/index.html'));
        });
        this.expressApp.get('/production', (req, res) => {
            res.sendFile(path.join(__dirname + '/app/production.html'));
        });

        console.log('Listening on Port 3000!');
    }
    getMode() {
        return this.mode;
    }
    getWriter() {
        return this.writer;
    }
    getSocketIO() {
        return this.io;
    }
    getSocketer() {
        return this.socketer;
    }
    getDataer() {
        return this.dataer;
    }
    getTraining() {
        return this.training;
    }
    getProduction() {
        return this.production;
    }
    handleMessage(data) {
        if (data[0] == 'fft') {
            this.dataer.handleFFTData(data);
        }
        // TODO: why is there fft in the /openbci address?
        else if (data[0] == '/openbci' && data.length < 10){
            this.dataer.handleTimeSeriesData(data);
        }
    }
    handleSocketIOConnection(socket) {
        console.log('A user connected socket');
        socket.on('stop', () => this.handleSocketStopped());
        socket.on('collectQueue', clientRequest => this.handleCollectQueue(clientRequest));
        socket.on('from sensors', data => this.handleSensorData(data));
        socket.on('from self-driving (forward)', data => this.handleSelfDrivingGoForward(data));
        socket.on('from self-driving (safety)', data => this.handleSelfDrivingSafety(data));
        socket.on('data from ML', data => this.handleMLData(data));
        socket.on('production', data => this.handleProductionRequest(data));
    }
    handleSocketStopped() {
        this.training.finishedRecording();
    }
    handleCollectQueue(clientRequest) {
        this.setMode('training');
        this.training.setQueue(clientRequest);
    }
    handleSensorData(data) {
		// Let's see what security says 🚨
        console.log(data);
        data.state = state;
        this.socketer.sendSensorData(data);
    }
    handleSelfDrivingGoForward(data) {
        // data: {response, TURN_TIME, stop-permanent}
        this.production.forwardMode(data);
    }
    handleSelfDrivingSafety(data) {
        this.production.updateSafety(data);
    }
    handleMLData(data) {
        this.production.updateFromML(data);
    }
    handleProductionRequest(data) {
        if (data['on']) {
			this.setMode('production');
			console.log(this.mode);
		}

		else {
			this.setMode('training')
			console.log(this.mode);
		}
    }
}