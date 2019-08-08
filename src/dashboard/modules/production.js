const SEND_RATE = .2; // seconds
const EXPECTED_SAMPLE_RATE = 250; // samples per second

module.exports = class Production {
    constructor(top) {
        this.top = top;
        this.toSendToML = [];
        this.SAMPLES_TO_SEND_TO_ML = EXPECTED_SAMPLE_RATE * SEND_RATE;
        this.TURN_TIME = 3000; // milliseconds

        this.state = 'stop';
        this.canGo = {left: 1,
            right: 1,
            forward: 1};
        this.collectionTimer = null;
        this.lastRequestSensorDataTime = this.top.getTime();
    }
    turnOn() {
        this.top.getSocketer().initializeSelfDriving();
        
        // request data every 400 ms
        this.collectionTimer = setInterval(() => {
            let currentTime = this.top.getTime();
            if (currentTime - this.lastRequestSensorDataTime > 500) {
                this.lastRequestSensorDataTime = currentTime;
                this.top.getSocketer().requestSensorData();
            }
        }, 400);
    }
    turnOff() {
        // nop
        clearInterval(this.collectionTimer)
    }
    receiveTimeSeriesData() {
        this.toSendToML.push(data.slice(1));
        if (this.toSendToML.length >= this.SAMPLES_TO_SEND_TO_ML){
            this.top.getSocketer().sendTimeSeriesToML(this.toSendToML);
            this.toSendToML = [];
        }
    }
    getState() {
        return this.state;
    }
    setState(state) {
        this.state = state;
    }
    forwardMode() {
        // data: {response, TURN_TIME, stop-permanent}
		if (this.state == 'forward') {
            console.log('response from SD: ' + data['response']);
            if (data['response'] != null) {
                this.top.getSocketer().sendMessageToWheelchair(data['response']);
            }
            if (data['response'] == 'L' || data['response'] == 'R') {
              setTimeout(function(){
                if (this.canGo.forward) {
                    this.top.getSocketer().sendMessageToWheelchair('F');
                } else {
                    this.top.getSocketer().sendMessageToWheelchair('S');
                }
              }, data['duration']);
            }
            else if (data['response'] == 'S') {
                this.top.getSocketer().sendStateToML(state);
                this.setState('stop');
            }
        }
    }
    updateSafety(data) {
        this.canGo = {left: data['left'],
			right: data['right'],
			forward: data['forward']};
		console.log(this.canGo);
		if (this.canGo.forward == 0 && this.state =='forward') {
            this.setState('stop');
			console.log('CAN\'T GO FORWARD. STOPPING');
			this.top.getSocketer().sendMessageToWheelchair('S');
		}
		else if (this.canGo.left == 0 && this.state=='turning-L') {
			if (this.canGo.forward == 1) {
                this.setState('forward');
				console.log('CAN\'T TURN LEFT. GOING FORWARD');
				this.top.getSocketer().sendMessageToWheelchair('F');
			}
			else {
                this.setState('stop');
				console.log('CAN\'T TURN LEFT. CAN\'T GO FORWARD. STOPPING');
				this.top.getSocketer().sendMessageToWheelchair('S');
			}
		}
		else if (this.canGo.right == 0 && this.state=='turning-R') {
			if (this.canGo.forward == 1) {
                this.setState('state');
				console.log('CAN\'T TURN RIGHT. GOING FORWARD');
				this.top.getSocketer().sendMessageToWheelchair('F');
			}
			else {
                this.setState('stop');
				console.log('CAN\'T TURN RIGHT. CAN\'T GO FORWARD. STOPPING');
				this.top.getSocketer().sendMessageToWheelchair('S');
			}
		}
		this.top.getSocketer().sendStateToML(state);
    }
    updateFromML(data) {
        if (data.response != null) {
			console.log(data);
		}
    
		if (this.state == 'stop') {
			if (data['response'] == 'BLINK') {
				// go forward
				if (this.canGo.forward == 1) {
					this.top.getSocketer().sendMessageToWheelchair('F');
					this.setState('forward');
				}
				else {
					console.log('CAN\'T GO FORWARD');
				}
			}
		} else if (this.state == 'forward') {
			if (data['response'] == 'BLINK') {
				// stop
				this.top.getSocketer().sendMessageToWheelchair('S');
				this.setState('intermediate');
			}
		} else if (this.state == 'intermediate') {
			if (data['response'] == 'BLINK') {
				this.setState('stop');
			} else if (data['response'] == 'L') {
				if (this.canGo.left == 1) {
					this.top.getSocketer().sendMessageToWheelchair('L');
					this.setState('turning-' + data['response']);
					setTimeout(function(){
						if (this.canGo.forward == 1) {
							this.setState('forward');
							this.top.getSocketer().sendStateToML(state);
							this.top.getSocketer().sendMessageToWheelchair('F');
						}
						else {
							this.setState('stop');
							this.top.getSocketer().sendStateToML(state);
							this.top.getSocketer().sendMessageToWheelchair('S');
						}
					}, this.TURN_TIME);
				}
			} else if (data['response'] == 'R') {
				if (this.canGo.right == 1) {
                    this.top.getSocketer().sendMessageToWheelchair('R');
					this.setState('turning-' + data['response']);
					setTimeout(function(){
						if (this.canGo.forward == 1) {
							this.setState('forward');
							this.top.getSocketer().sendStateToML(state);
							this.top.getSocketer().sendMessageToWheelchair('F');
						}
						else {
							this.setState('stop');
							this.top.getSocketer().sendStateToML(state);
							this.top.getSocketer().sendMessageToWheelchair('S');
						}
					}, this.TURN_TIME);
				}
			}
		}
		this.top.getSocketer().sendStateToML(state);
		this.top.getSocketer().sendDataForMLVisualization(data);
		console.log(data['response']);
    }
}