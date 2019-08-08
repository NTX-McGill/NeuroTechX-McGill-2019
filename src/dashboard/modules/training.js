module.exports = class Training {
    constructor(top) {
        this.top = top;
        this.collecting = false; // will determine if collecting and sending to file currently.
        this.collectionTimer = null;
    }
    turnOn() {
        // nop
    }
    turnOff() {
        // nop
        if (this.collectionTimer) {
            clearInterval(this.collectionTimer);
            this.collectionTimer = null;
        }
    }
    startCollecting() {
        this.collecting = true;
    }
    stopCollecting() {
        this.collecting = false;
    }
    receiveTimeSeriesData(data) {
        if (this.collecting) {
			this.top.getWriter().appendSample(data.slice(1), 'time');
        }
    }
    setQueue(clientRequest) {
        let collectQueue = clientRequest['queue'];
		let trialName = clientRequest['trialName'];
		let loop = clientRequest['loop'];
		console.log(collectQueue);
		console.log('This is trial: ' + trialName);

		this.top.getWriter().setActiveSensors(clientRequest['sensors']);

		let totalTime = 0;
		let times = [];
		collectQueue.forEach((command) => {
			totalTime += command[1];
			times.push(totalTime);
		});

		this.top.getWriter().setDirection(collectQueue[0][0]);
		this.top.getWriter().setupCsvWriters(trialName);
		this.startCollecting();

		let j = 0;
		let time = 1;
		this.collectionTimer = setInterval(() => {
			if (time < totalTime) {
				if (time >= times[j]){
					// move onto next commmand
					this.top.getWriter().endTest(true);
					j += 1;
					this.top.getWriter().setDirection(collectQueue[j][0]); // setup new one!
				}
			}
			else {
				this.stopCollecting();
				this.top.getWriter().endTest(true);

				console.log('Trial over.');
				if (loop == true){
					time = 0;
					this.collecting = true;
					j = 0;
					this.top.getWriter().setDirection(collectQueue[0][0]);
				}
				else {
					clearInterval(this.collectionTimer);
				}
			}
			time++;
		}, 1000); // each command is in multiples of 1 second
    }
    finishedRecording() {
        clearInterval(this.collectionTimer);
		this.stopCollecting();
		this.top.getWriter().endTest(false);
    }

}