// const { app, BrowserWindow } = require('electron');
const dgram = require('dgram');
// const events = require('events');
const express = require('express');
const app_express = express();
const server = app_express.listen(3000);
const io = require('socket.io').listen(server);
const fs = require('fs');
const osc = require('node-osc');
const createCSVWriter = require('csv-writer').createObjectCsvWriter;
const path = require('path');

var oscServer = new osc.Server(12345, '127.0.0.1');
// var spawn = require("child_process").spawn; // to run
var mode = "training";


/*
Training Parameters
*/
var csvTimeWriter;
//Other values will only be updated if collecting is true!
var collecting = false; //Will determine if collecting and sending to file currently.
var duration = 0;
var direction = "none";
var active = [];
var collectionTimer=null;
var loop = false;
var trialName = null;
var timeTesting = getTimeValue();
var numSamples = 0;
var counterSpect1 = 0;
var counterSpect2 = 0;

/*
Setting up CSV writers
*/

/* Formatting header of time CSV */
const timeHeader = [{id: 'time', title: 'TIME'},
                    {id: 'channel1', title: 'CHANNEL 1'},
                    {id: 'channel2', title: 'CHANNEL 2'},
                    {id: 'channel3', title: 'CHANNEL 3'},
                    {id: 'channel4', title: 'CHANNEL 4'},
                    {id: 'channel5', title: 'CHANNEL 5'},
                    {id: 'channel6', title: 'CHANNEL 6'},
                    {id: 'channel7', title: 'CHANNEL 7'},
                    {id: 'channel8', title: 'CHANNEL 8'},
                    {id: 'direction', title: 'STATE'},]

/* Setting up array for actually storing the time data where each index has
the header data (time, channels 1-8) */
const timeHeaderToWrite = {
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

var timeSamples = [timeHeaderToWrite];
// Global variable will be used to store time data

/*
Production Parameters
*/
const SEND_RATE = .2; // seconds
const EXPECTED_SAMPLE_RATE = 250; // samples per second
const SAMPLES_TO_SEND = EXPECTED_SAMPLE_RATE * SEND_RATE;
var toSend = [];
var state = "forward" // forward, turning, stop
const TURN_TIME = 3000; // milliseconds
var canGo = {left: 1,
             right: 1,
             forward: 1};
var stopTime = 0;
const MAX_STOP_TIME = 20;
var roboticsTime = getTimeValue()
var roboticsCurrentTime;

/*
Express
*/
// Sets static directory as public
app_express.use(express.static(__dirname + '/public'));

app_express.get('/', (req, res) => {
  res.sendFile(path.join(__dirname + '/public/index.html'));
});

app_express.get('/production', (req, res) => {
  res.sendFile(path.join(__dirname + '/public/production.html'));
});

console.log('Listening on Port 3000!')


/* Gets the current time */
function getTimeValue() {
  var dateBuffer = new Date();
  var Time = dateBuffer.getTime();
  //Milliseconds since 1 January 1970
  return Time;
}


/* Sets the csvwriters to the correct paths! */
function setupCsvWriters(){
    let date = new Date();
    var day = date.getFullYear() + '-' + (date.getMonth()+1) + '-' +
                   date.getDate() + '-' + date.getHours() + '-' +
                   date.getMinutes() + '-' + date.getSeconds();
   //Formatting date as YYYY-MM-DD-hr-min-sec

    csvTimeWriter = createCSVWriter({
          path: __dirname + '/data/' + trialName + '-'
                          + day + '.csv',
          //File name of CSV for time test
          header: timeHeader,
          append: true
    });
}

oscServer.on("message", function (data) {
  let time = getTimeValue(); // milliseconds since January 1 1970. Adjust?
  let dataWithoutFirst = []; // TODO

  let toWrite = {'time': time, 'data': data.slice(1), 'direction': direction};
  var numPacketsSpect = 5;       // we send the fft once for every n packets we get, can tune according to the resolution and time length you want to see

  if (data[0] == 'fft'){
    if (data[1] == 1) {     // channel 1
    counterSpect1 += 1;
      if (counterSpect1 % numPacketsSpect == 0) {
        io.sockets.emit('fft-test', {'data': data.slice(1)});
        // console.log(counter);
      }
    }
    if (data[1] == 8) {     // channel 2
      counterSpect2 += 1;
      if (counterSpect2 % numPacketsSpect == 0) {
        io.sockets.emit('fft-test2', {'data': data.slice(1)});
      }
    }
  }
  // TODO: why is there fft in the /openbci address?
  else if (data[0] == '/openbci' && data.length < 10){
    if (collecting) {
      appendSample(toWrite, type="time");
    }
    io.sockets.emit('timeseries', {'time': time, 'eeg': data.slice(1)});
    //This data is used to make the graphs

    numSamples++;
    if ((time - timeTesting) > 1000) { // check every second
      io.sockets.emit('sample rate', {'sample rate': numSamples});
      if (numSamples < EXPECTED_SAMPLE_RATE*0.9 || // check for ± 10%
          numSamples > EXPECTED_SAMPLE_RATE*1.1) {
            // console.log("\n-------- IRREGULAR SAMPLE RATE --------")
          }
      timeTesting = time;
      // console.log("Sample rate: " + numSamples);
      numSamples = 0;
    }

    if(mode == "production"){
      toSend.push(data.slice(1));
      if(toSend.length >= SAMPLES_TO_SEND){
        io.sockets.emit('timeseries-prediction', {'data': toSend});
        toSend = [];
      }
    }
  }
});

/* When we're collecting data (collecing = True), this function is run for every
sample. It writes samples to a CSV file, recording the collected data and the time.

Takes in 'data' object which has 'time' and 'data' attributes, and type (fft or time) */
function appendSample(data, type){
  channelData = [];
  for (i = 0; i < 8; i++) {
    if (active[i] == 1) {//Only get data for active channels
        channelData[i] = data['data'][i];
    }
    else {
      channelData[i] = null;
    }
  }

  if (type == 'time') {
    let timeSampleToPush = {time: data['time'],
                    channel1: channelData[0],
                    channel2: channelData[1],
                    channel3: channelData[2],
                    channel4: channelData[3],
                    channel5: channelData[4],
                    channel6: channelData[5],
                    channel7: channelData[6],
                    channel8: channelData[7],
                    direction: data['direction'],
                  }
    //channelData is 1D for time
    timeSamples.push(timeSampleToPush);
    //Updating global timeSamples variable
  }
}


/*END OF TRIAL*/


/* This function runs when a trial is finishing.If data is meant to be saved,
test number increments by one and testNumber is reset. timeSamples and
fftSamples are reset as well, to just the headers.Takes boolean argument. */
function endTest(saved){
  if(saved){
    // time data is written to CSV
    csvTimeWriter.writeRecords(timeSamples).then(() => {
      console.log('Added some time samples');
    });
  }
  else{
      console.log("User terminated trial. No data saved.")
  }

  //Both global variables are reset
  timeSamples = [];
}


/*USER CONTROL OF COLLECTING BOOLEAN WITH SOCKET IO*/


// Socket IO:
io.on('connection', function(socket){
  console.log('A user connected socket');

  socket.on('stop', function(){
      clearInterval(collectionTimer);
      collecting = false;
      endTest(false);
  });

  // Training Mode
  socket.on('collectQueue', function(clientRequest){
    mode = "training";
    timeSamples = [timeHeaderToWrite];
    collectQueue = clientRequest['queue'];
    trialName = clientRequest['trialName'];
    loop = clientRequest['loop'];
    console.log(collectQueue);
    console.log("This is trial: " + trialName);


    active = clientRequest['sensors'];

    let totalTime = 0;
    let times = [];
    collectQueue.forEach(function(command){
      totalTime+=command[1];
      times.push(totalTime);
    });

    // console.log(totalTime);

    direction = collectQueue[0][0];
    setupCsvWriters();
    collecting = true;

    let j = 0;
    let time = 1;
    collectionTimer = setInterval(function(){
        if (time < totalTime) {
          if (time >= times[j]){
            // move onto next commmand
            endTest(true, true); //end old test
            j += 1;
            direction = collectQueue[j][0]; //setup new one!
          }
        }
        else {
          collecting = false;
          endTest(true, true);

          console.log("Trial over.");
          if(loop == true){
              time = 0;
              collecting = true;
              j = 0;
              direction = collectQueue[0][0];
          }
          else{
              clearInterval(collectionTimer);
          }
        }
        time++;
    }, 1000);

  });

  // Production


  //
  // socket.on("from sensors", function(data){
  //   // data: {front, left, right, front-tilt}
  //   io.sockets.emit('to safety', data);
  // });
  //
  // socket.on("from safety", function(data){
  //
  //
  // });

  // request data every 200 ms

  collectionTimer = setInterval(function(){
    currentTime = getTimeValue();
    if (currentTime - roboticsTime > 200) {
      roboticsTime = currentTime;
      io.sockets.emit('to robotics', {response: 'D'}) // request data
      console.log('hi')
    }
  }, 200); // 5 times/sec

  socket.on("from sensors", function(data){
    // if (state == "forward") {
      // Let's roll 🚘 and see what security says 🚨
      console.log(data)
      data.state = state
      io.sockets.emit("to self-driving", data);
    // }
    // else {
      //
      // io.sockets.emit("to safety", data);
    // }
  });

  socket.on("from self-driving (forward)", function(data){
    // data: {response, TURN_TIME, stop-permanent}
    if (state == "forward") {
      io.sockets.emit('to robotics', {'response': data['response']});
      if (data['response'] == "L" | data['response'] == "R") {
        // stopTime = 0;
        // turn for data['TURN_TIME'] milliseconds
        setTimeout(function(){
          io.sockets.emit('to robotics', {'response': "F"});
        }, data['duration']);
      }
      else if (data['response'] == "S") {
        // stopTime++;
        // if (stopTime > MAX_STOP_TIME) {
        //   state = "stop"
        //
        //   console.log("BACK TO STOP MODE")
        //   // stopTime = 0;
        io.sockets.emit('to ML (state)', {'state': state});
        state = "stop"
        // }
      }
    }
    // else {
    //   stopTime = 0;
    // }
  });
  socket.on("from self-driving (safety)", function(data) {
    canGo = {left: data['left'],
             right: data['right'],
             forward: data['forward']};
    if (canGo.left == 0 && state=="turning-L") {
      if (canGo.forward == 1) {
        state = "forward";
        console.log("CAN'T TURN LEFT. GOING FORWARD");
        io.sockets.emit('to robotics', {'response': "F"});
      }
      else {
        state = "stop";
        console.log("CAN'T TURN LEFT. CAN'T GO FORWARD. STOPPING")
        io.sockets.emit('to robotics', {'response': "S"});
      }
    }
    else if (canGo.right == 0 && state=="turning-R") {
      if (canGo.forward == 1) {
        state = "forward";
        console.log("CAN'T TURN RIGHT. GOING FORWARD");
        io.sockets.emit('to robotics', {'response': "F"});
      }
      else {
        state = "stop";
        console.log("CAN'T TURN RIGHT. CAN'T GO FORWARD. STOPPING");
        io.sockets.emit('to robotics', {'response': "S"});
      }
    }
    io.sockets.emit('to ML (state)', {'state': state});
  });

  socket.on("data from ML", function(data){
    if (state == "stop") {
      if (data['response'] == "BLINK") {
        // go forward
        if (canGo.forward == 1) {
          io.sockets.emit('to robotics', {'response': "F"});
          state = "forward";
        }
        else {
          console.log("CAN'T GO FORWARD");
        }
      }
    } else if (state == "forward") {
      if (data['response'] == "BLINK") {
        // stop
        io.sockets.emit('to robotics', {'response': "S"});
        state = "intermediate"
      }
    } else if (state == "intermediate") {
      if (data['response'] == "BLINK") {
        state = "stop"
      } else if (data['response'] == "L") {
        if (canGo.left == 1) {
          io.sockets.emit('to robotics', {'response': "L"});
          state="turning-" + data['response']
          setTimeout(function(){
            state="forward";
            io.sockets.emit('to ML (state)', {'state': state});
            io.sockets.emit('to robotics', {'response': "F"});
          }, TURN_TIME);
        }
      } else if (data['response'] == "R") {
        if (canGo.right == 1) {
          io.sockets.emit('to robotics', {'response': "R"});
          state="turning-" + data['response']
          setTimeout(function(){
            state="forward";
            io.sockets.emit('to ML (state)', {'state': state});
            io.sockets.emit('to robotics', {'response': "F"});
          }, TURN_TIME);
        }
      }
    }

    io.sockets.emit('to ML (state)', {'state': state});

    io.sockets.emit('ML graphs', data);
    console.log(data['response']);
  });

  // socket.on("from safety", function(data) {
  //   io.sockets.emit('to robotics', {'response': data['response']});
  // })

  socket.on('production', function(data){
    toSend = [];
    if (data['on'] == true) {
      mode = "production";
      console.log(mode);
    }

    else {
      mode = "training";
      console.log(mode);
    }
    // var process = spawn('python',["../real_time_ML.py"]);
    // console.log('spawned')
  });
});
