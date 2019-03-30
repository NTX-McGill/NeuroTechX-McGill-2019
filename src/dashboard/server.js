// const { app, BrowserWindow } = require('electron');
const dgram = require('dgram');
// const events = require('events');
const express = require('express');
const app_express = express();
const server = app_express.listen(3000);
const io = require('socket.io').listen(server);
const fs = require('fs');

var osc = require('node-osc');
var oscServer = new osc.Server(12345, '127.0.0.1');

const createCSVWriter = require('csv-writer').createObjectCsvWriter;
var csvTimeWriter;

const sendRate = .2; // seconds
const expectedSampleRate = 250; // samples per second
var samplesToSend = expectedSampleRate * sendRate;
var toSend = [];
var mode = "training";

//Will determine if collecting and sending to file currently.
//Other values will only be updated if collecting is true!
var collecting = false;
var duration = 0;
var direction = "none";
var active = [];
var collectionTimer=null;
var loop = false;

var path = require('path');

var spawn = require("child_process").spawn; // to run

/*EXPRESS*/
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


/*SETTING UP CSV WRITERS*/

/*Time csv writer*/
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
//Global variable will be used to store time data


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


var trialName=null;
var timeTesting = getTimeValue();
var numSamples = 0;
var counterSpect1 = 0;
var counterSpect2 = 0;
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
      if (numSamples < expectedSampleRate*0.9 || // check for ± 10%
          numSamples > expectedSampleRate*1.1) {
            // console.log("\n-------- IRREGULAR SAMPLE RATE --------")
          }
      timeTesting = time;
      // console.log("Sample rate: " + numSamples);
      numSamples = 0;
    }

    if(mode == "production"){
      toSend.push(data.slice(1));
      if(toSend.length >= samplesToSend){
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


//Socket IO:
io.on('connection', function(socket){
  console.log('A user connected socket');

  if(mode == "production"){
    socket.on("data from ML", function(data){
      io.sockets.emit('to robotics', {'response': data['response']});
      console.log(data['response']);
    });
  }

  socket.on('stop', function(){
      clearInterval(collectionTimer);
      collecting = false;
      endTest(false);
  });

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
  //Production

  socket.on("data from ML", function(data){
    io.sockets.emit('to robotics', {'response': data['response']});
    io.sockets.emit('ML graphs', data);
    console.log(data['response']);
  });

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
