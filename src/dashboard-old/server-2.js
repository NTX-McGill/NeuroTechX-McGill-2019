// const { app, BrowserWindow } = require('electron');
const dgram = require('dgram');
const events = require('events');
const express = require('express');
const app_express = express();
const server = app_express.listen(3000);
const io = require('socket.io').listen(server);
const fs = require('fs');


const createCSVWriter = require('csv-writer').createObjectCsvWriter;
var csvTimeWriter, csvFFTWriters;

//Will determine if collecting and sending to file currently.
//Other values will only be updated if collecting is true!
var collecting = false;
var duration = 0;
var direction = "none";
var active = [];
var collectionTimer=null;

/*EXPRESS*/


// Sets static directory as public
app_express.use(express.static(__dirname + '/public'));

app_express.get('/', (req, res) => {
  res.send('index');
});

console.log('Listening on Port 3000!')


/*TIME*/


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
                    {id: 'channel8', title: 'CHANNEL 8'}]

/* Setting up array for actually storing the time data where each index has
the header data (time, channels 1-8) */
const timeHeaderToWrite = {time: 'Time',
                  channel1: 'Channel 1',
                  channel2: 'Channel 2',
                  channel3: 'Channel 3',
                  channel4: 'Channel 4',
                  channel5: 'Channel 5',
                  channel6: 'Channel 6',
                  channel7: 'Channel 7',
                  channel8: 'Channel 8'
                };

var timeSamples = [timeHeaderToWrite];
//Global variable will be used to store time data


/*FFT csv writer*/
/* fft CSV will have header with the time and 1 - 125Hz */
const fftHeader = [{id: 'time', title: 'TIME'}];
for (i=0; i<125; i++) {
  fftHeader.push({id: 'f' + (i+1), title: (i+1) + 'Hz'})
}

/* Same as above */
const fftHeaderToWrite = {time: 'Time'};
for (i=0; i<125; i++) {
  fftHeaderToWrite['f' + (i+1)] = (i+1) + 'Hz';
}

/* Initialize fftSamples to a list of headers for each channel. At index 0 are
channel one's headers which are time, and 1-125Hz, etc. */
const fftSamplesHeaders = [];
for (i=0; i<8; i++) {
  fftSamplesHeaders.push([fftHeaderToWrite]);
}

var fftSamples = fftSamplesHeaders;
//Global variable. Will be used to store fft data.



/* Sets the csvwriters to the correct paths! */
function setupCsvWriters(){
    let date = new Date();
    var day = date.getFullYear() + '-' + date.getMonth() + '-' +
                   date.getDate() + '-' + date.getHours() + '-' +
                   date.getMinutes() + '-' + date.getSeconds();
   //Formatting date as YYYY-MM-DD-hr-min-sec

   csvTimeStampWriter = createCSVWriter({
         path: __dirname + '/data/time-stamp-' + testNumber + '-'
                         + day + '.csv',
         //File name of CSV for time test
         header: [{id: 'start_time', title: 'START TIME'},
                             {id: 'direction', title: 'DIRECTION'},
                             {id: 'duration', title: 'DURATION'}],
         append: true
   });

   csvTimeStampWriter.writeRecords([{start_time: 'START TIME',
               direction: 'DIRECTION',
               duration: 'DURATION',
             }]);


    csvTimeWriter = createCSVWriter({
          path: __dirname + '/data/time-test-' + testNumber + '-' + direction + '-'
                          + day + '.csv',
          //File name of CSV for time test
          header: timeHeader,
          append: true
    });
    csvFFTWriters = [];
    //For fft, makes array of CSV writers for each channel
    for (i=0; i<8; i++) {
      csvFFTWriters.push(createCSVWriter({
        path: __dirname + '/data/fft-' + (i+1) + '-test-' + testNumber + '-'
                        + direction + '-' + day + '.csv',//File name of CSVs for fft
        header: fftHeader,
        append: true
      }));
    }
}



/*NAMING OF SAMPLES*/


/* These are manual settings that we can use to keep track of testNumber as an example */
var settings = JSON.parse(fs.readFileSync(__dirname + '/data_settings.json', 'utf8'));
console.log("Currently running on these settings: \n" + settings);
let testNumber = settings['testNumber'];//Could read in data from dashboard





/*SETTING UP NETWORKING*/


/*UDP Client*/
/* Function that creates a UDP client to listen to the OpenBCI GUI */
function UDPClient(port, host) {
  this.port = port;
  this.host = host;
  this.data = [];
  this.events = new events.EventEmitter();
  this.connection = dgram.createSocket('udp4');
  this.connection.on('listening', this.onListening.bind(this));
  this.connection.on('message', this.onMessage.bind(this));
  this.connection.bind(this.port, this.host);
};

/* Function that logs if UDP client is listening */
UDPClient.prototype.onListening = function() {
  console.log('Listening for data...');
};

/* Function that, upon message from OpenBCI UDP, emits an event called 'sample'
for further classification.*/
UDPClient.prototype.onMessage = function(msg) {
  parsedMessage = JSON.parse(msg.toString())
  this.events.emit('sample', parsedMessage);
  // for spectrogram
  // const byteMessage = Buffer.from(msg.toString());
  // if (parsedMessage['type'] == 'fft') {
  //   broadcasting_client.send(msg, 12346, 'localhost', (err) => {
  //     broadcasting_client.close();
  //   });
  // }
};


/* Here we actually create the UDP Client listening to 127.0.0.1:12345
OpenBCI GUI must be set to communicate on this port for fft and time data */
var client = new UDPClient(12345, "127.0.0.1");



/* RECORDING DATA */


/*
When message from OpenBCI is received, onMessage function emits event 'sample'
that this function recognizes. onMessage passes the parsed message (data)
which we append to a CSV. We then ping the client.
Data Format: {
                'time': time,
                'eeg': {'data': [0.5,3,-5,40,5,32,8,1]}
                    data[index] is the eeg value at sensor-index
              }
*/
client.events.on('sample', function(data) {
  let time = getTimeValue();//Milliseconds since January 1 1970. Adjust?
  let toWrite = {'time': time, 'data': data['data']};
  if (data['type'] == 'fft') {
    if (collecting) {
      appendSample(toWrite, type="fft"); // Write to file
    }
    io.sockets.emit('fft', {'time': time, 'eeg': data});
    // Regardless of if we're collecting, we're always sending data to client
    // This data is used to make the graphs
  }
  else {
    if (collecting) {
      appendSample(toWrite, type="time");
    }
    io.sockets.emit('timeseries', {'time': time, 'eeg': data});
    //This data is used to make the graphs
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
  //When fft data is passed
  if (type =='fft') {
    let fftSamplesToPush = [];
    //For each channel gets values for 1-125Hz
    for (i=0; i<8; i++) {
      fftSamplesToPush.push({time: data['time']});
      for (j=0; j<125; j++) {
         fftSamplesToPush[i]['f' + (j+1)] = channelData[i][j];
         //channelData is 2D for fft
      }
    }
    for (i=0; i<8; i++) {
      fftSamples[i].push(fftSamplesToPush[i]);
      //Pushing 8 125 value arrays to global fftSamples variable
    }
  }

  else if (type == 'time') {
    let timeSampleToPush = {time: data['time'],
                    channel1: channelData[0],
                    channel2: channelData[1],
                    channel3: channelData[2],
                    channel4: channelData[3],
                    channel5: channelData[4],
                    channel6: channelData[5],
                    channel7: channelData[6],
                    channel8: channelData[7]
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
    settings['testNumber'] += 1;
    let settingsString = JSON.stringify(settings);

    //Updating json file containing test number
    fs.writeFile('data_settings.json', settingsString, 'utf8', function(err){
      if (err) throw err;
      console.log('Updated Test Number!');
      testNumber = settings['testNumber'];
    });
  }
  else{
      console.log("User terminated trial. No data saved.")
  }

  //Both global variables are reset
  timeSamples = [timeHeaderToWrite];
  fftSamples = fftSamplesHeaders;
}

function writeTime(start_time, duration) {
  // currrent_time = getTimeValue();
  // testNumber, direction
  csvTimeStampWriter.writeRecords([{start_time: start_time,
                    direction: direction, duration: duration}]).then(() => {
    console.log('Added time stamp 😀');
  });
}


// test_number<other info>.csv
// time | direction | duration




// const broadcasting_client = dgram.createSocket('udp4');




/*USER CONTROL OF COLLECTING BOOLEAN WITH SOCKET IO*/


//Socket IO:
io.on('connection', function(socket){
  console.log('A user connected');

  socket.on('stop', function(){
      clearInterval(collectionTimer);
      collecting = false;
      endTest(false);
  });

  socket.on('collectQueue', function(clientRequest){
    collectQueue = clientRequest['queue'];
    console.log(collectQueue);

    active = clientRequest['sensors'];

    let totalTime = 0;
    let times = [];
    collectQueue.forEach(function(command){
      totalTime+=command[1];
      times.push(totalTime);
    });

    console.log(totalTime);


    direction = collectQueue[0][0];
    setupCsvWriters();
    // collecting = true;
    let start_time = getTimeValue();

    let j = 0;
    let time = 0;
    collectionTimer = setInterval(function(){
        if (time < totalTime) {
          if (time >= times[j]){
            // move onto next commmand
            if (j > 0){
              writeTime(start_time, times[j]-times[j-1]);

            }
            else{
              writeTime(start_time, times[j]);

            }
            start_time = getTimeValue();


            // endTest(true, true); //end old test
            j += 1;
            direction = collectQueue[j][0]; //setup new one!
            // setupCsvWriters();
          }
        }
        else {
          if (times.length > 1) {
              writeTime(start_time, times[j]-times[j-1]);
          }
          collecting = false;
          endTest(true, true);
          clearInterval(collectionTimer);
          console.log("Trial over. Ready for more data.");

        }
        time++;
    }, 1000);


  });



  // socket.on('collect', function(collectsocket){
  //   /* From the client when a button is clicked a collect message will be sent! */
  //   /* Will include, duration, direction and visible channels as an array */
  //   duration = collectsocket['duration'];
  //   direction = collectsocket['command'];
  //   active = collectsocket['sensors'];
  //   setupCsvWriters();
  //
  //   //Sets up collection queue and whether to loop
  //   //loop = collectsocket['loop']; ONCE LOOP IS ADDED!
  //   //collectQueue = collectsocket['queue'];
  //
  //
  //   let timeLeft = duration;
  //   collecting = true;
  //
  //   let collectionTimer = setInterval(function(){
  //       timeLeft--;
  //       if(timeLeft <= 0){
  //         collecting = false;
  //         clearInterval(collectionTimer);
  //         endTest(true, true);
  //       }
  //   }, 1000);
  //

  //
  //   console.log(collectsocket);
  // });
});





// let win;
// function createWindow () {
//   // Create the browser window.
//   win = new BrowserWindow({ width: 1000, height: 600 });
//
//   // and load the index.html of the app.
//   win.loadFile('public/index.html');
//   win.webContents.openDevTools()
// }
//
// app.on('ready', createWindow);
