const dgram = require('dgram');
const events = require('events');
const express = require('express');
const app = express();
const server = app.listen(3000);
const io = require('socket.io').listen(server);
const fs = require('fs');


const createCsvWriter = require('csv-writer').createObjectCsvWriter;
var csvWriter;

//Will determine if collecting and sending to file currently.
//Other values will only be updated if collecting is true!
var collecting = false;
var duration = 0;
var direction = "none";
var active = [];
var samples = [];


/* These are manual settings that we can use to keep track of testNumber as an example */
var settings = JSON.parse(fs.readFileSync('data_settings.json', 'utf8'));
console.log("Currently running on these settings: \n" + settings);
let testNumber = settings['testNumber'];

/* Gets the current time */
function getTimeValue() {
  var dateBuffer = new Date();
  var Time = dateBuffer.getTime();
  return Time;
}

/* Sets the csvwriters to the correct paths! */
function setupCSVWriters(){
    for(i = 0; i < 8; i++){
      samples.push([]);
    }
    csvWriter = createCsvWriter({
          path: __dirname + '/data/test-' + testNumber + '-' + direction + '.csv',
          header: [
              {id: 'time', title: 'TIME'},
              {id: 'channel1', title: 'CHANNEL 1'},
              {id: 'channel2', title: 'CHANNEL 2'},
              {id: 'channel3', title: 'CHANNEL 3'},
              {id: 'channel4', title: 'CHANNEL 4'},
              {id: 'channel5', title: 'CHANNEL 5'},
              {id: 'channel6', title: 'CHANNEL 6'},
              {id: 'channel7', title: 'CHANNEL 7'},
              {id: 'channel8', title: 'CHANNEL 8'}
          ],
          append: true
    });
}


/* When data is collecting, samples will also write to file! */
function appendSample(data){
  channelData = []
  for (i = 0; i < 8; i++) {
    if (active[i]) {
        channelData[i] = data['data']['data'][i];
    }
    else {
      channelData[i] = null;
    }
  }
  sampleToPush = {time: data['time'],
                  channel1: channelData[0],
                  channel2: channelData[1],
                  channel3: channelData[2],
                  channel4: channelData[3],
                  channel5: channelData[4],
                  channel6: channelData[5],
                  channel7: channelData[6],
                  channel8: channelData[7]
                }
  samples.push(sampleToPush);
}

/* Updates test number on data_settings file */
function endTest(){
  settings['testNumber'] += 1;
  let settingsString = JSON.stringify(settings);

  fs.writeFile('data_settings.json', settingsString, 'utf8', function(err){
    if (err) throw err;
    console.log('Updated Test Number!');
    testNumber = settings['testNumber'];
  });
  // [ {ENTRY 1}, {ENTRY 2}]
  csvWriter.writeRecords(samples).then(() => {
    console.log('Added some samples');
  });

// time | channel1 | chanel2 | ... | channel8


}

/* Creates a UDP client to listen to the OpenBCI GUI */
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

/* Prints listening */
UDPClient.prototype.onListening = function() {
  console.log('Listening for data...');
};

/* On message from OpenBCI UDP, emits an event called sample for further classification */
UDPClient.prototype.onMessage = function(msg) {
  this.events.emit('sample', JSON.parse(msg.toString()));
};

/* Creates UDP Client */
var client = new UDPClient(12345, "127.0.0.1");

/* On sample received runs the following code: */
client.events.on('sample', function(data) {
  // console.log(data);
  /* If it is a fast fourier transform or timeseries value does the following */
  if(data['type'] == 'fft'){

    /* Sends a socket to the client called fft that has all the data */
    io.sockets.emit('fft', data);
  }
  else{
    let time = getTimeValue();
    if(collecting){
      //Then we want to write to proper test file!
      let toWrite = {'time': time, 'data': data};
      appendSample(toWrite);
    }


    /* Sends a socket to the client called timeseries that has the data as well as time */
    /* Data Format:
                  {
                    'time': time,
                    'eeg': {
                              'data': [0.5,3,-5,40,5,32,8,1] data[index] is the eeg value at sensor-index
                           }
                  }
    */
    io.sockets.emit('timeseries', {'time': time, 'eeg': data});
  }
});


//Socket IO:
io.on('connection', function(socket){
  console.log('a user connected');
  socket.on('collect', function(collectsocket){
    /* From the client when a button is clicked a collect message will be sent! */
    /* Will include, duration, direction and visible channels as an array */
    duration = collectsocket['duration'];
    direction = collectsocket['command'];
    active = collectsocket['sensors'];
    setupCSVWriters();
    let timeLeft = duration;
    collecting = true;

    let collectionTimer = setInterval(function(){
        timeLeft--;
        if(timeLeft <= 0){
          collecting = false;
          clearInterval(collectionTimer);
          endTest();
        }
    }, 1000);

    console.log(collectsocket);
  });
});


//Sets static directory as public
app.use(express.static(__dirname + '/public'));

app.get('/', (req, res) => {
  res.send('index');
});


console.log('Listening on Port 3000!')
