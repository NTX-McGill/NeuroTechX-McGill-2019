const dgram = require('dgram');
const events = require('events');
const express = require('express');
const app = express();
const server = app.listen(3000);
const io = require('socket.io').listen(server);

const Cyton = require("@openbci/cyton");
const k = require("@openbci/utilities").constants;
const fs = require('fs');

var settings = JSON.parse(fs.readFileSync('data_settings.json', 'utf8'));
console.log("Currently running on these settings: \n" + settings);
let testNumber = settings['testNumber'];

function getTimeValue() {
  var dateBuffer = new Date();
  var Time = dateBuffer.getTime();
  return Time;
}

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

UDPClient.prototype.onListening = function() {
  console.log('Listening for data...');
};

UDPClient.prototype.onMessage = function(msg) {
  this.events.emit('sample', JSON.parse(msg.toString()));
};

var client = new UDPClient(12345, "127.0.0.1");

client.events.on('sample', function(data) {
  if(data['type'] == 'fft'){
    io.sockets.emit('fft', data);
  }
  else{
    let time = getTimeValue();
    // console.log(data);
    io.sockets.emit('timeseries', {'time': time, 'eeg': data});
  }
});


//Socket IO:
io.on('connection', function(socket){
  console.log('a user connected');
  socket.on('collect', function(collectsocket){
    //Request sent for collection
    console.log(collectsocket);
  });
});



//Sets static directory as public
app.use(express.static(__dirname + '/public'));

app.get('/', (req, res) => {
  res.send('index');
});


console.log('Listening on Port 3000!')
