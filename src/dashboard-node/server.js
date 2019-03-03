const express = require('express');
const app = express();
const server = app.listen(3000);
const io = require('socket.io').listen(server);

const Cyton = require("@openbci/cyton");
const k = require("@openbci/utilities").constants;

function getTimeValue() {
  var dateBuffer = new Date();
  var Time = dateBuffer.getTime();
  return Time;
}

let previousTime = 0;
let counter = 0;

const ourBoard = new Cyton();
ourBoard
  .connect(k.OBCISimulatorPortName) // This will set `simulate` to true
  .then(boardSerial => {

    return ourBoard.streamStart();
  })
  .catch(err => {
    /** Handle connection errors */
    console.log(ourBoard.getInfo());
  });
ourBoard.on('sample',(sample) => {
    let timern = getTimeValue();
    if (timern - previousTime > 1000) {
        // console.log(counter);
        previousTime = getTimeValue();
        counter = 0;
    } else {
        // ends with 0
        counter++;

    }
    let channelName = "Channel1";
    io.sockets.emit(channelName, {data: sample.channelData[0].toFixed(8)*1000000});
});

ourBoard.on('error', (callback) => {
  console.log(callback);
});

//Socket IO:
io.on('connection', function(socket){
  console.log('a user connected');
});

//Sets static directory as public
app.use(express.static(__dirname + '/public'));

app.get('/', (req, res) => {
  res.send('index');
});


console.log('Listening on Port 3000!')
