// Configurable
var specHeight = 120;     // pixel height of spectrogram
var specWidth = 450;     // pixel width of spectrogram
var num_samples = 100;    // number of samples we want to display
var num_freqs = 40;       // upper frequency limit of plotting (default 40 hz)
var sgram = [];
var sgram2 = [];
var socket = io();
var h = Math.floor(specHeight/num_freqs);
var w = Math.floor(specWidth/num_samples);
colors = [ '#440154',
'#440658',
'#450b5c',
'#450f61',
'#451465',
'#461969',
'#461e6d',
'#462272',
'#472776',
'#472c7a',
'#46307c',
'#44347e',
'#433880',
'#423c82',
'#404183',
'#3f4585',
'#3e4987',
'#3c4d89',
'#3b518b',
'#39558b',
'#38588c',
'#365c8c',
'#345f8c',
'#33638d',
'#31668d',
'#2f6a8d',
'#2e6d8e',
'#2c718e',
'#2b748e',
'#2a788e',
'#287b8e',
'#277f8e',
'#26828d',
'#25868d',
'#23898d',
'#228d8d',
'#21908d',
'#22938c',
'#22968a',
'#239a89',
'#249d88',
'#24a086',
'#25a385',
'#26a784',
'#26aa82',
'#27ad81',
'#2eb07d',
'#34b47a',
'#3bb776',
'#42bb72',
'#48be6e',
'#4fc16b',
'#55c567',
'#5cc863',
'#65ca5e',
'#6dcc58',
'#76cf53',
'#7fd14d',
'#87d348',
'#90d542',
'#99d83d',
'#a1da37',
'#aadc32',
'#b3dd31',
'#bcde2f',
'#c6e02e',
'#cfe12c',
'#d8e22b',
'#e1e329',
'#ebe528',
'#f4e626',
'#fde725' ];

var t = function(p) {
  p.setup = function() {
    var canvas = p.createCanvas(specWidth, specHeight);
    p.noStroke();
    // p.background(240, 240, 240);  // TODO: so the spectrogram doesn't completely cover the canvas, not sure why. left this here in case you wanted to fix it
    socket.on('fft-test', function (data) {
      newDrawing(data['data']);
    });
  }
  function newDrawing(data){
    // Append to buffer
    sgram.push(data);
    if (sgram.length > num_samples){
      // Remove from buffer once it's reached the desired number of samples
      sgram.splice(0,1);
    }
    //var numExceeded = 0;
    for (var i = 0; i < sgram.length; i++) {
      for (var j = 0; j < num_freqs; j++) {
        var scaled = Math.log(sgram[i][j] + 1)/1.75;        // I picked this function to scale the colors by hand (trial and error)
        var color_idx = Math.min(Math.floor(scaled * colors.length), colors.length - 1);
        p.fill(colors[color_idx]);
        p.rect(i*w,(num_freqs - j - 1)*h, w, h);
        /*if (color_idx == colors.length - 1){              // uncomment this to evaluate the color map
          numExceeded += 1;
        }*/
      }
    }
    //console.log(numExceeded);
  }
}

var myp5 = new p5(t, 'spectrogram-holder');


// Do the same thing for the second spectrogram
var s = function(p) {
  p.setup = function() {
    var canvas = p.createCanvas(specWidth, specHeight);
    p.noStroke();
    socket.on('fft-test2', function (data) {
      newDrawing(data['data']);
    });
  }

  function newDrawing(data){
    sgram2.push(data);
    if (sgram2.length > num_samples){
      sgram2.splice(0,1);
    }
    for (var i = 0; i < sgram2.length; i++) {
      for (var j = 0; j < num_freqs; j++) {
        var scaled = Math.log(sgram2[i][j] + 1)/1.75;
        var color_idx = Math.min(Math.floor(scaled * colors.length), colors.length - 1);
        p.fill(colors[color_idx]);
        p.rect(i*w,(num_freqs - j - 1)*h, w, h);
      }
    }
  }
}

var myp5 = new p5(s, 'spectrogram2-holder');
