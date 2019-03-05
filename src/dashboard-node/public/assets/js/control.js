// Add javascript here, sorry for the mess and repitition!
$(document).ready(function() {
  var timeLeft;
  //Active graphs (Used for sending information to server)
  let active = [1,1,1,1,1,1,1,1];
  var collecting = false;

  //Used for showcasing other dashboard or training dashboard
  $('#tabs li').on('click', function() {
    var tab = $(this).data('tab');

    $('#tabs li').removeClass('is-active');
    $(this).addClass('is-active');

    $('#tab-content div').removeClass('is-active');
    $('div[data-content="' + tab + '"]').addClass('is-active');
  });


  //Used for changing which graphs are shown and updates active array accordingly
  $('.graph-controls input').mousedown(function() {
    var input = $(this).attr("name");
    var indexChar = input[input.length-1];
    var index = parseInt(indexChar, 10)-1;

    $('#media-sensor' + (index+1)).toggleClass("hide");
    if(active[index] == 1){
      active[index] = 0;
    }
    else{
      active[index] = 1;
    }
  });


  //Sample Data and Graphs Please Ignore this block:
  // var data = [
  //   { label: 'Layer 1', values: [ {x: 0, y: 0}, {x: 1, y: 1}, {x: 2, y: 2} ] },
  //   { label: 'Layer 2', values: [ {x: 0, y: 0}, {x: 1, y: 1}, {x: 2, y: 4} ] }
  // ];
  // var sinLayer = {label: 'sin', values: []},
  //   cosLayer = {label: 'cos', values: []}
  //
  //   for (var x = 0; x <= 2*Math.PI; x += Math.PI / 64) {
  //     sinLayer.values.push({ x: x, y: Math.sin(x) + 1 });
  //     cosLayer.values.push({ x: x, y: Math.cos(x) + 1 });
  //   }
  // var sensorChart2 = $('#sensor2').epoch({
  //     type: 'area',
  //     data: [sinLayer, cosLayer],
  //     axes: ['left', 'right', 'bottom']
  // });
  // var sensorChart3 = $('#sensor3').epoch({
  //     type: 'area',
  //     data: [sinLayer, cosLayer],
  //     axes: ['left', 'right', 'bottom']
  // });
  // var sensorChart4 = $('#sensor4').epoch({
  //     type: 'area',
  //     data: [sinLayer, cosLayer],
  //     axes: ['left', 'right', 'bottom']
  // });
  // var sensorChart5 = $('#sensor5').epoch({
  //     type: 'area',
  //     data: [sinLayer, cosLayer],
  //     axes: ['left', 'right', 'bottom']
  // });
  // var sensorChart6 = $('#sensor6').epoch({
  //     type: 'area',
  //     data: [sinLayer, cosLayer],
  //     axes: ['left', 'right', 'bottom']
  // });
  // var sensorChart7 = $('#sensor7').epoch({
  //     type: 'area',
  //     data: [sinLayer, cosLayer],
  //     axes: ['left', 'right', 'bottom']
  // });
  // var sensorChart8 = $('#sensor8').epoch({
  //     type: 'area',
  //     data: [sinLayer, cosLayer],
  //     axes: ['left', 'right', 'bottom']
  // });


  //Depending on range, changes the countdown text and vice versa
  var range = $('.input-range').val();
  $(".input-range").on('input', function(){
      if(!collecting){
        range = $(".input-range").val();
        $(".timer").val(range);
      } else{
        //Maintains value from timer!
        range = $(".timer").val();
        $(".input-range").val(range);
      }
  });

  $(".timer").on('input', function(){
    if(!collecting){
      range = $(".timer").val();
      $(".input-range").val(range);
    }
    else {
      //Maintains value from range!
      range = $(".input-range").val();
      $(".timer").val(range);
    }
  });

  /* IMPORTANT BLOCK FOR DATA COLLECTION! */
  // If one of the collection buttons are clicked does the following:
  $(".selection").click(function() {
    if(!collecting){
      //Gets the button that was clicked
      var clicked = $(this);

      //Sets the duration from the value that was present
      var duration = $(".input-range").val();

      //If the duration is not 0 then does the following:
      if(duration != 0){
          collecting = true;
          //Gets the proper direction and sends left/right/rest, the duration and active sensors to server
          if(clicked.is('.direction-left')){
            socket.emit("collect", {command: "left", duration: duration, sensors: active});
          }
          else if(clicked.is('.direction-right')){
            socket.emit("collect", {command: "right", duration: duration, sensors: active});
          }
          else if(clicked.is('.direction-rest')){
            socket.emit("collect", {command: "rest", duration: duration, sensors: active});
          }

        //Allows the countdown to work ***VERY crude currently, need to fix! ***
        timeLeft = duration;

        $('.selection').addClass('active');
        $(".selection").removeClass('imagery-inactive');
        var collectionTimer = setInterval(function(){
          timeLeft -= 1;
          $('.timer').val(timeLeft);
          $(".input-range").val(timeLeft);

          if(timeLeft <= 0){
            //END OF COLLECTION DUTIES:
            clearInterval(collectionTimer);
            timeLeft = duration;
            if (clicked.hasClass('toggle')) {
              clicked.removeClass('toggle');
            }
            else {
              $('.selection').removeClass('toggle');
              clicked.addClass('toggle');

            }
            $('.selection').removeClass('active');
            $('.selection').addClass('imagery-inactive');
            collecting = false;
            $(".timer").val(10);
            $(".input-range").val(10);
          }
        }, 1000);

        //Adds a toggle class for the button that was clicked
        if (clicked.hasClass('toggle')) {
          clicked.removeClass('toggle');
        }
        else {
          $('.selection').removeClass('toggle');
          clicked.addClass('toggle');
        }


      }
    }

  })

  //GRAPHING!
  function getTimeValue() {
    var dateBuffer = new Date();
    var Time = dateBuffer.getTime();
    return Time;
  }
  var charts = [], lines = [];
  var colors = ["#6dbe3d","#c3a323","#EB9486","#787F9A","#97A7B3","#9F7E69","#d97127", "#259188"]

  for(i = 0; i < 8; i++) {
    charts.push(new SmoothieChart({grid:{fillStyle:'transparent'},
                                   labels:{fillStyle:'transparent'},
                                   maxValue: 400,
                                   minValue: -400}));
    charts[i].streamTo(document.getElementById('smoothie-chart-' + (i+1)), 500);
    lines.push(new TimeSeries());
  }

  let timeElapsed = new Date().getTime()
  socket.on('timeseries', function(timeseries) {
      // console.log(channelOne.data);
      for(i = 0; i < 8; i++){
        lines[i].append(new Date().getTime(), timeseries['eeg']['data'][i]);
      }

      // if (counter == 10) {
        // let newData = (new Date().getTime(), timeseries['eeg']['data'][0]);
        // console.log(timeseries['eeg']['data'][0])
        // console.log(counter)
        if (new Date().getTime() -  timeElapsed > 1000){
          for(i = 0; i < 8; i++){
            charts[i].addTimeSeries(lines[i], {lineWidth:2,
                                               strokeStyle:colors[i]});
            timeElapsed = new Date().getTime();
            lines[i] = new TimeSeries();
          }
        }

          // counter = 0;
      // } else {
          // ends with 0
          // counter++;
      // }
      // console.log(timeseries['eeg']['data'][0]);

      // console.log(channelOne.data + " and time: " + getTimeValue());
      // sensorChart1.push(newData);
  });

  // setInterval(function() {
  //   line.append(new Date().getTime(), Math.random())
  // }, 100);
  //
  // chart.addTimeSeries(line, {lineWidth:2, strokeStyle:'#6834c8'});



//
//
// var sensorChart1 = $('#sensor1').epoch({
//     type: 'time.area',
//     data: lineChartData,
//     axes: ['left', 'right', 'bottom']
// });
// let counter = 0;
// let previousTime = 0;

  $('#stop').click(function(){
    socket.emit("stop", {});
    timeLeft = 0;

  });

});
