// Add javascript here, sorry for the mess and repitition!
$(document).ready(function() {
  //Timer:

  //Active graphs
  let active = [1,1,1,1,1,1,1,1];

  $('#tabs li').on('click', function() {
    var tab = $(this).data('tab');

    $('#tabs li').removeClass('is-active');
    $(this).addClass('is-active');

    $('#tab-content div').removeClass('is-active');
    $('div[data-content="' + tab + '"]').addClass('is-active');
  });

  $('.graph-controls input').mousedown(function() {

    var input = $(this).attr("name");
    var indexChar = input[input.length-1];
    var index = parseInt(indexChar, 10)-1;


    $('article[id="media-' + input + '"]').toggleClass("hide");
    if(active[index] == 1){
      active[index] = 0;
    }
    else{
      active[index] = 1;
    }


  });

  var data = [
    { label: 'Layer 1', values: [ {x: 0, y: 0}, {x: 1, y: 1}, {x: 2, y: 2} ] },
    { label: 'Layer 2', values: [ {x: 0, y: 0}, {x: 1, y: 1}, {x: 2, y: 4} ] }
  ];
  var sinLayer = {label: 'sin', values: []},
    cosLayer = {label: 'cos', values: []}

    for (var x = 0; x <= 2*Math.PI; x += Math.PI / 64) {
      sinLayer.values.push({ x: x, y: Math.sin(x) + 1 });
      cosLayer.values.push({ x: x, y: Math.cos(x) + 1 });
    }





  var sensorChart2 = $('#sensor2').epoch({
      type: 'area',
      data: [sinLayer, cosLayer],
      axes: ['left', 'right', 'bottom']
  });
  var sensorChart3 = $('#sensor3').epoch({
      type: 'area',
      data: [sinLayer, cosLayer],
      axes: ['left', 'right', 'bottom']
  });
  var sensorChart4 = $('#sensor4').epoch({
      type: 'area',
      data: [sinLayer, cosLayer],
      axes: ['left', 'right', 'bottom']
  });
  var sensorChart5 = $('#sensor5').epoch({
      type: 'area',
      data: [sinLayer, cosLayer],
      axes: ['left', 'right', 'bottom']
  });
  var sensorChart6 = $('#sensor6').epoch({
      type: 'area',
      data: [sinLayer, cosLayer],
      axes: ['left', 'right', 'bottom']
  });
  var sensorChart7 = $('#sensor7').epoch({
      type: 'area',
      data: [sinLayer, cosLayer],
      axes: ['left', 'right', 'bottom']
  });
  var sensorChart8 = $('#sensor8').epoch({
      type: 'area',
      data: [sinLayer, cosLayer],
      axes: ['left', 'right', 'bottom']
  });

  var range = $('.input-range').val();

  $(".input-range").on('input', function(){
      range = $(".input-range").val();
      $(".timer").val(range);
  });

  $(".timer").on('input', function(){
      range = $(".timer").val();
      $(".input-range").val(range);
  });

  // Timer and states and emitting collection to node server!
$(".selection").click(function() {
  console.log("clicked?");
  var clicked = $(this);
  var duration = $(".input-range").val();
  if(duration != 0){
    if(clicked.is('.direction-left')){
      socket.emit("collect", {command: "left", duration: duration, sensors: active});
    }
    else if(clicked.is('.direction-right')){
      socket.emit("collect", {command: "right", duration: duration, sensors: active});
    }
    else if(clicked.is('.direction-rest')){
      socket.emit("collect", {command: "rest", duration: duration, sensors: active});
    }

  let timeLeft = duration;
  let collectionTimer = setInterval(function(){
    $('.timer').val(timeLeft);
    timeLeft -= 1;
    if(timeLeft <= 0){
      clearInterval(collectionTimer);
      timeLeft = duration;
    }
  }, 1000);

  if (clicked.hasClass('toggle')) {
    clicked.removeClass('toggle');
  }
  else {
    $('.selection').removeClass('toggle');
    clicked.addClass('toggle');
  }
  }


})


});
