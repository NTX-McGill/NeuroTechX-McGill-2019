$(document).ready(function() {
  // For production dashboard
  $('#startProduction').on('click', function(){
    $('#startProduction').toggleClass('btn-danger');
    if ($('#startProduction').hasClass('btn-danger')){
      socket.emit("production", {on: true});
      $('#startProduction').html("Stop &nbsp; <i class='fas fa-stop fa-sm text-white'></i>");
    }
    else {
      socket.emit("production", {on: false});
      $('#startProduction').html("Start &nbsp; <i class='fas fa-play fa-sm text-white'></i>");
    }
  });
});
