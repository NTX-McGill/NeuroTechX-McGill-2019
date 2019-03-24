$(document).ready(function() {
  // For production dashboard
  $('#startProduction').on('click', function(){
    socket.emit("production", {on: true});
    $('#startProduction').toggleClass('btn-danger');
    if ($('#startProduction').hasClass('btn-danger')){
        $('#startProduction').html("Stop &nbsp; <i class='fas fa-stop fa-sm text-white'></i>");
    }
    else{
        $('#startProduction').html("Start &nbsp; <i class='fas fa-play fa-sm text-white'></i>");
    }
  });
});
