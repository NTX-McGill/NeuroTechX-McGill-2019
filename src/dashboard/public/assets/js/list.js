$(document).ready(function() {
  console.log("List javascript loaded!");
  var list = document.getElementById('commandBank');
  var trialName;
  //SETS ACTIVE TO ALL OF THEM FOR NOW!
  var active = [1,1,1,1,1,1,1,1];
  //Made global so stop button can clear it
  var collectionTimer = null;

  //To remove an element from the queue
  $("#commandList").on("click",".remove",function(){
    console.log("here?");
    event.preventDefault();
    $(this).parent().remove();
  });

  //On left, right, or rest button click!
  $(".selection").click(function() {
    var clicked = $(this);
    var duration = $(".timer").val();

    if(clicked.is('.direction-left')){
      //Make list item with left and duration!
      $("#commandList").append($("<div class='list-group-item tinted' data-direction='Left' data-duration='" + duration + "'><i class='fas fa-arrows-alt handle'></i> Left " + duration + "s <a href='#' class='remove'>REMOVE ME</a></div>"));
    }
    else if(clicked.is('.direction-right')){
      $("#commandList").append($("<div class='list-group-item tinted' data-direction='Right' data-duration='" + duration + "'><i class='fas fa-arrows-alt handle'></i> Right " + duration + "s <a href='#' class='remove'>REMOVE ME</a></div>"));

    }
    else if(clicked.is('.direction-rest')){
      $("#commandList").append($("<div class='list-group-item tinted' data-direction='Rest' data-duration='" + duration + "'><i class='fas fa-arrows-alt handle'></i> Rest " + duration + "s <a href='#' class='remove'>REMOVE ME</a></div>"));
    }
    else{

      // var loop = $("#loop").prop("checked") //returns true or false!
      //if 'else', must be collect!

      //Amount of elements in the queue

      //Flashes bright green briefly
      if((count != 0) && !$( "#btn-collect" ).hasClass( "btn-danger" )){ //Non empty list and not already clicked
          var queue = [];
          var count = $("#commandList div").length;
          trialName = $('#trial-name').val();
          $('#btn-collect').toggleClass('btn-danger');
          $('#btn-collect').html("Stop");
          $('#command-display').toggleClass('command-display-flash');
          //For each element in the queue, push their direction and duration
          $('#commandList').children('div').each(function () {
              var itemDuration = $(this).data("duration");
              var itemDirection = $(this).data("direction")
              queue.push([itemDirection, itemDuration]);
        });

        // if(loop){
        //   queue.push(["loop", 0]);
        // }

        //Finally emits a collectQueue!

        //Gives the queue array with the direcions/durations and active sensors
        socket.emit("collectQueue", {queue: queue, sensors: active, trialName: trialName});

        let totalTime = 0;
        let times = [];
        /* Creates an array with cumulative times:
            Time 1: 5
            Time 2: 5
            Time 3: 10

            times = [5, 10, 20]
        */
        queue.forEach(function(command){
          totalTime+=command[1];
          times.push(totalTime);
        });


        direction = queue[0][0];
        //This is the direction of the first element
        let durationLeft = times[0] - 0;//Do we need - 0?

        //Sets display to first elements command/time
        $('#current-command').html(direction);
        $('#collectTime').html(durationLeft);
        let j = 0;
        let time = 0;
        // $('#command-display').addClass('command-display-use');

        //Controlling the timer.
        collectionTimer = setInterval(function(){
            if (time < totalTime) {
                $('#command-display').addClass('command-display-use')
                if($('#command-display').hasClass('command-display-flash')){
                  $('#command-display').removeClass('command-display-flash')
                }

              if (time >= times[j]){
              //This means we've gotten to the end of element j's duration
                console.log("next command");
                $('#command-display').addClass('command-display-flash');
                j += 1;
                direction = queue[j][0];
                $('#current-command').html(direction); //Setup direction again
              }
              //If we're not at end of duration, decrement time
              durationLeft = times[j] - time;

              $('#collectTime').html(durationLeft);
              time++;
            }
            else {
                $('#command-display').removeClass('command-display-use');
                $('#btn-collect').toggleClass('btn-danger');
                $('#btn-collect').html('Collect');
                $('#current-command').html("--");
                $('#collectTime').html("N/A");
                clearInterval(collectionTimer);
            }
        }, 1000);

      }
      else if($('#btn-collect').hasClass('btn-danger')){
          console.log("danger danger");
          clearInterval(collectionTimer);
          socket.emit("stop");
          $('#btn-collect').toggleClass('btn-danger');
          $('#btn-collect').html("Collect");
          $('#current-command').html("--");
          $('#collectTime').html("N/A");
          $('#command-display').removeClass('command-display-use');

      }
      else{
        console.log("Empty list nice try!");
      }

    }

  });

});
