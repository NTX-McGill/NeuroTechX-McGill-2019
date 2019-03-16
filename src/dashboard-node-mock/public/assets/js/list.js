$(document).ready(function() {
  console.log("List javascript loaded!");
  var list = document.getElementById('commandBank');

  //SETS ACTIVE TO ALL OF THEM FOR NOW!
  var active = [1,1,1,1,1,1,1,1];

  //ON REMOVE!
  $("#commandList").on("click",".remove",function(){
    console.log("here?");
    event.preventDefault();
    $(this).parent().remove();
  });

  //On button click!
  $(".selection").click(function() {
    var clicked = $(this);
    var duration = $(".timer").val();

    if(clicked.is('.direction-left')){
      //Make list item with left and duration!
      $("#commandList").append($("<div class='list-group-item tinted' data-direction='left' data-duration='" + duration + "'><i class='fas fa-arrows-alt handle'></i> Left " + duration + "s <a href='#' class='remove'>REMOVE ME</a></div>"));

    }
    else if(clicked.is('.direction-right')){
      $("#commandList").append($("<div class='list-group-item tinted' data-direction='right' data-duration='" + duration + "'><i class='fas fa-arrows-alt handle'></i> Right " + duration + "s <a href='#' class='remove'>REMOVE ME</a></div>"));

    }
    else if(clicked.is('.direction-rest')){
      $("#commandList").append($("<div class='list-group-item tinted' data-direction='rest' data-duration='" + duration + "'><i class='fas fa-arrows-alt handle'></i> Rest " + duration + "s <a href='#' class='remove'>REMOVE ME</a></div>"));
    }
    else{
      // var loop = $("#loop").prop("checked") //returns true or false!
      //Must be collect!
      var queue = [];
      var count = $("#commandList div").length;
      console.log(count);

      if(count != 0){ //Non empty list!

        $('#commandList').children('div').each(function () {
            var itemDuration = $(this).data("duration");
            var itemDirection = $(this).data("direction")

            //
            queue.push([itemDirection, itemDuration]);
        });

        // if(loop){
        //   queue.push(["loop", 0]);
        // }

        //Finally emits a collectQueue!


        socket.emit("collectQueue", {queue: queue, sensors: active});

        let totalTime = 0;
        let times = [];
        queue.forEach(function(command){
          totalTime+=command[1];
          times.push(totalTime);
        });

        direction = queue[0][0];
        let durationLeft = times[0] - 0;

        $('#collectDirection').html(direction);
        $('#collectTime').html(durationLeft);
        let j = 0;
        let time = 0;

        let collectionTimer = setInterval(function(){
            if (time < totalTime) {
              if (time >= times[j]){
                j += 1;
                direction = queue[j][0];
                $('#collectDirection').html(direction); //setup new one!
              }
              durationLeft = times[j] - time;
              $('#collectTime').html(durationLeft);
              time++;
            }
            else {
              $('#collectDirection').html("NONE");
              $('#collectTime').html("N/A");
              clearInterval(collectionTimer);
            }
        }, 1000);

      }
      else{
        console.log("Empty list nice try!");
      }

    }

  });

});
