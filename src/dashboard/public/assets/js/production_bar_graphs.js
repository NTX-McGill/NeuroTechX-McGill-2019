var socket = io();
const muStartChannel = 10;
const muEndChannel = 12

var ctx = document.getElementById('machine-learning-bar-graph').getContext('2d');

var chart = new Chart(ctx, {
  type: 'bar',
  data: {
      labels: ["Left", "Right"],
      datasets: [
        {
          // backgroundColor: ["#3e95cd", "#8e5ea2","#3cba9f","#e8c3b9","#c45850"],
          data: [0.5,0.3]
        }
      ]
    },
  options: {
    legend: false,
    scales: {
      yAxes: [{
        ticks: {
          beginAtZero: true,
          max: 1,
        }
      }]
    }
  }
});

var ctx2 = document.getElementById('machine-learning-history').getContext('2d');

var historyChart = new Chart(ctx2, {
  type: 'line',
  data: {
      datasets: [
        {
          // backgroundColor: ["#3e95cd", "#8e5ea2","#3cba9f","#e8c3b9","#c45850"],
          data: [{
            x: 1,
            y: 0.3
          }, {
            x: 2,
            y: 0.5
          }, {
            x: 3,
            y: 0.7
          }]
        }
      ]
    },
  options: {
    legend: false,
    scales: {
      xAxes: [{
                type: 'realtime',
                position: 'bottom',
                refresh: 1000,      // onRefresh callback will be called every 1000 ms
                    delay: 1000,        // delay of 1000 ms, so upcoming values are known before plotting a line
                    pause: false,       // chart is not paused
                    ttl: undefined,
            }],
      yAxes: [{
        ticks: {
          beginAtZero: true,
          max: 1,
        }
      }]
    }
  }
});

var x_val = 10;
socket.on('ML graphs', function (data) { // 1st channel ("left")
  // update left datapoint
  console.log('received data from ML')
  chart.data.datasets[0].data[0] = data['left-prob'];
  chart.data.datasets[0].data[1] = data['right-prob'];
  historyChart.data.datasets[0].data.push({
    x: x_val,
    y: x_val/50
  })
  chart.update()
  x_val++
  historyChart.update( {
    preservation: true
  })
});
