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
    tooltips: {enabled:false},
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
var config = {
            type: 'line',
            data: {
                datasets: [{
                    label: 'Right Probability',
                    backgroundColor: '#4e73df',
                    borderColor: '#4e73df',
                    fill: false,
                    lineTension: 0,
                    // borderDash: [8, 4],
                    data: []
                }]
            },
            options: {
                title: {
                    display: false,
                    text: 'Push data feed sample'
                },
                scales: {
                    xAxes: [{
                        type: 'realtime',
                        realtime: {
                            duration: 20000,
                            delay: 2000,
                        }
                    }],
                    yAxes: [{
                        scaleLabel: {
                            display: true,
                            labelString: 'value'
                        }
                    }]
                },
                tooltips: {
                    mode: 'nearest',
                    intersect: false
                },
                hover: {
                    mode: 'nearest',
                    intersect: false
                }
            }
        };
var historyChart = new Chart(ctx2, config);

var x_val = 10;
socket.on('ML graphs', function (data) { // 1st channel ("left")
  // update left datapoint
  console.log('received data from ML')
  chart.data.datasets[0].data[0] = data['left-prob'];
  chart.data.datasets[0].data[1] = data['right-prob'];
  historyChart.config.data.datasets[0].data.push({
    x: Date.now(),
    y: data['right-prob']
  })
  chart.update()
  x_val++
  historyChart.update( {
    preservation: true
  })
});
