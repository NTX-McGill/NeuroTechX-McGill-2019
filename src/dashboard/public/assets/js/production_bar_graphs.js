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


socket.on('data from ML', function (data) { // 1st channel ("left")
  // update left datapoint
  console.log(data)
  chart.data.datasets[0].data[0] = data['response']['left-prob'];
  chart.data.datasets[0].data[1] = data['response']['right-prob'];
  historyChart.data.datasets[0].data.push(data['response']['right-prob']);
  chart.update()
});
