var socket = io();
const muStartChannel = 10;
const muEndChannel = 12

var ctx = document.getElementById('neurofeedback').getContext('2d');

var chart = new Chart(ctx, {
  type: 'bar',
  data: {
      labels: ["Left", "Right"],
      datasets: [
        {
          // backgroundColor: ["#3e95cd", "#8e5ea2","#3cba9f","#e8c3b9","#c45850"],
          data: [74,20]
        }
      ]
    },
  options: {
    legend: false,
    scales: {
      yAxes: [{
        ticks: {
          beginAtZero: true,
          max: 5,
        }
      }]
    }
  }
});

function getMu(data) {
  muData = data['data'].slice(muStartChannel, muEndChannel + 1)
  muData = muData.reduce((partial_sum, a) => partial_sum + a) // calculate sum
  muData /= (muEndChannel - muStartChannel)
  return muData
}

socket.on('fft-test', function (data) { // 1st channel ("left")
  // update left datapoint
  chart.data.datasets[0].data[0] = getMu(data)

  chart.update()
});

socket.on('fft-test2', function (data) { // 2nd channel ("right")
  // update right datapoint
  chart.data.datasets[0].data[1] = getMu(data)

  chart.update()
});
