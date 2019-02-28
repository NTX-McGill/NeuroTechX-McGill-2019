//Stores express as a function call variable
const express = require('express')
const app = express()

//Sets static directory as public
app.use(express.static(__dirname + '/public'));

app.get('/', (req, res) => {
  res.send('index');
});

app.listen(3000);
console.log('Listening on Port 3000!')
