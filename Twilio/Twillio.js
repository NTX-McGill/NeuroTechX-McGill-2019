// Download the helper library from https://www.twilio.com/docs/node/install
// Your Account Sid and Auth Token from twilio.com/console
// DANGER! This is insecure. See http://twil.io/secure

const accountSid = 'ACc5c51487eada9738d03e22544149c704';
const authToken = 'a72d787007075c47d1b11e7a6b76f0ca';
const client = require('twilio')(accountSid, authToken);

client.messages
      .create({
         body: 'McAvoy or Stewart? These timelines can get so confusing.',
         from: '+14388069184',
         statusCallback: 'http://postb.in/1234abcd',
         to: '+15149173648'
       })
      .then(message => console.log(message.sid));
