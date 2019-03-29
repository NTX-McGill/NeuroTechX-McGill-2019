
/**
 * Copyright 2017 Google Inc. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

var config = {
  "mapsApiKey": "AIzaSyDUdvR-J8U5-N7DKxM4sROaKbYNN4UdRXY",
  "firebaseApiKey": "AIzaSyD_neZBWffT7yRpj6RoGfe6AyQwVBM6vEc",
  "firebaseDatabaseURL": "https://wheelchair-e5f28.firebaseio.com",
};

var app = firebase.initializeApp({
  apiKey: config.firebaseApiKey,
  databaseURL: config.firebaseDatabaseURL,
});

var database = app.database();

var map;

var counter = 0;
var previousMarker;

database.ref('raw-locations').on('value', function(data) {
    console.log('value');
  $('#loading').hide();

  var transports = data.val();


  transports = Object.keys(transports).map(function(id) {
      var transport = transports[id][0];
    transport.id = id;
    transport.power = Math.round(transport.power);
    transport.time = moment(transport.time).fromNow();
    transport.map = 'https://maps.googleapis.com/maps/api/staticmap?size=800x500'
        + '&markers=color:blue%7Clabel:' + transport.id + '%7C' + transport.lat
        + ',' + transport.lng + '&key=' + config.mapsApiKey + '&zoom=15';
    return transport;
  });

  console.log(transports);
  if(counter == 0){

        map = new google.maps.Map(document.getElementById('map'), {
          center: {lat: transports[0].lat, lng: transports[0].lng},
          zoom: 18
      });
  }

  var image = {
      url: 'wheelchair-marker-icon.png',
      scaledSize: new google.maps.Size(60, 60)
  };
  if(previousMarker != null){
      previousMarker.setMap(null);
  }
  var marker = new google.maps.Marker({position: {lat: transports[0].lat, lng: transports[0].lng}, map: map, icon: image});
  map.setCenter({lat: transports[0].lat, lng: transports[0].lng})
  previousMarker = marker;
  counter++;
});
