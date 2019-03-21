/*
IBT-2 Motor Control Board driven by Arduino.
 
Speed and direction controlled by a potentiometer attached to analog input 0.
One side pin of the potentiometer (either one) to ground; the other side pin to +5V
 
Connection to the IBT-2 board:
IBT-2 pins 5 (R_IS) and 6 (L_IS) not connected
*/
 
int SENSOR_PIN = 0; // center pin of the potentiometer
 
int LPWM_Output_L = 5; // Arduino PWM output pin 5; connect to IBT-2 pin 1 (RPWM)
int RPWM_Output_L = 6; // Arduino PWM output pin 6; connect to IBT-2 pin 2 (LPWM)


int LPWM_Output_R = 10; // Arduino PWM output pin 5; connect to IBT-2 pin 1 (RPWM)
int RPWM_Output_R = 9; // Arduino PWM output pin 6; connect to IBT-2 pin 2 (LPWM)
 
void setup()
{
  pinMode(RPWM_Output_L, OUTPUT);
  pinMode(LPWM_Output_L, OUTPUT);
  pinMode(RPWM_Output_R, OUTPUT);
  pinMode(LPWM_Output_R, OUTPUT);

  //for data reeding
  Serial.begin(9600);
}


void left(int sensorValue){
  if (sensorValue < 0)
  {
    // reverse rotation
    int reversePWM = -(sensorValue);
    analogWrite(LPWM_Output_L, 0);
    analogWrite(RPWM_Output_L, reversePWM);
  }
  else
  {
    // forward rotation
    int forwardPWM = (sensorValue - 512);
    analogWrite(LPWM_Output_L, forwardPWM);
    analogWrite(RPWM_Output_L, 0);
  }
}

void right(int sensorValue){
  if (sensorValue < 0)
  {
    // reverse rotation
    int reversePWM = -(sensorValue);
    analogWrite(LPWM_Output_R, 0);
    analogWrite(RPWM_Output_R, reversePWM);
  }
  else
  {
    // forward rotation
    int forwardPWM = (sensorValue - 512);
    analogWrite(LPWM_Output_R, forwardPWM);
    analogWrite(RPWM_Output_R, 0);
  }
}

//Code by Shawn Vosburg (Robotics team)
//Written for NeuroTech
//Declare the global var
byte state = 'S';
float timeout = 250; //number of milliseconds for ramping up. 
signed int maxMotorSpeed = 200;//OUT O 256
signed int lMotorSpeed = 0,rMotorSpeed = 0;



void loop() {

  //check to see if data is present. 
  if(Serial.available() > 0)
  {
    //====================================================
    //READ DATA FROM SERIAL PORT
    //====================================================
    state = Serial.read();
    //Serial.write(state); //Send back data for debugging. 
  
    //====================================================
    //SMOOTH ACCELERATION OF MOTORS
    //====================================================
    unsigned long start= millis();
    signed long deltaT = 0;
    signed int lPrevSpeed = lMotorSpeed;
    signed int rPrevSpeed = rMotorSpeed;
    switch(state)
    {
      
    //CASE 1: FORWARD
    case('F'):
      do
      {
        deltaT = millis() - start;
        rMotorSpeed = min(maxMotorSpeed, (deltaT/timeout * maxMotorSpeed + rPrevSpeed));
        lMotorSpeed = min(maxMotorSpeed, (deltaT/timeout * maxMotorSpeed + lPrevSpeed));
      }
      while((lMotorSpeed != maxMotorSpeed || rMotorSpeed != maxMotorSpeed) && deltaT <= timeout);
      break;
      
    //CASE 2: STOP
    case('S'):
      do
      {
        deltaT = millis() - start;
        //rightspeed update
        if(rPrevSpeed <0)                    rMotorSpeed = min(0, int(deltaT/timeout * maxMotorSpeed + rPrevSpeed));
        else if (rPrevSpeed >0)              rMotorSpeed = max(0, int(-1*deltaT/timeout * maxMotorSpeed + rPrevSpeed));
        //left speed update
        if(lPrevSpeed <0)                    lMotorSpeed = min(0, int(deltaT/timeout * maxMotorSpeed + lPrevSpeed));
        else if (lPrevSpeed >0)              lMotorSpeed = max(0, int(-1*deltaT/timeout * maxMotorSpeed + lPrevSpeed));
      }
      while((lMotorSpeed != 0 || rMotorSpeed != 0) && deltaT <= timeout);
      break;
    //CASE 3: LEFT 
    case('R'):
      do
      {
        deltaT = millis() - start;
        
        //rightspeed update
        if(rPrevSpeed <0)                    rMotorSpeed = min(0, int(deltaT/timeout * maxMotorSpeed + rPrevSpeed));
        else if (rPrevSpeed >0)              rMotorSpeed = max(0, int(-1*deltaT/timeout * maxMotorSpeed + rPrevSpeed));
        //left speed update
        lMotorSpeed = min(maxMotorSpeed, int(deltaT/timeout * maxMotorSpeed + lPrevSpeed));
      }
      while((lMotorSpeed != maxMotorSpeed || rMotorSpeed != 0) && deltaT <= timeout);
      break;
    //CASE 4: RIGHT
    case('L'):
      do
      {
        deltaT = millis() - start;
        
        //rightspeed update
        rMotorSpeed = min(maxMotorSpeed, int(deltaT/timeout * maxMotorSpeed + rPrevSpeed));
        
        //left speed update
        if(lPrevSpeed <0)                    lMotorSpeed = min(0, int(deltaT/timeout * maxMotorSpeed + lPrevSpeed));
        else if (lPrevSpeed >0)              lMotorSpeed = max(0, int(-1*deltaT/timeout * maxMotorSpeed + lPrevSpeed));
      }
      while((lMotorSpeed != 0 || rMotorSpeed != maxMotorSpeed)&& deltaT <= timeout);
      break;

    //CASE 5: DEBUG
    case('D'):
      Serial.write(rMotorSpeed);
      Serial.write(lMotorSpeed);
      break;
    }
    right(rMotorSpeed);
    left(lMotorSpeed);
    //Debugging motorSpeed.
    //Serial.write(rMotorSpeed);
    //Serial.write(lMotorSpeed);
  }
}
