//Necessary Libraries
#include <BH1750.h>
#include <Wire.h>
#include <Servo.h>
#include <avr/wdt.h>

BH1750 lightSensor;
Servo relay;

float lux = 0;
int SOLENOID_PIN = 14;


unsigned long now;
unsigned long prev_time;


void setup() {

  Serial.begin(9600);
  // I2C Bus
  Wire.begin();

  lightSensor.begin();
  digitalWrite(SOLENOID_PIN, LOW);

  prev_time = millis();
}

void loop() {
  //Stores previous light value
  lux = lightSensor.readLightLevel();
  now = millis();
  Serial.println(lux);
  delay(40);
  //Check if threshold reached
  if ((lux < 600) || (now - prev_time > 30000)) {
    digitalWrite(SOLENOID_PIN, HIGH);
    //Serial.println("Activating Solenoid");
    delay(15);
    //Serial.println(digitalRead(GOOD_PIN));
    //Serial.println("Deactivating Solenoid");
    digitalWrite(SOLENOID_PIN, LOW);
    delay(500);
    prev_time = now;
    if (now>600000){
      reboot();
    }
  }
}

void reboot() {
  wdt_disable();
  wdt_enable(WDTO_15MS);
  while (1) {}
}