// Arduino Uno Q, STM32 side (own scope, milestone E3).
// Receives the Fraktion index from the Linux side and drives LEDs/servo as the "ejection".
// Skeleton: verify the Arduino_RouterBridge API on the board first; plain Serial is the fallback.

const int LED_PINS[] = {2, 3, 4, 5, 6};   // pet_bottle, plastic_film, beverage_carton, metal_can, paper_cardboard
const int LED_RESIDUAL = 7;

void setup() {
  for (int p : LED_PINS) pinMode(p, OUTPUT);
  pinMode(LED_RESIDUAL, OUTPUT);
  Serial.begin(115200);
}

void eject(int idx) {
  int pin = (idx >= 0 && idx < 5) ? LED_PINS[idx] : LED_RESIDUAL;
  digitalWrite(pin, HIGH);
  delay(300);
  digitalWrite(pin, LOW);
}

void loop() {
  if (Serial.available()) {
    int idx = Serial.parseInt();   // -1 = Sortierrest
    eject(idx);
  }
}
