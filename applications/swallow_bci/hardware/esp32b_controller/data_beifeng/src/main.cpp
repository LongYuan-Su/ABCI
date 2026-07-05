#include <Arduino.h>

#include "AWQ215.h"
#include "WirelessControl.h"
#include "buzzer.h"

namespace {

constexpr uint32_t USB_BAUD = 115200;

}  // namespace

void setup() {
  Serial.begin(USB_BAUD);

  AWQ215::begin();
  Buzzer::begin();
  WirelessControl::begin();
}

void loop() {
  WirelessControl::handleClient();
  Buzzer::update();
}
