#include <Arduino.h>
#include <stdio.h>

#include "AWQ215.h"
#include "OLED.h"
#include "WirelessControl.h"
#include "buzzer.h"
#include "pins.h"

namespace {

constexpr uint32_t USB_BAUD = 115200;
constexpr unsigned long CHANNEL_TIME_ROW_INTERVAL_MS = 250;

unsigned long lastChannelTimeRowMs = 0;
uint8_t nextChannelTimeRow = 0;

bool timeReached(unsigned long now, unsigned long target) {
  return static_cast<long>(now - target) >= 0;
}

void drawChannelTimeHeader() {
  OLED_printCentered(0, "AQW ON TIME");
}

void updateChannelTimeRow(uint8_t channelIndex) {
  if (channelIndex >= AWQ215::channelCount()) {
    return;
  }

  char line[OLED_TEXT_COLUMNS + 1] = {};
  snprintf(
      line,
      sizeof(line),
      "%s %lus %s",
      AWQ215::channelName(channelIndex),
      static_cast<unsigned long>(AWQ215::channelOnSeconds(channelIndex)),
      AWQ215::isChannelEnabled(channelIndex) ? "ON" : "OFF");
  OLED_printLine(channelIndex + 2, line);
}

void drawInitialChannelTimeDisplay() {
  OLED_clear();
  drawChannelTimeHeader();

  for (uint8_t index = 0; index < AWQ215::channelCount(); ++index) {
    updateChannelTimeRow(index);
  }
}

void updateNextChannelTimeRow() {
  updateChannelTimeRow(nextChannelTimeRow);
  ++nextChannelTimeRow;

  if (nextChannelTimeRow >= AWQ215::channelCount()) {
    nextChannelTimeRow = 0;
  }
}

}  // namespace

void setup() {
  Serial.begin(USB_BAUD);

  AWQ215::begin();
  Buzzer::begin();
  OLED_begin(Pins::I2C_SDA, Pins::I2C_SCL);
  drawInitialChannelTimeDisplay();
  lastChannelTimeRowMs = millis() + CHANNEL_TIME_ROW_INTERVAL_MS;
  WirelessControl::begin();
}

void loop() {
  WirelessControl::handleClient();
  Buzzer::update();

  const unsigned long now = millis();
  if (timeReached(now, lastChannelTimeRowMs)) {
    updateNextChannelTimeRow();
    lastChannelTimeRowMs = now + CHANNEL_TIME_ROW_INTERVAL_MS;
  }
}
