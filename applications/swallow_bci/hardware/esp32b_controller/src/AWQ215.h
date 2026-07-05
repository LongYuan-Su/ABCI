#pragma once

#include <Arduino.h>

namespace AWQ215 {

void begin();
bool setChannelByName(const String& name, bool enabled);
void setAll(bool enabled);
String buildStatusJson();
uint8_t channelCount();
const char* channelName(uint8_t channelIndex);
bool isChannelEnabled(uint8_t channelIndex);
uint32_t channelOnSeconds(uint8_t channelIndex);

}  // namespace AWQ215
