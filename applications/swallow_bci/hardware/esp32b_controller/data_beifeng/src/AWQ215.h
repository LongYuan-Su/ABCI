#pragma once

#include <Arduino.h>

namespace AWQ215 {

void begin();
bool setChannelByName(const String& name, bool enabled);
void setAll(bool enabled);
String buildStatusJson();

}  // namespace AWQ215
