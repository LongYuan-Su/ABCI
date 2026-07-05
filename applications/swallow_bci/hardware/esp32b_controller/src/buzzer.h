#pragma once

#include <Arduino.h>

namespace Buzzer {

void begin();
void update();
void setEnabled(bool enabled);
bool isEnabled();
void playChannelOnCue();
void playChannelOffCue();

}  // namespace Buzzer
