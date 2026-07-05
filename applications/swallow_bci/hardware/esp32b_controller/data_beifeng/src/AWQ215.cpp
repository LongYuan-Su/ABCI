#include "AWQ215.h"

#include "buzzer.h"
#include "pins.h"

namespace AWQ215 {
namespace {

constexpr uint8_t AQW_ENABLE_LEVEL = HIGH;
constexpr uint8_t AQW_DISABLE_LEVEL = LOW;

struct AqwChannel {
  const char* name;
  uint8_t pin;
  bool enabled;
};

AqwChannel aqwChannels[] = {
    {"CH1", Pins::CH1, false},
    {"CH2", Pins::CH2, false},
    {"CH3", Pins::CH3, false},
    {"CH4", Pins::CH4, false},
};
constexpr uint8_t AQW_CHANNEL_COUNT = sizeof(aqwChannels) / sizeof(aqwChannels[0]);

int findChannelIndex(const String& name) {
  for (uint8_t index = 0; index < AQW_CHANNEL_COUNT; ++index) {
    if (name.equalsIgnoreCase(aqwChannels[index].name)) {
      return index;
    }
  }

  return -1;
}

void refreshStatusLed() {
  bool anyChannelEnabled = false;

  for (uint8_t index = 0; index < AQW_CHANNEL_COUNT; ++index) {
    anyChannelEnabled = anyChannelEnabled || aqwChannels[index].enabled;
  }

  digitalWrite(Pins::LED, anyChannelEnabled ? HIGH : LOW);
}

void playStateCue(bool enabled) {
  if (enabled) {
    Buzzer::playChannelOnCue();
    return;
  }

  Buzzer::playChannelOffCue();
}

bool setChannelByIndex(uint8_t channelIndex, bool enabled) {
  if (channelIndex >= AQW_CHANNEL_COUNT) {
    return false;
  }

  const bool changed = aqwChannels[channelIndex].enabled != enabled;
  aqwChannels[channelIndex].enabled = enabled;
  digitalWrite(aqwChannels[channelIndex].pin, enabled ? AQW_ENABLE_LEVEL : AQW_DISABLE_LEVEL);
  refreshStatusLed();
  return changed;
}

}  // namespace

void begin() {
  for (uint8_t index = 0; index < AQW_CHANNEL_COUNT; ++index) {
    digitalWrite(aqwChannels[index].pin, AQW_DISABLE_LEVEL);
    pinMode(aqwChannels[index].pin, OUTPUT);
    aqwChannels[index].enabled = false;
  }

  pinMode(Pins::LED, OUTPUT);
  digitalWrite(Pins::LED, LOW);
}

bool setChannelByName(const String& name, bool enabled) {
  const int channelIndex = findChannelIndex(name);
  if (channelIndex < 0) {
    return false;
  }

  if (setChannelByIndex(static_cast<uint8_t>(channelIndex), enabled)) {
    playStateCue(enabled);
  }

  return true;
}

void setAll(bool enabled) {
  bool anyChannelChanged = false;

  for (uint8_t index = 0; index < AQW_CHANNEL_COUNT; ++index) {
    anyChannelChanged = setChannelByIndex(index, enabled) || anyChannelChanged;
  }

  if (anyChannelChanged) {
    playStateCue(enabled);
  }
}

String buildStatusJson() {
  String json = "{";

  for (uint8_t index = 0; index < AQW_CHANNEL_COUNT; ++index) {
    if (index > 0) {
      json += ",";
    }

    json += "\"";
    json += aqwChannels[index].name;
    json += "\":";
    json += aqwChannels[index].enabled ? "true" : "false";
  }

  json += "}";
  return json;
}

}  // namespace AWQ215
