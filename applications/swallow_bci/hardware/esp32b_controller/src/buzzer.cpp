#include "buzzer.h"

#include "pins.h"

namespace Buzzer {
namespace {

constexpr bool ACTIVE_LOW = false;
constexpr unsigned long BEEP_DURATION_MS = 500;
constexpr unsigned long BEEP_GAP_MS = 250;
constexpr uint8_t CHANNEL_ON_BEEP_COUNT = 2;
constexpr uint8_t CHANNEL_OFF_BEEP_COUNT = 1;

bool enabledState = false;
uint8_t remainingBeeps = 0;
bool cueRunning = false;
bool beepPhaseActive = false;
unsigned long nextPhaseMs = 0;

uint8_t levelFor(bool enabled) {
  if (ACTIVE_LOW) {
    return enabled ? LOW : HIGH;
  }

  return enabled ? HIGH : LOW;
}

bool timeReached(unsigned long now, unsigned long target) {
  return static_cast<long>(now - target) >= 0;
}

void startCue(uint8_t beepCount) {
  remainingBeeps = beepCount;
  cueRunning = beepCount > 0;
  beepPhaseActive = cueRunning;

  setEnabled(cueRunning);
  nextPhaseMs = millis() + BEEP_DURATION_MS;
}

}  // namespace

void begin() {
  pinMode(Pins::BUZZER, OUTPUT);
  setEnabled(false);
}

void update() {
  if (!cueRunning || !timeReached(millis(), nextPhaseMs)) {
    return;
  }

  if (beepPhaseActive) {
    setEnabled(false);
    --remainingBeeps;

    if (remainingBeeps == 0) {
      cueRunning = false;
      return;
    }

    beepPhaseActive = false;
    nextPhaseMs = millis() + BEEP_GAP_MS;
    return;
  }

  beepPhaseActive = true;
  setEnabled(true);
  nextPhaseMs = millis() + BEEP_DURATION_MS;
}

void setEnabled(bool enabled) {
  enabledState = enabled;
  digitalWrite(Pins::BUZZER, levelFor(enabled));
}

bool isEnabled() {
  return enabledState;
}

void playChannelOnCue() {
  startCue(CHANNEL_ON_BEEP_COUNT);
}

void playChannelOffCue() {
  startCue(CHANNEL_OFF_BEEP_COUNT);
}

}  // namespace Buzzer
