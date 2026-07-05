#include "BAT_ADC.h"

#include <stdio.h>

#include "driver/adc.h"
#include "esp_adc_cal.h"
#include "esp_err.h"

#define BAT_ADC_DEFAULT_VREF_MV 1100
#define BAT_ADC_SAMPLE_COUNT 16
#define BAT_ADC_DIVIDER_TOP_OHMS 200000U
#define BAT_ADC_DIVIDER_BOTTOM_OHMS 100000U

typedef struct {
  uint16_t millivolts;
  uint8_t percent;
} BatteryPoint;

static const BatteryPoint BATTERY_CURVE[] = {
    {4200, 100},
    {4100, 90},
    {4000, 80},
    {3920, 70},
    {3850, 60},
    {3790, 50},
    {3730, 40},
    {3680, 30},
    {3600, 20},
    {3500, 10},
    {3300, 0},
};

static adc_unit_t adcUnit = ADC_UNIT_1;
static adc1_channel_t adc1Channel = ADC1_CHANNEL_0;
static adc2_channel_t adc2Channel = ADC2_CHANNEL_0;
static adc_atten_t adcAttenuation = ADC_ATTEN_DB_11;
static adc_bits_width_t adcWidth = ADC_WIDTH_BIT_12;
static esp_adc_cal_characteristics_t adcCharacteristics;
static bool adcReady = false;
static bool usingAdc2 = false;
static bool hasReading = false;
static uint16_t lastAdcMilliVolts = 0;
static uint16_t lastBatteryMilliVolts = 0;
static uint8_t lastPercent = BAT_ADC_INVALID_PERCENT;

static bool mapGpioToAdc(uint8_t pin) {
  switch (pin) {
    case 36:
      adcUnit = ADC_UNIT_1;
      adc1Channel = ADC1_CHANNEL_0;
      usingAdc2 = false;
      return true;
    case 37:
      adcUnit = ADC_UNIT_1;
      adc1Channel = ADC1_CHANNEL_1;
      usingAdc2 = false;
      return true;
    case 38:
      adcUnit = ADC_UNIT_1;
      adc1Channel = ADC1_CHANNEL_2;
      usingAdc2 = false;
      return true;
    case 39:
      adcUnit = ADC_UNIT_1;
      adc1Channel = ADC1_CHANNEL_3;
      usingAdc2 = false;
      return true;
    case 32:
      adcUnit = ADC_UNIT_1;
      adc1Channel = ADC1_CHANNEL_4;
      usingAdc2 = false;
      return true;
    case 33:
      adcUnit = ADC_UNIT_1;
      adc1Channel = ADC1_CHANNEL_5;
      usingAdc2 = false;
      return true;
    case 34:
      adcUnit = ADC_UNIT_1;
      adc1Channel = ADC1_CHANNEL_6;
      usingAdc2 = false;
      return true;
    case 35:
      adcUnit = ADC_UNIT_1;
      adc1Channel = ADC1_CHANNEL_7;
      usingAdc2 = false;
      return true;
    case 4:
      adcUnit = ADC_UNIT_2;
      adc2Channel = ADC2_CHANNEL_0;
      usingAdc2 = true;
      return true;
    case 0:
      adcUnit = ADC_UNIT_2;
      adc2Channel = ADC2_CHANNEL_1;
      usingAdc2 = true;
      return true;
    case 2:
      adcUnit = ADC_UNIT_2;
      adc2Channel = ADC2_CHANNEL_2;
      usingAdc2 = true;
      return true;
    case 15:
      adcUnit = ADC_UNIT_2;
      adc2Channel = ADC2_CHANNEL_3;
      usingAdc2 = true;
      return true;
    case 13:
      adcUnit = ADC_UNIT_2;
      adc2Channel = ADC2_CHANNEL_4;
      usingAdc2 = true;
      return true;
    case 12:
      adcUnit = ADC_UNIT_2;
      adc2Channel = ADC2_CHANNEL_5;
      usingAdc2 = true;
      return true;
    case 14:
      adcUnit = ADC_UNIT_2;
      adc2Channel = ADC2_CHANNEL_6;
      usingAdc2 = true;
      return true;
    case 27:
      adcUnit = ADC_UNIT_2;
      adc2Channel = ADC2_CHANNEL_7;
      usingAdc2 = true;
      return true;
    case 25:
      adcUnit = ADC_UNIT_2;
      adc2Channel = ADC2_CHANNEL_8;
      usingAdc2 = true;
      return true;
    case 26:
      adcUnit = ADC_UNIT_2;
      adc2Channel = ADC2_CHANNEL_9;
      usingAdc2 = true;
      return true;
    default:
      return false;
  }
}

static bool readRawSample(int* rawSample) {
  if (usingAdc2) {
    if (adc2_get_raw(adc2Channel, adcWidth, rawSample) != ESP_OK) {
      return false;
    }

    if (hasReading && *rawSample <= 16) {
      return false;
    }

    return true;
  }

  *rawSample = adc1_get_raw(adc1Channel);
  return *rawSample >= 0;
}

static uint16_t calculateBatteryMilliVolts(uint32_t adcMilliVolts) {
  return (uint16_t)((adcMilliVolts * (BAT_ADC_DIVIDER_TOP_OHMS + BAT_ADC_DIVIDER_BOTTOM_OHMS)) /
                    BAT_ADC_DIVIDER_BOTTOM_OHMS);
}

static uint8_t calculatePercent(uint16_t batteryMilliVolts) {
  const size_t pointCount = sizeof(BATTERY_CURVE) / sizeof(BATTERY_CURVE[0]);

  if (batteryMilliVolts >= BATTERY_CURVE[0].millivolts) {
    return 100;
  }

  if (batteryMilliVolts <= BATTERY_CURVE[pointCount - 1].millivolts) {
    return 0;
  }

  for (size_t index = 0; index + 1 < pointCount; ++index) {
    const BatteryPoint highPoint = BATTERY_CURVE[index];
    const BatteryPoint lowPoint = BATTERY_CURVE[index + 1];

    if (batteryMilliVolts <= highPoint.millivolts && batteryMilliVolts >= lowPoint.millivolts) {
      const uint16_t voltageRange = highPoint.millivolts - lowPoint.millivolts;
      const uint16_t voltageOffset = batteryMilliVolts - lowPoint.millivolts;
      const uint8_t percentRange = highPoint.percent - lowPoint.percent;
      return (uint8_t)(lowPoint.percent + ((uint32_t)voltageOffset * percentRange) / voltageRange);
    }
  }

  return 0;
}

bool BAT_ADC_begin(uint8_t adcPin) {
  adcReady = false;
  hasReading = false;
  lastPercent = BAT_ADC_INVALID_PERCENT;

  if (!mapGpioToAdc(adcPin)) {
    return false;
  }

  if (usingAdc2) {
    if (adc2_config_channel_atten(adc2Channel, adcAttenuation) != ESP_OK) {
      return false;
    }
  } else {
    if (adc1_config_width(adcWidth) != ESP_OK) {
      return false;
    }

    if (adc1_config_channel_atten(adc1Channel, adcAttenuation) != ESP_OK) {
      return false;
    }
  }

  esp_adc_cal_characterize(adcUnit, adcAttenuation, adcWidth, BAT_ADC_DEFAULT_VREF_MV, &adcCharacteristics);
  adcReady = true;
  return BAT_ADC_update();
}

bool BAT_ADC_update(void) {
  if (!adcReady) {
    return false;
  }

  uint32_t rawSum = 0;
  uint8_t validSamples = 0;

  for (uint8_t sampleIndex = 0; sampleIndex < BAT_ADC_SAMPLE_COUNT; ++sampleIndex) {
    int rawSample = 0;
    if (readRawSample(&rawSample)) {
      rawSum += (uint32_t)rawSample;
      ++validSamples;
    }
  }

  if (validSamples == 0) {
    return false;
  }

  const uint32_t rawAverage = rawSum / validSamples;
  lastAdcMilliVolts = (uint16_t)esp_adc_cal_raw_to_voltage(rawAverage, &adcCharacteristics);
  lastBatteryMilliVolts = calculateBatteryMilliVolts(lastAdcMilliVolts);
  lastPercent = calculatePercent(lastBatteryMilliVolts);
  hasReading = true;
  return true;
}

bool BAT_ADC_hasReading(void) {
  return hasReading;
}

uint16_t BAT_ADC_getAdcMilliVolts(void) {
  return lastAdcMilliVolts;
}

uint16_t BAT_ADC_getBatteryMilliVolts(void) {
  return lastBatteryMilliVolts;
}

uint8_t BAT_ADC_getPercent(void) {
  return lastPercent;
}

void BAT_ADC_formatPercent(char* buffer, size_t bufferSize) {
  if (buffer == NULL || bufferSize == 0) {
    return;
  }

  if (!hasReading || lastPercent == BAT_ADC_INVALID_PERCENT) {
    snprintf(buffer, bufferSize, "BAT --%%");
    return;
  }

  snprintf(buffer, bufferSize, "BAT %u%%", (unsigned int)lastPercent);
}
