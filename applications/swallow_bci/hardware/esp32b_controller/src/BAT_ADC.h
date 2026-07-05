#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BAT_ADC_INVALID_PERCENT 255

bool BAT_ADC_begin(uint8_t adcPin);
bool BAT_ADC_update(void);
bool BAT_ADC_hasReading(void);
uint16_t BAT_ADC_getAdcMilliVolts(void);
uint16_t BAT_ADC_getBatteryMilliVolts(void);
uint8_t BAT_ADC_getPercent(void);
void BAT_ADC_formatPercent(char* buffer, size_t bufferSize);

#ifdef __cplusplus
}
#endif
