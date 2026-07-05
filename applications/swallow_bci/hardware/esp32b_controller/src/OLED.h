#pragma once

#include <stdbool.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define OLED_DEFAULT_ADDRESS 0x3C
#define OLED_WIDTH 128
#define OLED_HEIGHT 64
#define OLED_TEXT_ROWS 8
#define OLED_TEXT_COLUMNS 21

bool OLED_begin(uint8_t sdaPin, uint8_t sclPin);
bool OLED_beginWithAddress(uint8_t sdaPin, uint8_t sclPin, uint8_t address);
bool OLED_isReady(void);
void OLED_clear(void);
void OLED_displayOn(bool enabled);
void OLED_setCursor(uint8_t column, uint8_t row);
void OLED_print(const char* text);
void OLED_printLine(uint8_t row, const char* text);
void OLED_printCentered(uint8_t row, const char* text);
void OLED_printRight(uint8_t row, const char* text);
void OLED_showStartupScreen(const char* title, const char* subtitle);

#ifdef __cplusplus
}
#endif
