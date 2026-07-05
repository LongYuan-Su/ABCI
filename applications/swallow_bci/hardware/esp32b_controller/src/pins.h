#pragma once
#include <Arduino.h>

namespace Pins {

// 四路 AQW214S 控制
constexpr uint8_t CH1 = 32;   // ch1_signal
constexpr uint8_t CH2 = 33;   // ch2_signal
constexpr uint8_t CH3 = 5;    // ch3_signal
constexpr uint8_t CH4 = 18;   // ch4_signal

// 蜂鸣器
constexpr uint8_t BUZZER = 4; // BUZZ

// 板载 LED
constexpr uint8_t LED = 27;   // LED9

// I2C
constexpr uint8_t I2C_SDA = 16;
constexpr uint8_t I2C_SCL = 17;

// 电池电压采样
constexpr int BAT_ADC = 34;   // BATTERY_ADC，GPIO34 / ADC1_CH6

}
