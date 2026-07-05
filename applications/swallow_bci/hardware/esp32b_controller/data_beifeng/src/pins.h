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

// 你的原理图 BATTERY_ADC 接到 GPIO19，GPIO19 不是 ESP32 ADC 引脚。
// 这版先禁用，后面如果飞线到 GPIO34，再改成 34。
constexpr int BAT_ADC = -1;
// constexpr int BAT_ADC = 34;

}
