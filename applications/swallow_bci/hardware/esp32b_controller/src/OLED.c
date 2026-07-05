#include "OLED.h"

#include <stddef.h>
#include <string.h>

#include "driver/gpio.h"
#include "driver/i2c.h"
#include "esp_err.h"
#include "freertos/FreeRTOS.h"
#include "freertos/task.h"

#define OLED_I2C_PORT I2C_NUM_0
#define OLED_I2C_CLOCK_HZ 400000
#define OLED_PAGE_COUNT (OLED_HEIGHT / 8)
#define OLED_FONT_WIDTH 5
#define OLED_CHAR_WIDTH 6
#define OLED_COMMAND_CONTROL 0x00
#define OLED_DATA_CONTROL 0x40
#define OLED_WRITE_TIMEOUT_MS 50
#define OLED_DATA_CHUNK_SIZE 16

static uint8_t oledAddress = OLED_DEFAULT_ADDRESS;
static bool oledReady = false;
static uint8_t cursorColumn = 0;
static uint8_t cursorRow = 0;

static const uint8_t GLYPH_SPACE[OLED_FONT_WIDTH] = {0x00, 0x00, 0x00, 0x00, 0x00};
static const uint8_t GLYPH_UNKNOWN[OLED_FONT_WIDTH] = {0x02, 0x01, 0x51, 0x09, 0x06};
static const uint8_t GLYPH_DOT[OLED_FONT_WIDTH] = {0x00, 0x60, 0x60, 0x00, 0x00};
static const uint8_t GLYPH_COLON[OLED_FONT_WIDTH] = {0x00, 0x36, 0x36, 0x00, 0x00};
static const uint8_t GLYPH_MINUS[OLED_FONT_WIDTH] = {0x08, 0x08, 0x08, 0x08, 0x08};
static const uint8_t GLYPH_SLASH[OLED_FONT_WIDTH] = {0x20, 0x10, 0x08, 0x04, 0x02};
static const uint8_t GLYPH_PERCENT[OLED_FONT_WIDTH] = {0x63, 0x13, 0x08, 0x64, 0x63};
static const uint8_t GLYPH_0[OLED_FONT_WIDTH] = {0x3E, 0x51, 0x49, 0x45, 0x3E};
static const uint8_t GLYPH_1[OLED_FONT_WIDTH] = {0x00, 0x42, 0x7F, 0x40, 0x00};
static const uint8_t GLYPH_2[OLED_FONT_WIDTH] = {0x42, 0x61, 0x51, 0x49, 0x46};
static const uint8_t GLYPH_3[OLED_FONT_WIDTH] = {0x21, 0x41, 0x45, 0x4B, 0x31};
static const uint8_t GLYPH_4[OLED_FONT_WIDTH] = {0x18, 0x14, 0x12, 0x7F, 0x10};
static const uint8_t GLYPH_5[OLED_FONT_WIDTH] = {0x27, 0x45, 0x45, 0x45, 0x39};
static const uint8_t GLYPH_6[OLED_FONT_WIDTH] = {0x3C, 0x4A, 0x49, 0x49, 0x30};
static const uint8_t GLYPH_7[OLED_FONT_WIDTH] = {0x01, 0x71, 0x09, 0x05, 0x03};
static const uint8_t GLYPH_8[OLED_FONT_WIDTH] = {0x36, 0x49, 0x49, 0x49, 0x36};
static const uint8_t GLYPH_9[OLED_FONT_WIDTH] = {0x06, 0x49, 0x49, 0x29, 0x1E};
static const uint8_t GLYPH_A[OLED_FONT_WIDTH] = {0x7E, 0x11, 0x11, 0x11, 0x7E};
static const uint8_t GLYPH_B[OLED_FONT_WIDTH] = {0x7F, 0x49, 0x49, 0x49, 0x36};
static const uint8_t GLYPH_C[OLED_FONT_WIDTH] = {0x3E, 0x41, 0x41, 0x41, 0x22};
static const uint8_t GLYPH_D[OLED_FONT_WIDTH] = {0x7F, 0x41, 0x41, 0x22, 0x1C};
static const uint8_t GLYPH_E[OLED_FONT_WIDTH] = {0x7F, 0x49, 0x49, 0x49, 0x41};
static const uint8_t GLYPH_F[OLED_FONT_WIDTH] = {0x7F, 0x09, 0x09, 0x09, 0x01};
static const uint8_t GLYPH_G[OLED_FONT_WIDTH] = {0x3E, 0x41, 0x49, 0x49, 0x7A};
static const uint8_t GLYPH_H[OLED_FONT_WIDTH] = {0x7F, 0x08, 0x08, 0x08, 0x7F};
static const uint8_t GLYPH_I[OLED_FONT_WIDTH] = {0x00, 0x41, 0x7F, 0x41, 0x00};
static const uint8_t GLYPH_J[OLED_FONT_WIDTH] = {0x20, 0x40, 0x41, 0x3F, 0x01};
static const uint8_t GLYPH_K[OLED_FONT_WIDTH] = {0x7F, 0x08, 0x14, 0x22, 0x41};
static const uint8_t GLYPH_L[OLED_FONT_WIDTH] = {0x7F, 0x40, 0x40, 0x40, 0x40};
static const uint8_t GLYPH_M[OLED_FONT_WIDTH] = {0x7F, 0x02, 0x0C, 0x02, 0x7F};
static const uint8_t GLYPH_N[OLED_FONT_WIDTH] = {0x7F, 0x04, 0x08, 0x10, 0x7F};
static const uint8_t GLYPH_O[OLED_FONT_WIDTH] = {0x3E, 0x41, 0x41, 0x41, 0x3E};
static const uint8_t GLYPH_P[OLED_FONT_WIDTH] = {0x7F, 0x09, 0x09, 0x09, 0x06};
static const uint8_t GLYPH_Q[OLED_FONT_WIDTH] = {0x3E, 0x41, 0x51, 0x21, 0x5E};
static const uint8_t GLYPH_R[OLED_FONT_WIDTH] = {0x7F, 0x09, 0x19, 0x29, 0x46};
static const uint8_t GLYPH_S[OLED_FONT_WIDTH] = {0x46, 0x49, 0x49, 0x49, 0x31};
static const uint8_t GLYPH_T[OLED_FONT_WIDTH] = {0x01, 0x01, 0x7F, 0x01, 0x01};
static const uint8_t GLYPH_U[OLED_FONT_WIDTH] = {0x3F, 0x40, 0x40, 0x40, 0x3F};
static const uint8_t GLYPH_V[OLED_FONT_WIDTH] = {0x1F, 0x20, 0x40, 0x20, 0x1F};
static const uint8_t GLYPH_W[OLED_FONT_WIDTH] = {0x3F, 0x40, 0x38, 0x40, 0x3F};
static const uint8_t GLYPH_X[OLED_FONT_WIDTH] = {0x63, 0x14, 0x08, 0x14, 0x63};
static const uint8_t GLYPH_Y[OLED_FONT_WIDTH] = {0x07, 0x08, 0x70, 0x08, 0x07};
static const uint8_t GLYPH_Z[OLED_FONT_WIDTH] = {0x61, 0x51, 0x49, 0x45, 0x43};

static bool writeI2cBytes(const uint8_t* bytes, size_t length) {
  return i2c_master_write_to_device(
             OLED_I2C_PORT,
             oledAddress,
             bytes,
             length,
             pdMS_TO_TICKS(OLED_WRITE_TIMEOUT_MS)) == ESP_OK;
}

static bool writeCommand(uint8_t command) {
  const uint8_t packet[] = {OLED_COMMAND_CONTROL, command};
  return writeI2cBytes(packet, sizeof(packet));
}

static bool writeCommandList(const uint8_t* commands, size_t commandCount) {
  for (size_t commandIndex = 0; commandIndex < commandCount; ++commandIndex) {
    if (!writeCommand(commands[commandIndex])) {
      return false;
    }
  }

  return true;
}

static bool setAddressWindow(uint8_t startColumn, uint8_t endColumn, uint8_t startPage, uint8_t endPage) {
  const uint8_t commands[] = {
      0x21,
      startColumn,
      endColumn,
      0x22,
      startPage,
      endPage,
  };

  return writeCommandList(commands, sizeof(commands));
}

static bool writeDataBuffer(const uint8_t* buffer, size_t length) {
  size_t bytesWritten = 0;

  while (bytesWritten < length) {
    size_t chunkLength = length - bytesWritten;
    if (chunkLength > OLED_DATA_CHUNK_SIZE) {
      chunkLength = OLED_DATA_CHUNK_SIZE;
    }

    uint8_t packet[OLED_DATA_CHUNK_SIZE + 1] = {OLED_DATA_CONTROL};
    memcpy(&packet[1], &buffer[bytesWritten], chunkLength);

    if (!writeI2cBytes(packet, chunkLength + 1)) {
      return false;
    }

    bytesWritten += chunkLength;
  }

  return true;
}

static const uint8_t* glyphForCharacter(char inputCharacter) {
  if (inputCharacter >= 'a' && inputCharacter <= 'z') {
    inputCharacter = (char)(inputCharacter - ('a' - 'A'));
  }

  switch (inputCharacter) {
    case ' ':
      return GLYPH_SPACE;
    case '.':
      return GLYPH_DOT;
    case ':':
      return GLYPH_COLON;
    case '-':
    case '_':
      return GLYPH_MINUS;
    case '/':
      return GLYPH_SLASH;
    case '%':
      return GLYPH_PERCENT;
    case '0':
      return GLYPH_0;
    case '1':
      return GLYPH_1;
    case '2':
      return GLYPH_2;
    case '3':
      return GLYPH_3;
    case '4':
      return GLYPH_4;
    case '5':
      return GLYPH_5;
    case '6':
      return GLYPH_6;
    case '7':
      return GLYPH_7;
    case '8':
      return GLYPH_8;
    case '9':
      return GLYPH_9;
    case 'A':
      return GLYPH_A;
    case 'B':
      return GLYPH_B;
    case 'C':
      return GLYPH_C;
    case 'D':
      return GLYPH_D;
    case 'E':
      return GLYPH_E;
    case 'F':
      return GLYPH_F;
    case 'G':
      return GLYPH_G;
    case 'H':
      return GLYPH_H;
    case 'I':
      return GLYPH_I;
    case 'J':
      return GLYPH_J;
    case 'K':
      return GLYPH_K;
    case 'L':
      return GLYPH_L;
    case 'M':
      return GLYPH_M;
    case 'N':
      return GLYPH_N;
    case 'O':
      return GLYPH_O;
    case 'P':
      return GLYPH_P;
    case 'Q':
      return GLYPH_Q;
    case 'R':
      return GLYPH_R;
    case 'S':
      return GLYPH_S;
    case 'T':
      return GLYPH_T;
    case 'U':
      return GLYPH_U;
    case 'V':
      return GLYPH_V;
    case 'W':
      return GLYPH_W;
    case 'X':
      return GLYPH_X;
    case 'Y':
      return GLYPH_Y;
    case 'Z':
      return GLYPH_Z;
    default:
      return GLYPH_UNKNOWN;
  }
}

static void drawCharacterAt(uint8_t column, uint8_t row, char character) {
  if (!oledReady || column >= OLED_TEXT_COLUMNS || row >= OLED_TEXT_ROWS) {
    return;
  }

  uint8_t characterBuffer[OLED_CHAR_WIDTH] = {0};
  const uint8_t* glyph = glyphForCharacter(character);
  memcpy(characterBuffer, glyph, OLED_FONT_WIDTH);

  const uint8_t startColumn = (uint8_t)(column * OLED_CHAR_WIDTH);
  const uint8_t endColumn = (uint8_t)(startColumn + OLED_CHAR_WIDTH - 1);

  if (setAddressWindow(startColumn, endColumn, row, row)) {
    writeDataBuffer(characterBuffer, sizeof(characterBuffer));
  }
}

static void drawPage(uint8_t row, const uint8_t* pageBuffer) {
  if (!oledReady || row >= OLED_TEXT_ROWS) {
    return;
  }

  if (setAddressWindow(0, OLED_WIDTH - 1, row, row)) {
    writeDataBuffer(pageBuffer, OLED_WIDTH);
  }
}

bool OLED_begin(uint8_t sdaPin, uint8_t sclPin) {
  return OLED_beginWithAddress(sdaPin, sclPin, OLED_DEFAULT_ADDRESS);
}

bool OLED_beginWithAddress(uint8_t sdaPin, uint8_t sclPin, uint8_t address) {
  oledAddress = address;
  oledReady = false;

  i2c_config_t i2cConfig = {
      .mode = I2C_MODE_MASTER,
      .sda_io_num = (gpio_num_t)sdaPin,
      .scl_io_num = (gpio_num_t)sclPin,
      .sda_pullup_en = GPIO_PULLUP_ENABLE,
      .scl_pullup_en = GPIO_PULLUP_ENABLE,
      .master.clk_speed = OLED_I2C_CLOCK_HZ,
      .clk_flags = 0,
  };

  if (i2c_param_config(OLED_I2C_PORT, &i2cConfig) != ESP_OK) {
    return false;
  }

  const esp_err_t installResult = i2c_driver_install(OLED_I2C_PORT, I2C_MODE_MASTER, 0, 0, 0);
  if (installResult != ESP_OK && installResult != ESP_ERR_INVALID_STATE) {
    return false;
  }

  vTaskDelay(pdMS_TO_TICKS(50));

  const uint8_t initCommands[] = {
      0xAE,
      0xD5, 0x80,
      0xA8, 0x3F,
      0xD3, 0x00,
      0x40,
      0x8D, 0x14,
      0x20, 0x00,
      0xA1,
      0xC8,
      0xDA, 0x12,
      0x81, 0xCF,
      0xD9, 0xF1,
      0xDB, 0x40,
      0xA4,
      0xA6,
      0x2E,
      0xAF,
  };

  if (!writeCommandList(initCommands, sizeof(initCommands))) {
    return false;
  }

  oledReady = true;
  OLED_clear();
  return true;
}

bool OLED_isReady(void) {
  return oledReady;
}

void OLED_clear(void) {
  if (!oledReady) {
    return;
  }

  const uint8_t blankBuffer[OLED_DATA_CHUNK_SIZE] = {0};
  if (!setAddressWindow(0, OLED_WIDTH - 1, 0, OLED_PAGE_COUNT - 1)) {
    return;
  }

  for (uint16_t chunkIndex = 0; chunkIndex < (OLED_WIDTH * OLED_PAGE_COUNT) / OLED_DATA_CHUNK_SIZE; ++chunkIndex) {
    if (!writeDataBuffer(blankBuffer, sizeof(blankBuffer))) {
      return;
    }
  }

  cursorColumn = 0;
  cursorRow = 0;
}

void OLED_displayOn(bool enabled) {
  if (!oledReady) {
    return;
  }

  writeCommand(enabled ? 0xAF : 0xAE);
}

void OLED_setCursor(uint8_t column, uint8_t row) {
  cursorColumn = column < OLED_TEXT_COLUMNS ? column : OLED_TEXT_COLUMNS - 1;
  cursorRow = row < OLED_TEXT_ROWS ? row : OLED_TEXT_ROWS - 1;
}

void OLED_print(const char* text) {
  if (!oledReady || text == NULL) {
    return;
  }

  for (size_t textIndex = 0; text[textIndex] != '\0'; ++textIndex) {
    if (text[textIndex] == '\n') {
      cursorColumn = 0;
      if (cursorRow + 1 < OLED_TEXT_ROWS) {
        ++cursorRow;
      }
      continue;
    }

    drawCharacterAt(cursorColumn, cursorRow, text[textIndex]);

    if (cursorColumn + 1 < OLED_TEXT_COLUMNS) {
      ++cursorColumn;
    } else {
      cursorColumn = 0;
      if (cursorRow + 1 < OLED_TEXT_ROWS) {
        ++cursorRow;
      }
    }
  }
}

void OLED_printLine(uint8_t row, const char* text) {
  if (!oledReady || row >= OLED_TEXT_ROWS) {
    return;
  }

  uint8_t lineBuffer[OLED_WIDTH] = {0};

  if (text != NULL) {
    for (uint8_t characterIndex = 0; characterIndex < OLED_TEXT_COLUMNS && text[characterIndex] != '\0'; ++characterIndex) {
      const uint8_t* glyph = glyphForCharacter(text[characterIndex]);
      const uint8_t pixelColumn = (uint8_t)(characterIndex * OLED_CHAR_WIDTH);
      memcpy(&lineBuffer[pixelColumn], glyph, OLED_FONT_WIDTH);
    }
  }

  drawPage(row, lineBuffer);
}

void OLED_printCentered(uint8_t row, const char* text) {
  if (!oledReady || row >= OLED_TEXT_ROWS) {
    return;
  }

  char centeredLine[OLED_TEXT_COLUMNS + 1];
  memset(centeredLine, ' ', OLED_TEXT_COLUMNS);
  centeredLine[OLED_TEXT_COLUMNS] = '\0';

  if (text != NULL) {
    size_t textLength = strlen(text);
    if (textLength > OLED_TEXT_COLUMNS) {
      textLength = OLED_TEXT_COLUMNS;
    }

    const size_t startColumn = (OLED_TEXT_COLUMNS - textLength) / 2;
    memcpy(&centeredLine[startColumn], text, textLength);
  }

  OLED_printLine(row, centeredLine);
}

void OLED_printRight(uint8_t row, const char* text) {
  if (!oledReady || row >= OLED_TEXT_ROWS) {
    return;
  }

  char rightLine[OLED_TEXT_COLUMNS + 1];
  memset(rightLine, ' ', OLED_TEXT_COLUMNS);
  rightLine[OLED_TEXT_COLUMNS] = '\0';

  if (text != NULL) {
    size_t textLength = strlen(text);
    if (textLength > OLED_TEXT_COLUMNS) {
      textLength = OLED_TEXT_COLUMNS;
    }

    const size_t startColumn = OLED_TEXT_COLUMNS - textLength;
    memcpy(&rightLine[startColumn], text, textLength);
  }

  OLED_printLine(row, rightLine);
}

void OLED_showStartupScreen(const char* title, const char* subtitle) {
  if (!oledReady) {
    return;
  }

  OLED_clear();
  OLED_printCentered(1, title);
  OLED_printCentered(3, subtitle);
  OLED_printCentered(5, "SDA16 SCL17");
}
