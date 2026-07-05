#include <HTTPClient.h>
#include <WiFi.h>

namespace {

constexpr char WIFI_SSID[] = "ESP32B_CTRL";
constexpr char WIFI_PASSWORD[] = "12345678";
constexpr char ESP32B_BASE_URL[] = "http://192.168.4.1";

void sendCommand(const char* target, const char* action) {
  if (WiFi.status() != WL_CONNECTED) {
    return;
  }

  HTTPClient http;
  String url = String(ESP32B_BASE_URL) + "/cmd?target=" + target + "&action=" + action;

  http.begin(url);
  int statusCode = http.GET();

  Serial.print("GET ");
  Serial.print(url);
  Serial.print(" -> ");
  Serial.println(statusCode);

  if (statusCode > 0) {
    Serial.println(http.getString());
  }

  http.end();
}

void connectToEsp32B() {
  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);

  Serial.print("Connecting to ");
  Serial.println(WIFI_SSID);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println();
  Serial.print("ESP32A IP: ");
  Serial.println(WiFi.localIP());
}

}  // namespace

void setup() {
  Serial.begin(115200);
  connectToEsp32B();
}

void loop() {
  sendCommand("CH1", "ON");
  delay(1000);
  sendCommand("CH1", "OFF");
  delay(500);

  sendCommand("CH2", "ON");
  delay(1000);
  sendCommand("CH2", "OFF");
  delay(500);

  sendCommand("ALL", "OFF");
  delay(3000);
}
