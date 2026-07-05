#include "WirelessControl.h"

#include <Arduino.h>
#include <WebServer.h>
#include <WiFi.h>
#include <WiFiUdp.h>

#include "AWQ215.h"

namespace WirelessControl {
namespace {

constexpr char WIFI_AP_SSID[] = "ESP32B_CTRL";
constexpr char WIFI_AP_PASSWORD[] = "12345678";
constexpr uint8_t WIFI_CHANNEL = 6;
constexpr uint8_t WIFI_MAX_CLIENTS = 2;
constexpr uint16_t TCP_CONTROL_PORT = 3333;
constexpr uint16_t UDP_CONTROL_PORT = 3333;
constexpr size_t CONTROL_COMMAND_BUFFER_SIZE = 64;
constexpr uint8_t UDP_RESPONSE_REPEAT_COUNT = 2;
constexpr uint8_t UDP_RESPONSE_REPEAT_GAP_MS = 2;
constexpr unsigned long TCP_IDLE_TIMEOUT_MS = 30000;

WebServer server(80);
WiFiServer tcpServer(TCP_CONTROL_PORT);
WiFiClient tcpClient;
WiFiUDP udpControl;
char tcpCommandBuffer[CONTROL_COMMAND_BUFFER_SIZE] = {};
char udpCommandBuffer[CONTROL_COMMAND_BUFFER_SIZE] = {};
size_t tcpCommandLength = 0;
unsigned long lastTcpActivityMs = 0;

bool equalsIgnoreCase(const String& left, const char* right) {
  return left.equalsIgnoreCase(right);
}

bool parseEnabledState(const String& action, bool& enabled) {
  if (equalsIgnoreCase(action, "ON") || action == "1") {
    enabled = true;
    return true;
  }

  if (equalsIgnoreCase(action, "OFF") || action == "0") {
    enabled = false;
    return true;
  }

  return false;
}

void sendJson(int statusCode, const String& body) {
  server.send(statusCode, "application/json", body);
}

void handleRoot() {
  server.send(
      200,
      "text/plain",
      "ESP32B WiFi control ready\n"
      "Use /cmd?target=CH1&action=ON\n"
      "Use /cmd?target=ALL&action=OFF\n"
      "Use /status\n"
      "Low latency UDP: 192.168.4.1:3333\n"
      "Low latency TCP: 192.168.4.1:3333\n");
}

void handlePing() {
  sendJson(200, "{\"ok\":true,\"message\":\"pong\"}");
}

void handleStatus() {
  sendJson(200, AWQ215::buildStatusJson());
}

void handleCommand() {
  if (!server.hasArg("target") || !server.hasArg("action")) {
    sendJson(400, "{\"ok\":false,\"error\":\"missing target or action\"}");
    return;
  }

  const String target = server.arg("target");
  const String action = server.arg("action");
  bool enabled = false;

  if (!parseEnabledState(action, enabled)) {
    sendJson(400, "{\"ok\":false,\"error\":\"bad action\"}");
    return;
  }

  if (equalsIgnoreCase(target, "ALL")) {
    AWQ215::setAll(enabled);
    sendJson(200, "{\"ok\":true,\"target\":\"ALL\"}");
    return;
  }

  if (!AWQ215::setChannelByName(target, enabled)) {
    sendJson(404, "{\"ok\":false,\"error\":\"bad channel\"}");
    return;
  }

  String response = "{\"ok\":true,\"target\":\"";
  response += target;
  response += "\",\"enabled\":";
  response += enabled ? "true" : "false";
  response += "}";
  sendJson(200, response);
}

void handleNotFound() {
  sendJson(404, "{\"ok\":false,\"error\":\"not found\"}");
}

void sendTcpResponse(const char* response) {
  tcpClient.println(response);
  tcpClient.flush();
  lastTcpActivityMs = millis();
}

void sendTcpResponse(const String& response) {
  tcpClient.println(response);
  tcpClient.flush();
  lastTcpActivityMs = millis();
}

void sendUdpResponse(const String& response) {
  for (uint8_t index = 0; index < UDP_RESPONSE_REPEAT_COUNT; ++index) {
    udpControl.beginPacket(udpControl.remoteIP(), udpControl.remotePort());
    udpControl.println(response);
    udpControl.endPacket();

    if (index + 1 < UDP_RESPONSE_REPEAT_COUNT) {
      delay(UDP_RESPONSE_REPEAT_GAP_MS);
    }
  }
}

String processControlCommand(char* command) {
  char* target = strtok(command, " ,=\r\n\t");
  char* action = strtok(nullptr, " ,=\r\n\t");

  if (target == nullptr) {
    return "";
  }

  const String targetString(target);

  if (equalsIgnoreCase(targetString, "PING")) {
    return "OK PONG";
  }

  if (equalsIgnoreCase(targetString, "STATUS")) {
    return AWQ215::buildStatusJson();
  }

  if (action == nullptr) {
    return "ERR FORMAT";
  }

  bool enabled = false;
  if (!parseEnabledState(String(action), enabled)) {
    return "ERR ACTION";
  }

  if (equalsIgnoreCase(targetString, "ALL")) {
    AWQ215::setAll(enabled);
    return enabled ? "OK ALL ON" : "OK ALL OFF";
  }

  if (!AWQ215::setChannelByName(targetString, enabled)) {
    return "ERR CHANNEL";
  }

  String response = "OK ";
  response += targetString;
  response += ' ';
  response += enabled ? "ON" : "OFF";
  return response;
}

void processTcpCommand(char* command) {
  const String response = processControlCommand(command);
  if (response.length() > 0) {
    sendTcpResponse(response);
  }
}

void handleTcpCommandByte(char received) {
  if (received == '\r') {
    return;
  }

  if (received == '\n') {
    tcpCommandBuffer[tcpCommandLength] = '\0';
    processTcpCommand(tcpCommandBuffer);
    tcpCommandLength = 0;
    tcpCommandBuffer[0] = '\0';
    return;
  }

  if (tcpCommandLength >= CONTROL_COMMAND_BUFFER_SIZE - 1) {
    tcpCommandLength = 0;
    tcpCommandBuffer[0] = '\0';
    sendTcpResponse("ERR TOO LONG");
    return;
  }

  tcpCommandBuffer[tcpCommandLength] = received;
  ++tcpCommandLength;
}

void handleUdpClient() {
  int packetSize = udpControl.parsePacket();
  while (packetSize > 0) {
    if (packetSize >= static_cast<int>(CONTROL_COMMAND_BUFFER_SIZE)) {
      while (udpControl.available() > 0) {
        udpControl.read();
      }

      sendUdpResponse("ERR TOO LONG");
    } else {
      const int commandLength = udpControl.read(udpCommandBuffer, CONTROL_COMMAND_BUFFER_SIZE - 1);
      if (commandLength > 0) {
        udpCommandBuffer[commandLength] = '\0';
        const String response = processControlCommand(udpCommandBuffer);
        if (response.length() > 0) {
          sendUdpResponse(response);
        }
      }
    }

    packetSize = udpControl.parsePacket();
  }
}

void acceptTcpClient() {
  WiFiClient newClient = tcpServer.available();
  if (!newClient) {
    return;
  }

  if (tcpClient && tcpClient.connected()) {
    tcpClient.stop();
    Serial.println("TCP control client replaced");
  }

  tcpClient = newClient;
  tcpClient.setNoDelay(true);
  tcpCommandLength = 0;
  tcpCommandBuffer[0] = '\0';
  lastTcpActivityMs = millis();
  sendTcpResponse("OK ESP32B TCP READY");
  Serial.println("TCP control client connected");
}

void handleTcpClient() {
  if (!tcpClient || !tcpClient.connected()) {
    if (tcpClient) {
      tcpClient.stop();
    }

    acceptTcpClient();
    return;
  }

  if (millis() - lastTcpActivityMs > TCP_IDLE_TIMEOUT_MS) {
    tcpClient.stop();
    tcpCommandLength = 0;
    tcpCommandBuffer[0] = '\0';
    Serial.println("TCP control client idle timeout");
    acceptTcpClient();
    return;
  }

  while (tcpClient.available() > 0) {
    lastTcpActivityMs = millis();
    handleTcpCommandByte(static_cast<char>(tcpClient.read()));
  }
}

void printStartupInfo() {
  Serial.print("WiFi AP SSID: ");
  Serial.println(WIFI_AP_SSID);
  Serial.print("WiFi AP password: ");
  Serial.println(WIFI_AP_PASSWORD);
  Serial.print("ESP32B IP: ");
  Serial.println(WiFi.softAPIP());
  Serial.print("TCP control port: ");
  Serial.println(TCP_CONTROL_PORT);
  Serial.print("UDP control port: ");
  Serial.println(UDP_CONTROL_PORT);
}

}  // namespace

void begin() {
  WiFi.mode(WIFI_AP);
  WiFi.setSleep(false);
  WiFi.softAP(WIFI_AP_SSID, WIFI_AP_PASSWORD, WIFI_CHANNEL, false, WIFI_MAX_CLIENTS);

  server.on("/", HTTP_GET, handleRoot);
  server.on("/ping", HTTP_GET, handlePing);
  server.on("/status", HTTP_GET, handleStatus);
  server.on("/cmd", HTTP_GET, handleCommand);
  server.onNotFound(handleNotFound);
  server.begin();
  tcpServer.begin();
  tcpServer.setNoDelay(true);
  udpControl.begin(UDP_CONTROL_PORT);

  printStartupInfo();
}

void handleClient() {
  server.handleClient();
  handleUdpClient();
  handleTcpClient();
}

}  // namespace WirelessControl
