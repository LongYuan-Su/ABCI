#include "WirelessControl.h"

#include <Arduino.h>
#include <WebServer.h>
#include <WiFi.h>

#include "AWQ215.h"

namespace WirelessControl {
namespace {

constexpr char WIFI_AP_SSID[] = "ESP32B_CTRL";
constexpr char WIFI_AP_PASSWORD[] = "12345678";
constexpr uint8_t WIFI_CHANNEL = 6;
constexpr uint8_t WIFI_MAX_CLIENTS = 2;
constexpr uint16_t TCP_CONTROL_PORT = 3333;
constexpr size_t TCP_COMMAND_BUFFER_SIZE = 64;
constexpr unsigned long TCP_IDLE_TIMEOUT_MS = 30000;

WebServer server(80);
WiFiServer tcpServer(TCP_CONTROL_PORT);
WiFiClient tcpClient;
char tcpCommandBuffer[TCP_COMMAND_BUFFER_SIZE] = {};
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

void processTcpCommand(char* command) {
  char* target = strtok(command, " ,=");
  char* action = strtok(nullptr, " ,=");

  if (target == nullptr) {
    return;
  }

  const String targetString(target);

  if (equalsIgnoreCase(targetString, "PING")) {
    sendTcpResponse("OK PONG");
    return;
  }

  if (equalsIgnoreCase(targetString, "STATUS")) {
    sendTcpResponse(AWQ215::buildStatusJson());
    return;
  }

  if (action == nullptr) {
    sendTcpResponse("ERR FORMAT");
    return;
  }

  bool enabled = false;
  if (!parseEnabledState(String(action), enabled)) {
    sendTcpResponse("ERR ACTION");
    return;
  }

  if (equalsIgnoreCase(targetString, "ALL")) {
    AWQ215::setAll(enabled);
    sendTcpResponse(enabled ? "OK ALL ON" : "OK ALL OFF");
    return;
  }

  if (!AWQ215::setChannelByName(targetString, enabled)) {
    sendTcpResponse("ERR CHANNEL");
    return;
  }

  String response = "OK ";
  response += targetString;
  response += ' ';
  response += enabled ? "ON" : "OFF";
  sendTcpResponse(response);
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

  if (tcpCommandLength >= TCP_COMMAND_BUFFER_SIZE - 1) {
    tcpCommandLength = 0;
    tcpCommandBuffer[0] = '\0';
    sendTcpResponse("ERR TOO LONG");
    return;
  }

  tcpCommandBuffer[tcpCommandLength] = received;
  ++tcpCommandLength;
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

  printStartupInfo();
}

void handleClient() {
  server.handleClient();
  handleTcpClient();
}

}  // namespace WirelessControl
