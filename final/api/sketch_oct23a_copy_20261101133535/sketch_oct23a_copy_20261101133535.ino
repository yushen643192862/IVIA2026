#include <Stepper.h>

// ================= 1. 转盘 (Yaw) =================
const int stepsPerRevolution = 2048;
// 转盘接在 4, 6, 5, 7
Stepper panMotor(stepsPerRevolution, 4, 6, 5, 7);
int panPins[] = { 4, 6, 5, 7 };

// ================= 2. 滑台 (Z轴) =================
// 滑台接在 16, 17, 18
int stepPin = 16;
int dirPin = 17;
int enPin = 18;

void setup() {
  Serial.begin(115200);

  // 初始化转盘
  panMotor.setSpeed(6);  // 转速 6 转/分钟
  for (int i = 0; i < 4; i++) pinMode(panPins[i], OUTPUT);

  // 初始化滑台
  pinMode(stepPin, OUTPUT);
  pinMode(dirPin, OUTPUT);
  pinMode(enPin, OUTPUT);
  digitalWrite(enPin, LOW);  // 使能滑台电机（锁定）

  Serial.println("=== 双轴控制系统就绪 ===");
  Serial.println("输入 P90 转动转盘 90 度");
  Serial.println("输入 T10 上升滑台 10 毫米");
}

void loop() {
  if (Serial.available() > 0) {
    // 读取完整指令
    String command = Serial.readStringUntil('\n');
    command.trim();

    // ================= 处理转盘指令 (P开头) =================
    if (command.startsWith("P") || command.startsWith("p")) {
      float degrees = command.substring(1).toFloat();
      // 将角度换算成步数
      int steps = degrees * (2048.0 / 360.0);

      Serial.print("转盘转动: ");
      Serial.print(degrees);
      Serial.println(" 度");

      panMotor.step(steps);

      // 释放转盘电机（防止待机发烫）
      for (int i = 0; i < 4; i++) digitalWrite(panPins[i], LOW);

      Serial.println("PAN_DONE");  // 告诉电脑转盘转完了
    }

    // ================= 处理滑台指令 (T开头) =================
    else if (command.startsWith("T") || command.startsWith("t")) {
      float mm = command.substring(1).toFloat();

      // 滑台步数计算：丝杆转一圈(3200步)走 2mm，所以 1mm = 1600 步
      int steps = mm * 1600;

      // 判断方向
      if (steps > 0) {
        digitalWrite(dirPin, HIGH);  // 正数上升
      } else {
        digitalWrite(dirPin, LOW);  // 负数下降
        steps = -steps;             // 转成正数用于循环
      }

      Serial.print("滑台移动: ");
      Serial.print(mm);
      Serial.println(" mm");

      // 发送脉冲
      for (long i = 0; i < steps; i++) {
        digitalWrite(stepPin, HIGH);
        delayMicroseconds(600);
        digitalWrite(stepPin, LOW);
        delayMicroseconds(600);
      }

      Serial.println("TILT_DONE");  // 告诉电脑滑台动完了
    }

    else {
      Serial.println("指令错误，请输入 P 或 T");
    }

    // 清空串口缓存
    while (Serial.available() > 0) Serial.read();
  }
}