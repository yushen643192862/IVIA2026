import serial
import time

class ScannerHardware:
    def __init__(self, port='COM3', baudrate=115200):
        self.port = port
        self.baudrate = baudrate
        self.esp32 = None

    def connect(self):
        try:
            self.esp32 = serial.Serial(self.port, self.baudrate, timeout=1)
            time.sleep(2) 
            self.esp32.reset_input_buffer()
            print(f"✅ 成功连接到硬件 {self.port}")
            return True
        except Exception as e:
            print(f"❌ 连接失败: {e}")
            return False

    def _send_and_wait(self, command, expected_reply, timeout=15):
        if not self.esp32 or not self.esp32.is_open:
            return False

        self.esp32.write((command + "\n").encode())
        
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.esp32.in_waiting > 0:
                response = self.esp32.readline().decode('utf-8', errors='ignore').strip()
                if expected_reply in response:
                    return True
        print(f"  ⚠️ 等待超时！")
        return False

    def rotate_pan(self, degrees):
        """转盘旋转"""
        print(f"🔄 转盘旋转 {degrees} 度...")
        cmd = f"P{degrees}"
        return self._send_and_wait(cmd, "PAN_DONE")

    def move_tilt(self, mm):
        """滑台升降"""
        print(f"↕️ 滑台移动 {mm} mm...")
        cmd = f"T{mm}"
        return self._send_and_wait(cmd, "TILT_DONE")

    def close(self):
        if self.esp32 and self.esp32.is_open:
            self.esp32.close()
            print("🔌 串口已关闭。")