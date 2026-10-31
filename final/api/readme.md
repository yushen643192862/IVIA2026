# 3D 扫描仪远程控制 API

## 1. 环境安装
在终端运行：
`pip install -r requirements.txt`

## 2. 服务端（控制机器的同学运行）
1. 修改 `server.py` 中的串口端口：`bot = ScannerHardware(port='COM3')`（改为你自己的 COM 口）。
2. 确保 ESP32 已插在电脑上。
3. 运行服务器：
   `python server.py`
4. 查看本机局域网 IP（终端输入 `ipconfig`），并告诉其他同学，比如 `10.162.217.61`。
*(注意：如果弹出防火墙提示，请勾选允许访问)*

## 3. 客户端（需要调用接口的同学运行）
1. 新建一个测试文件 `test.py`：
```python
from scanner_client import ScannerRemote

# 填写服务端的 IP 地址
bot = ScannerRemote("http://10.162.217.61:8000")

# 调用函数控制
bot.rotate_pan(90)  # 转盘转 90 度
bot.move_tilt(10)   # 滑台升 10 毫米