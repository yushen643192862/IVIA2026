from scanner_client import ScannerRemote
import time

# 连接你的电脑
bot = ScannerRemote("http://10.162.217.61:8000")

# 1. 滑台先降到底部
bot.move_tilt(-10)
time.sleep(1)

# 2. 转一圈拍几张照
bot.rotate_pan(0)
time.sleep(1)
bot.rotate_pan(90)
time.sleep(1)
bot.rotate_pan(180)
time.sleep(1)

# 3. 滑台上升 5 毫米，再拍几张
bot.move_tilt(5)
bot.rotate_pan(270)
time.sleep(1)
bot.rotate_pan(0)

print("任务全部完成！")