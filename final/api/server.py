import time
from hardware_api import ScannerHardware

# 1. 实例化硬件并连接（改成你的 COM 口）
bot = ScannerHardware(port='COM3')

if bot.connect():
    try:
        print("\n--- 开始本地控制 ---")

        # 2. 像调普通 Python 函数一样，随意发指令
        bot.rotate_pan(90)   # 转 90 度
        time.sleep(1)
        
        bot.move_tilt(15)    # 上升 15 毫米
        time.sleep(1)
        
        bot.rotate_pan(180)  # 再转 180 度
        time.sleep(1)
        
        bot.move_tilt(-15)   # 下降 15 毫米

        # 3. 你可以随便写 for 循环
        # for i in range(4):
        #     bot.rotate_pan(i * 90)
        #     bot.move_tilt(10)
        #     time.sleep(1)

        print("\n🎉 动作全部执行完毕！")

    finally:
        # 4. 无论程序是否报错，最后都要安全关闭串口
        bot.close()