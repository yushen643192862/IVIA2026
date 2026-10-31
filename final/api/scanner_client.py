import requests

class ScannerRemote:
    def __init__(self, server_ip):
        """
        初始化远程连接
        参数: server_ip (例如: 'http://10.162.217.61:8000')
        """
        self.base_url = server_ip
        print(f"🔗 已连接到远程扫描仪: {self.base_url}")

    def rotate_pan(self, degrees):
        """控制转盘旋转 (正数顺时针，负数逆时针)"""
        print(f"⏳ 正在请求转台旋转 {degrees} 度...")
        url = f"{self.base_url}/rotate?degrees={degrees}"
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                print("✅ 转台已到位！")
                return True
            else:
                print(f"❌ 转台动作失败: {response.text}")
                return False
        except Exception as e:
            print(f"❌ 网络请求失败: {e}")
            return False

    def move_tilt(self, mm):
        """控制滑台升降 (正数上升，负数下降)"""
        print(f"⏳ 正在请求滑台移动 {mm} 毫米...")
        url = f"{self.base_url}/move?mm={mm}"
        try:
            response = requests.get(url, timeout=20)
            if response.status_code == 200:
                print("✅ 滑台已到位！")
                return True
            else:
                print(f"❌ 滑台动作失败: {response.text}")
                return False
        except Exception as e:
            print(f"❌ 网络请求失败: {e}")
            return False