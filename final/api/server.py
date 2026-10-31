from fastapi import FastAPI, HTTPException
import uvicorn
from hardware_api import ScannerHardware

app = FastAPI(title="3D扫描仪远程控制服务端")

# ⚠️ 这里写你自己电脑连 ESP32 的端口
bot = ScannerHardware(port='COM3')

@app.on_event("startup")
def startup_event():
    bot.connect()

@app.on_event("shutdown")
def shutdown_event():
    bot.close()

@app.get("/rotate")
def api_rotate(degrees: float):
    """别人调用的转盘接口"""
    success = bot.rotate_pan(degrees)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=1000, detail="硬件超时")

@app.get("/move")
def api_move(mm: float):
    """别人调用的滑台接口"""
    success = bot.move_tilt(mm)
    if success:
        return {"status": "success"}
    raise HTTPException(status_code=1000, detail="硬件超时")

if __name__ == "__main__":
    # 启动服务器，监听所有网络接口，端口 8000
    uvicorn.run(app, host="0.0.0.0", port=8000)