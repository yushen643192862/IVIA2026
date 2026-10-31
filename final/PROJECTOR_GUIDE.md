# HDMI 投影仪：第二屏全屏结构光

程序入口为 `tools/project_patterns.py`，图案生成代码为 `scanner/structured_light.py`。使用已有的 NumPy、OpenCV 和 Python Tk，无需投影仪 SDK。Windows 将 HDMI 投影仪作为扩展显示器，程序向该显示器输出无边框图案。

## 连接和启动

1. HDMI 连接投影仪，选择对应 HDMI 输入。
2. Windows 按 `Win+P`，选择“扩展”。“复制”模式无法独立控制投影仪内容。
3. 在 `final` 目录的 PowerShell 中运行：

```powershell
.\.venv\Scripts\python.exe tools/project_patterns.py --list-displays
```

输出 `id`、Windows 设备名、主屏标志、物理分辨率和虚拟桌面坐标。`--display` 使用这里的 `id`。连接后确认投影仪实际对应哪个编号，不假设枚举顺序。

在第二屏显示 Gray code：

```powershell
.\.venv\Scripts\python.exe tools/project_patterns.py --display 2
```

程序按显示器当前物理分辨率生成图片，不缩放图案、不在投影区域叠加文字，隐藏鼠标指针。支持负坐标和不同 DPI 的屏幕布局。窗口打开后通过键盘控制：

| 按键 | 操作 |
| --- | --- |
| 空格 / 右方向键 | 下一张 |
| 左方向键 / Backspace | 上一张 |
| A | 开始/暂停自动播放 |
| R | 回到第一张 |
| Esc | 关闭全屏窗口 |

若快捷键没有响应，先点击投影窗口使其获得键盘焦点。编号和状态打印在终端，投影画面保持纯图案。

## 自动播放

```powershell
.\.venv\Scripts\python.exe tools/project_patterns.py --display 2 --auto --settle-ms 500 --interval-ms 1500
```

每张图案提交后等待 500 ms，再保留 1500 ms 后切换下一张。只播放一轮，最后一张保持显示，R 可返回开头。自动播放没有触发相机；相机未接入程序前，建议手动切换，停稳并确认投影稳定后拍摄。

`READY` 只是软件等待计时到期，不代表获得了投影仪的硬件呈现确认。普通 HDMI 投影存在缓冲、刷新和色轮等延迟，不能仅凭这个时间戳宣称相机曝光同步。后续相机采集接口应在稳定等待后触发，并把返回图像关联到图案编号。

## 图案类型

```powershell
# 灰码：白场、黑场、横向坐标编码及反相、纵向坐标编码及反相
.\.venv\Scripts\python.exe tools/project_patterns.py --display 2 --mode graycode

# 棋盘格：每格 64 个投影像素（用于检查对焦/对比度，并非直接完成标定）
.\.venv\Scripts\python.exe tools/project_patterns.py --display 2 --mode checkerboard --cell-size 64

# 固定随机块纹理：每块 8 个像素；固定 seed 可重现
.\.venv\Scripts\python.exe tools/project_patterns.py --display 2 --mode random --cell-size 8 --seed 42

# 均匀白场或黑场
.\.venv\Scripts\python.exe tools/project_patterns.py --display 2 --mode white
.\.venv\Scripts\python.exe tools/project_patterns.py --display 2 --mode black
```

Gray code 默认编码每个投影像素。`--cell-size 4` 改为每 4×4 像素一个编码单元，最细图案更容易成像，但投影坐标分辨率降低。并非格子越密就越准确；需按投影对焦、相机清晰度、表面反光与曝光调整。

Gray code 帧数为：

```text
2 + 2 × [ceil(log2(逻辑宽度)) + ceil(log2(逻辑高度))]
逻辑尺寸 = ceil(投影尺寸 / cell-size)
```

例如 1920×1080、cell-size=1 时，共 46 张：2 张白/黑参考图和 44 张编码及反相图。白黑参考图排在最前，实际解码时须按 manifest 的 role 区分，不能盲目把全部图片当 OpenCV 的编码数组。

## 会话记录和离线导出

默认会话目录为 `outputs/projector/时间戳/`，也可以指定一个新的空目录：

```powershell
.\.venv\Scripts\python.exe tools/project_patterns.py --display 2 --output outputs/projector_session01
```

- `manifest.json`：屏幕信息、分辨率、图案类型、seed、逐帧名称和编码位。
- `display_events.jsonl`：逐帧提交和软件等待结束的时间记录；不是相机曝光时间。

离线导出不需要接入第二屏，也不打开窗口：

```powershell
.\.venv\Scripts\python.exe tools/project_patterns.py --export-only --width 1920 --height 1080 --mode graycode --output outputs/projector_patterns_1080p
```

生成逐帧 PNG 和 `manifest.json`。全屏模式始终采用屏幕当前分辨率；`--width`、`--height` 只用于离线导出。

## 和扫描流程的关系

目前工具负责图案生成、全屏显示、切换和记录，不包含工业相机 SDK 触发、相机/投影仪标定、结构光解码或深度计算。图案能全屏显示不等于系统已经能准确测深。

同一转角采集一套图案时，物体、相机、投影仪保持不动。完成后再旋转物体。正常无投影照片应另行采集；程序显示黑场不等于物理关闭投影仪，黑场仍可能有漏光。关闭投影、采用均匀照明后再拍真实颜色照片更可靠。

投影仪尽量采用原生分辨率和固定位置，关闭动态对比度、自动亮度和自动梯形校正等随画面变化的处理，固定相机曝光/增益/对焦。相机与投影仪需要在实际使用的固定几何和显示设置下完成标定。

## Basler自动采集

已增加单独的 `tools/capture_structured_light.py`：先框选白墙完成白平衡，再框选手办保存区域并检查对焦，手动旋转后按S拍完整的横/竖Gray code（密到疏、每位正反相）、黑白参考和单独颜色照片。1920×1080默认每组47张。启动和操作见 [CAPTURE_GUIDE.md](CAPTURE_GUIDE.md)。本页的 `project_patterns.py` 仍是独立投影工具，不会触发相机。
