# 36 角度自动扫描流程

入口是 `tools/auto_scan_workflow.py`。目前实现同一相机高度的 **36 组自动采集、解码、掩膜、三角测量、配准融合和缺口候选报告**。每组是现有完整 Gray code 序列（包括白光照片）；图片数由投影仪分辨率决定，不固定为 11 张。

## 接入运动 API

复制 `tools/motion_adapter_example.py` 为自己的适配器，填入设备 SDK 调用。必须提供：

```python
def rotate(degrees: float) -> None:  # 正数顺转，负数逆转
    ...

def slide(millimeters: float) -> None:  # 正数上升，负数下降
    ...

def wait_idle() -> None:  # 可选，等待电机完全停止
    ...
```

`rotate`、`slide` 必须在设备运动完成后才返回，或者提供可靠的 `wait_idle`。SDK 的正负方向若不同，在适配器内转换。请先用小角度空载验证方向、限位和实际步长。

## 命令

在 `final` 目录 PowerShell 执行：

```powershell
$run = 'outputs/structured_light/auto_scan_001'
.\.venv\Scripts\python.exe tools/auto_scan_workflow.py capture `
  --run-dir $run --motion-adapter tools/my_motion_adapter.py --display 2
```

打开的采集窗口仍要求先框选白墙并按 `W` 完成白平衡，再框选手办、按 `C`/`F` 检查对焦。完成后自动拍摄第 1 组，再每次转台旋转 +10° 拍下一组，直到第 36 组（0° 到 350°）。每组拍摄前重新检查清晰度和过曝；失败会停止自动推进。采集过程中不可移动物体、相机或投影仪。

确认 `capture_complete.json` 出现后重建：

```powershell
.\.venv\Scripts\python.exe tools/auto_scan_workflow.py reconstruct `
  --run-dir $run --calibration outputs/structured_light/calibration/current/calibration.json `
  --projector-offset 0 0
```

输出：`decode/`、`segmentation/`、`depth/`、`fusion/dense.ply` 和 `gap_candidates.json`。`--projector-offset` 必须针对本次几何检查；历史实验的 −102、−44 像素不能无验证地沿用。融合要求组间几何匹配及优化通过，失败会报错，保留前面阶段结果供排查。

## 缺口和补拍

`gap_candidates.json` 根据各组前景内部的无效深度连通区域排序，记录候选角度和像素位置。它是补拍**候选**，不是 3D 网格孔洞的证明；需要查看 `fusion/dense.ply` 及相邻组照片，排除自然遮挡、深色材质和掩膜错误。

当前代码**不会自动移动滑杆补拍并合并**。滑杆改变采集视点后，必须知道新高度的相机/投影仪外参，还要验证补拍点云与原模型的重叠配准。否则直接合并会重新产生重影。实际运动 API 未提供之前，适配器示例会明确报错。下一步需要设备 API 的函数签名、滑杆运动单位/范围，以及每个补拍高度的外参标定或可重复定位方案，才能安全完成闭环补拍。
