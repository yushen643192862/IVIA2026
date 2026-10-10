# 单相机标定与球体 AR — Assignment 3

本项目使用棋盘格照片标定 Basler 相机，分析采集方式对标定结果的影响，将相机姿态和棋盘格导出为 PLY，并在照片和实时相机画面中投影虚拟球体。

## 作业完成情况

| 要求 | 实现程序 | 验证情况 |
| --- | --- | --- |
| 内参与外参标定 | `calibrate.py` | 19 张照片成功完成标定 |
| 分析照片数量、棋盘覆盖和视角的影响 | `experiments.py` | 使用统一留出集完成 25 次标定实验 |
| 将相机中心与棋盘格导出为 PLY | `export_ply.py` | 已导出点云 |
| 将三维点投影到输入图片 | `project_sphere.py` | 19 张照片完成带明暗效果的球体投影，并导出组合三角网格 |
| 实时演示 | `live_sphere.py` | 19 张照片的离线姿态估计和绘制验证通过；实时硬件性能仍待现场验证 |

## 目录结构

```text
assignment3/                         19 张原始 BMP 照片
camera_calibration/
  calibrate.py                       标定与重投影误差计算
  experiments.py                     可复现的子集对比实验
  export_ply.py                      棋盘格与相机姿态可视化
  project_sphere.py                  静态球体 AR 与组合 PLY 网格
  live_sphere.py                     Basler/pypylon 实时球体 AR
  results/
    calibration.json                 可直接阅读的标定参数
    calibration.npz                  NumPy 参数文件
    per_image_errors.csv             各照片的标定重投影误差
    camera_centers.csv               相机中心与照片文件名对应表
    calibration_scene.ply            棋盘格、相机中心和视锥
    calibration_scene_with_sphere.ply 球体组合网格（输出到 results/ 时）
    camera_centers.ply               相机中心点云
    chessboard.ply                   棋盘格点云
    corners/                        角点检测可视化
    undistorted_example.jpg          去畸变示例图
    experiments/                    实验记录、分组名单与汇总
    ar_sphere/                      按下方命令生成的静态 AR 结果
    live_demo/                      离线验证、截图与录像
```

输出子目录按需创建，实际位置由 `--output` 决定。运行相应程序前，不一定存在全部目录。

## 环境安装

需要 Python、NumPy 和 OpenCV。Basler 采集还需要 pypylon 和对应的 pylon 相机驱动。

```powershell
pip install -r requirements.txt
```

以下命令均在仓库根目录 `lab3` 中运行。

## 相机标定

```powershell
python camera_calibration/calibrate.py --images assignment3 --output camera_calibration/results
```

棋盘格有 **10 列 × 7 行内角点**，对应 11 × 8 个格子。图像分辨率为 2592 × 1944。各照片共用内参和畸变参数，每张照片分别估计旋转和平移。

当前全部照片的重投影 RMSE 为 **2.175707 像素**：

```text
K = [[3692.588493,    0.000000, 1269.994118],
     [   0.000000, 3689.660601,  988.861738],
     [   0.000000,    0.000000,    1.000000]]

畸变参数 [k1, k2, p1, p2, k3]：
[-0.541522, 0.342266, 0.004883, -0.003261, -0.957076]
```

默认格子边长为 1，因此平移和导出的三维坐标以**一个格子的边长**为单位，不是毫米。需要测量拍摄时屏幕上实际显示的格子尺寸，并确认横纵边长相同。例如，实际测得边长为 20 mm 时：

```powershell
python camera_calibration/calibrate.py --images assignment3 --output camera_calibration/results --square-size 20 --unit mm
```

20 mm 仅为使用示例，不是当前数据的测量值。统一改变尺度会改变平移和几何尺寸，不改变内参和旋转。整组照片需要保持格子尺寸、相机光学设置及传感器模式一致。

坐标变换满足 `X_camera = R @ X_board + t`。第一枚检测到的内角点为棋盘坐标原点，X 沿列方向，Y 沿行方向。相机中心为 `C = -R.T @ t`。Rodrigues 旋转向量以弧度表示，不是欧拉角。对称棋盘格可能出现角点排序翻转，导出的姿态不能直接视为统一追踪坐标系中的拍摄轨迹。

## 标定影响因素对比实验

```powershell
python camera_calibration/experiments.py --images assignment3 --calibration camera_calibration/results/calibration.json --output camera_calibration/results/experiments
```

### 实验方法

- 19 张照片使用相同角点检测、亚像素优化设置和五参数畸变模型。
- 固定留出 4 张照片，不参与任何实验组的内参拟合：`14-35-51`、`14-38-53`、`14-52-44` 和 `14-56-42`（文件名末尾的时间）。
- 剩余 15 张为训练池。验证时固定各实验组的内参和畸变，仅使用 `solvePnP` 估计留出照片的外参。
- 验证 RMSE 为全部留出角点的二维欧氏残差平方均值的平方根，衡量图像空间的重投影一致性，不等同于真实物理测量精度。
- 数量实验分别使用 5、10、15 张训练照片。5 张和 10 张各随机抽取 10 组，随机种子为 20261010；15 张使用全部训练池，只运行一次。
- 覆盖实验比较棋盘归一化中心最靠近图像中心的 6 张照片与最远离中心的 6 张照片。这是**位置的近似指标**，不是对整幅图像角点覆盖范围的直接测量。
- 角度实验比较倾角最低和最高的各 6 张照片。倾角由此前全量标定的姿态计算：`acos(abs(R[2,2]))`。它是近似分组标签，不是独立测量的真实角度。
- 每次实验的训练与验证文件名及参数均保存在 `experiments.json` 和 `runs.csv`。

### 实验结果

| 因素 | 分组 | 训练照片数 | 次数 | 训练 RMSE 均值（px） | 验证 RMSE 均值（px） | 验证 RMSE 标准差（px） |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 照片数量 | 5 张 | 5 | 10 | 2.501 | 1.524 | 0.159 |
| 照片数量 | 10 张 | 10 | 10 | 2.295 | 1.360 | 0.038 |
| 照片数量 | 15 张 | 15 | 1 | 2.355 | 1.325 | 不适用 |
| 棋盘位置 | 中心组 | 6 | 1 | 3.002 | 1.498 | 不适用 |
| 棋盘位置 | 外围组 | 6 | 1 | 1.378 | 1.387 | 不适用 |
| 拍摄角度 | 小倾角组 | 6 | 1 | 2.621 | 3.288 | 不适用 |
| 拍摄角度 | 大倾角组 | 6 | 1 | 2.466 | 1.488 | 不适用 |

标准差描述不同抽样子集之间的变化，不代表独立拍摄批次的不确定性。仅运行一次的组无法估计重复实验波动；机器可读的汇总文件将其标准差记为零。

### 结果讨论

**照片数量。** 从 5 张增加到 10 张后，平均留出集 RMSE 降低约 10.8%；使用全部 15 张训练照片时，比 5 张组均值降低约 13.1%。估计焦距 `fx` 在不同子集间的标准差从 76.22 降至 26.25 像素。本数据中，更多照片提高了稳定性，但训练 RMSE 没有单调下降，因为新增照片也带来不同观测及可能更大的噪声。

**棋盘位置。** 外围组验证 RMSE 比中心组低约 7.4%，棋盘中心的平均归一化偏移为 0.323 和 0.064。但两组棋盘凸包平均面积分别占画面的 10.3% 和 23.6%，平均倾角也不同（19.9° 和 16.9°），清晰度同样存在差异。因此，这一比较支持采集不同位置的照片，但不能将差异完全归因于位置，也不能证明统计显著性。

**拍摄角度。** 大倾角组平均倾角为 35.8°，验证 RMSE 比平均倾角 9.0° 的小倾角组低约 54.8%。这与斜视提供较强几何约束的预期一致。两组棋盘大小、位置和图像质量仍有差异，因此结果属于观察性比较，不能作为严格控制条件下的因果估计。也不能推断角度越大越好，过度倾斜可能降低角点定位质量。

**局限性。** 数据仅含 19 张照片和 4 张固定验证照片。验证外参也由验证角点拟合，因此评估的是重新求姿态后的重投影一致性，而不是通过独立真值测得的三维精度。角度标签来自全量标定，包含留出照片的信息；实验内参只使用训练照片拟合，但角度分组并非完全独立。位置和角度组各运行一次。模糊、屏幕显示伪影、棋盘大小和光照没有独立控制，也没有真实相机参数或毫米级精度真值。全量标定与留出验证使用不同照片集合，不能将两者 RMSE 当作可直接互换的分数比较。

**采集建议。** 使用多张清晰照片，包含正面与适度倾斜视角，覆盖画面不同区域，保持对焦、焦距和图像几何设置一致。增加模糊或重复照片的收益，可能不如采集少量清晰且姿态多样的照片。

## 三维可视化

```powershell
python camera_calibration/export_ply.py --calibration camera_calibration/results/calibration.json
```

在 MeshLab 中导入 `camera_calibration/results/calibration_scene.ply`，启用点显示。黑白点表示棋盘格，红点表示相机中心，蓝线表示视锥。

## 静态球体 AR 与组合模型

```powershell
python camera_calibration/project_sphere.py --images assignment3 --calibration camera_calibration/results/calibration.json --output camera_calibration/results/ar_sphere --radius 0.5
```

程序保存叠加后的照片、总览图 `sphere_gallery.jpg` 和 `calibration_scene_with_sphere.ply`。组合三角网格包含棋盘格、半径为半个格子边长的蓝球、红色相机中心、蓝色视锥和绿色光轴。球体与棋盘相切，位于朝向相机的一侧。这些几何体用于虚拟展示，不是扫描重建的物体。将组合 PLY 导入 MeshLab 即可查看三维场景。

## Basler 实时演示

```powershell
python camera_calibration/live_sphere.py
```

程序通过 pypylon 连续获取最新帧，转换为 BGR，读取 `camera_calibration/results/calibration.json` 中已有的内参和畸变参数。每帧检测棋盘，用 `solvePnP` 估计姿态，并在原始带畸变画面上绘制球体。未检测到完整棋盘时不绘制球体。

- 使用同一台已标定相机，保持对焦、焦距、ROI 和像素合并（binning）设置不变。关闭占用相机连接的其他程序。
- 默认请求 2592 × 1944 分辨率，Basler 实际尺寸必须与请求尺寸一致。同宽高比缩放时程序按比例缩放内参，但仅适用于图像缩放，不能用于传感器裁剪或光学设置改变。
- 多台相机时，通过 `--serial SERIAL_NUMBER` 或 `--camera 1` 选择已标定相机。
- 固定曝光与增益示例：`--exposure-us 10000 --gain-db 0`。根据光照调整；未指定时保留相机当前值，并关闭自动曝光和自动增益。
- **Q/Esc** 退出，**S** 保存截图，**R** 开始或停止录像；结果位于 `camera_calibration/results/live_demo/`。
- AVI 录像采用固定 15 FPS，实际采集或绘制帧率不同时，播放时长可能与现场时长不同。

无需连接相机的离线验证：

```powershell
python camera_calibration/live_sphere.py --check-images assignment3
```

此前 19 张照片均通过检测、姿态估计和绘制验证。这不代表现场相机连接和实时帧率已验证。`offline_validation.json` 记录各照片的跟踪状态及姿态重投影误差。

## GitHub 上传说明

建议保留程序、本 README、依赖文件、标定数据和实验记录。19 张原始 BMP 合计约 287 MB（十进制），可考虑 Git LFS 或单独提供数据集。如果不上传原图，应说明获取方式，便于复现实验。`.gitignore` 已排除实时录像和 Python 临时文件。

这些程序不会发布仓库，也没有自动选择许可证。按课程要求注明 OpenCV、Basler/pypylon、复用代码及 AI 辅助的来源。Basler 连接流程参考了小组的 `lab1/Lab1_Group4/Code/grab_basler1.1.py`。
