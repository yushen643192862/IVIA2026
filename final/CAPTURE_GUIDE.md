# 手动旋转：完整 Gray code 结构光＋白光照片

入口：`tools/capture_structured_light.py`。Basler 接口参考 `../lab1/Lab1_Group4/Code/grab_basler1.1.py`，每次使用 `GrabOne()` 获取新帧，在投影图案稳定等待之后才启动该帧采集。不会读取旧的预览缓存来保存图案照片。

## 启动

在 `final` 目录打开 PowerShell，关闭其他占用相机的 pylon Viewer 或采集程序：

```powershell
.\.venv\Scripts\python.exe tools/capture_structured_light.py --serial 40031417 --display 5
```

当前检测到 Basler `daA2500-14uc`、序列号 `40031417`；HDMI 投影仪的 Windows 设备号是 `DISPLAY5`，1920×1080。只有一块副屏时可以不写 `--display`，程序自动选择该副屏。

安装采集依赖：

```powershell
.\.venv\Scripts\python.exe -m pip install -r tools/requirements-capture.txt
```

列出相机或检查连接（不打开投影/预览窗口）：

```powershell
.\.venv\Scripts\python.exe tools/capture_structured_light.py --list-cameras
.\.venv\Scripts\python.exe tools/capture_structured_light.py --serial 40031417 --check
```

## 首次准备

程序在投影仪上全屏显示，在主屏显示相机预览、原像素局部图、清晰度/过曝指标和控制按钮。

1. **调整曝光。** 原相机设置会保留，程序关闭自动曝光、自动增益和自动白平衡。预览若过曝，在界面修改曝光微秒和增益，点击“应用曝光/增益”。这会清除已确认的白平衡与对焦状态，需要重新检查。开始保存组后锁定这些设置，需要修改时新开会话。
2. **先框选白墙，再按 W。** 在预览上按住鼠标左键拖框，选一块均匀、未过曝的中性白墙。青框为白平衡采样区；按W后，程序投影均匀白场，等待稳定并估算固定软件BGR增益。完成前不会进入手办选区。B可返回步骤1重新选择。也可以框选放在物体位置的中性白卡/灰卡。
3. **再框选手办保存区域。** 白平衡完成后，拖框选择手办，黄色框同时用于对焦检查和最终图片裁剪。必须包含旋转后尾巴、武器和肢体的完整活动范围。右侧显示该区域中心最多280×280的原像素局部图。
4. **按 C 检查对焦。** 投影显示棋盘格，手动调整相机镜头或投影对焦。观察原像素局部图、Laplacian清晰度、对比度和过曝比例。
5. **按 F 确认。** 程序要求该ROI的清晰度、对比度和过曝指标通过，然后记录当前清晰度作为参考。第一组之前重新拖选手办会取消对焦确认，需要再次C/F；第一组开始后整个会话锁定裁剪框。

白墙必须确实接近中性、不过曝，且最好和手办处于相同照明。上部白墙未被投影照亮，而手办受投影白光照明时，两处光源颜色可能不同，白墙校准只能近似参考。程序不能从像素自动证明墙漆是标准中性白。

程序不控制普通手动镜头的对焦环，也不能只靠清晰度分数保证绝对准确对焦。相机和投影仪失焦都可能让投影条纹模糊；原像素局部图和人工确认仍需要保留。

## 每个角度的操作

1. 手动旋转物体，停稳。
2. 点击主屏预览窗口，按 **S**（按钮会显示当前每组张数）。
3. 程序先投射棋盘格做一次清晰度复查，此帧只检查，不保存进正式数据。
4. 从密到疏拍摄横向Gray code条纹，再从密到疏拍摄竖向条纹，每个位均拍正相和反相。每张投影后默认等待600ms，再获取新帧。
5. 拍摄原始黑场、白场参考图，再单独拍一张均匀白场颜色照片。所有照片只保存黄色框内的原像素裁剪区域，不缩放。
6. 完整保存后提示可以旋转，再手动转到下一角度并按S。

采集一组期间保持相机、投影仪和物体静止。长按S不会连续启动多组；采集中其他采集指令被忽略。Esc关闭程序，当前未完成组保留并标记 `complete=false`，不会被当作完整组。

白平衡在会话中校准一次并固定；每次S都重新检查清晰度。变换转角后如果清晰度低于参考的40%或绝对阈值，对比度不足、过曝超限，程序会拒绝开始或标记失败。不同表面与光照也会改变清晰度分数，遇到误拒绝时应重新选手办ROI并C/F确认，不应把分数当成独立的光学测量。

开始第一组后，为保持所有图像的坐标一致，不允许重新拖动保存框；可以在同一框内重新C/F确认清晰度，若需要改变框的位置或大小，应新开会话。

## 每组图案和数量

使用完整双向Gray code，每个编码位都拍正相与反相。横线编码y（行），竖线编码x（列）；每个方向按位号从低到高排列，即从密到疏。棋盘格用于前置清晰度检查，不额外当作深度编码。

当前1920×1080、`--cell-size 1`：

| 文件 | 内容 | 张数 |
| --- | --- | ---: |
| `01_horizontal_bit00.png`～`22_horizontal_bit10_inverse.png` | 横线，从密到疏；11个位×正反相 | 22 |
| `23_vertical_bit00.png`～`44_vertical_bit10_inverse.png` | 竖线，从密到疏；11个位×正反相 | 22 |
| `45_black_reference.png` | 原始黑场参考 | 1 |
| `46_white_reference.png` | 原始白场参考 | 1 |
| `47_color_white.png` | 白平衡后的白光颜色照片 | 1 |

合计47张。一般公式为 `2×[ceil(log2(逻辑宽度))+ceil(log2(逻辑高度))]+3`，逻辑尺寸为投影尺寸除以`cell-size`后向上取整。改变屏幕分辨率或编码单元大小时，按钮、进度和完成检查自动调整。

“白光”由投影仪输出均匀白场提供，不是软件物理关闭投影仪后切换外部灯。现在预览、原像素局部图和组目录下全部47张PNG都会应用同一套固定软件白平衡。测深用的46张编码图及黑白参考另存于`raw/`，保持原始相机响应，后续解码应使用`raw/`，不要混用校正图和原图。最后一张颜色照片没有重复的测深用途，不另存未校正副本。

每帧的`axis`、`bit`、`inverse`、`cell_size`、`purpose`保存在JSON。这里的采集排序与OpenCV或独立投影工具可能不同，解码应读取这些字段，不应只凭文件编号猜测位顺序。

该模块完成的是采集。准确深度仍需要相机/投影仪标定、置信度和阴影过滤、Gray code解码以及三角测量。不会把这些照片直接作为普通COLMAP多视角照片。旧的11张双频相移数据保留在旧会话，其生成函数也保留供复现，但不是当前采集默认方案。

## 输出结构

```text
outputs/structured_light/structured_capture/时间戳/
├── session.json                相机/投影/图案配置
├── white_balance.json          白墙ROI与固定软件增益
├── focus_check.json            对焦ROI、指标和确认参考
├── capture_roi.json            固定裁剪框、原尺寸、保存尺寸和坐标转换
├── events.jsonl                采集事件
├── group_0001/
│   ├── 01_horizontal_bit00.png
│   ├── ...
│   ├── 45_black_reference.png
│   ├── 46_white_reference.png
│   ├── 47_color_white.png
│   ├── raw/                    未做软件白平衡的46张测深原图
│   └── group.json              complete、逐帧时间戳、曝光、图案、质量指标
└── group_0002/...
```

只处理 `group.json` 中 `complete=true`、`saved_images=expected_images` 的组。失败组保留以便排查，重试会使用新的组号。文件独占创建，不覆盖已有照片。

每帧的`software_white_balance_applied`标记组目录下PNG是否已经校正，`raw_file`指向对应测深原图。对旧会话可运行 `tools/balance_capture_session.py 会话目录`，按该组记录的增益补做白平衡：先保留原始测深图片和`group_before_white_balance.json`，再更新导出图与记录；重复执行不会二次叠加增益。

恢复同一会话：

```powershell
.\.venv\Scripts\python.exe tools/capture_structured_light.py --serial 40031417 --display 5 --resume-session outputs/structured_light/structured_capture/20261026_181504_049876
```

恢复时检查相机曝光/图像设置、投影仪信息、图案计划和裁剪尺寸一致，再恢复白平衡与对焦基准。下一组使用未占用的新组号；每组仍有清晰度复查。相机或投影仪物理位置改变后不要仅依赖软件参数相同，应重新检查标定与选区。

白平衡框与手办框独立保存，白墙不会混入裁剪数据。各组`group.json`记录`crop_roi_xywh`、`sensor_image_size`和`saved_image_size`。裁剪保持像素比例，坐标关系为`u原图=u裁剪+x`、`v原图=v裁剪+y`。若用原图标定内参，裁剪图的`fx/fy`不变，`cx`需减去裁剪框x，`cy`需减去y；不能直接把原尺寸的主点坐标用于裁剪图。当前模块记录转换关系，不自动完成相机/投影仪标定。默认不另存全画幅副本，框外内容无法从裁剪图恢复。

## 联合标定拍照（J）

使用打印的 `outputs/structured_light/calibration_board/chessboard_9x6_20mm_A4.pdf`：9×6内角点、每格20mm。实际打印尺寸必须用尺核对；如果尺寸不同，修改对应 `board.json` 的 `square_size_mm`，启动时可通过 `--board` 指定其他描述文件。

相机与投影仪位置、镜头焦距和对焦固定。先完成白平衡，然后将实体棋盘格完整放进相机画面，在手办工作距离附近固定好。点击 **J：联合标定拍照**，也可按键盘J。无需重新框选手办；标定拍摄始终保留完整相机画面。

程序在白光下识别54个内角点，检查清晰度，通过后自动采集完整结构光序列（1920×1080、cell-size=1时为47张）。整个序列保持标定板静止；最后白光图再次检测角点，前后最大偏移超过1像素时整组标为无效。此检查不能保证板在序列中途未移动，必须用支撑固定。

保存位置：当前会话的 `calibration/pose_0001/`、`pose_0002/` 等。每个姿态包含白平衡PNG、`raw/`测深原图、`board_corners_before.png`角点检查图和`group.json`（全画幅角点、毫米坐标、相机/投影配置、完整性）。标定数据不会覆盖`group_*`手办照片，也不会改变手办裁剪区域。

换不同倾角、左右上下位置和前后距离，固定后再次按J，建议采集15～20个清晰姿态；标定板应覆盖手办所在工作体积。不要只在一个平面平移，也不要移动相机/投影仪。完成采集后还需解码并求解相机—投影仪参数；拍照按钮本身不输出标定结果。

## 可调整参数（启动命令）

## 从解码结果计算标定深度

使用 `tools/reconstruct_structured_depth.py` 读取已完成的解码目录及选定的标定数据，例如：

```powershell
$out = "outputs/structured_light/structured_depth_$(Get-Date -Format 'yyyyMMdd_HHmmss')"
.\.venv\Scripts\python.exe tools/reconstruct_structured_depth.py `
  outputs/structured_light/structured_decode_20261026_210453 `
  --calibration outputs/structured_light/calibration/current/calibration.json `
  --output $out
```

每个`group_*`输出`depth_mm.npz`（Z深度，单位毫米；无效值NaN）、`depth_preview.png`（按本视图2%～98%深度范围着色）、`cloud.ply`（彩色二进制点云，毫米）、`quality.json`。`status.json`记录完成数量；`depth_summary.json`记录整次测深结果，`calibration_used.json`保存参数快照。

程序恢复裁剪坐标、分别去除相机和投影仪畸变，然后用`X投影仪=R·X相机+t`进行三角测量。默认只保留相机Z深度50～3000mm、投影仪正深度、双侧重投影误差≤1.5像素及射线夹角≥0.25°的点。可用`--min-depth-mm`、`--max-depth-mm`、`--max-reprojection-error`调整范围；`--limit 1`只处理第一组。

若已有前景掩膜，通过`--masks-directory`传入目录，文件名为`group_0001.png`等，白色保留、黑色排除，尺寸必须与裁剪图一致，不自动缩放。没有提供掩膜时输出含背景的测深结果，不能视为已分割手办。

各组点云位于固定相机坐标系，但手办在组间旋转，因此需要下一步物体配准和融合；本命令不合并视角、不生成网格。小的重投影误差不等价于独立验证的毫米精度。

## 采集参数

### 当前结构光数据的固定补偿与直接扣绿流程

要使用这轮确定的阈值重新生成全部31组逐视角点云和对比图，直接运行：

```powershell
.\.venv\Scripts\python.exe tools/reconstruct_structured_groups.py `
  --output outputs/structured_light/structured_groups_3_2pct_new
```

此命令默认使用绝对对比度`3 DN`、相对对比度`2%`、白黑参考差`20 DN`、投影坐标补偿`(-102,-44)`，并使用当前直接扣绿掩膜。每组分别从旧保守深度图的空档导出远背景深度分界。可用`--bit-threshold 4 --relative-threshold 0.03`覆盖阈值，其他选项见`--help`。输出目录必须不存在。最终提供`decode/`、`depth/group_*/cloud.ply`和`comparisons/index.html`，原正式模型不会被覆盖。

单独运行`decode_capture_session.py`时，对比度默认值同样是`3 DN / 2%`；`reconstruct_structured_depth.py`可通过`--reference-depth-dir`指定旧保守深度图以过滤各组远背景。降低阈值增加可解码像素，也可能增加错误对应；逐组对比后再生成融合网格。

本轮数据使用显式经验补偿`--projector-offset -102 -44`，不修改原始解码结果或标定参数。这一补偿在31组数据上通过了双侧重投影检查，但实际毫米精度未独立验证。

```powershell
.\.venv\Scripts\python.exe tools/segment_structured_subject.py `
  outputs/structured_light/structured_decode_20261026_210453/color_images `
  --output outputs/structured_light/structured_subject_green_only_new

.\.venv\Scripts\python.exe tools/reconstruct_structured_depth.py `
  outputs/structured_light/structured_decode_20261026_210453 `
  --projector-offset -102 -44 `
  --masks-directory outputs/structured_light/structured_subject_green_only_new/masks `
  --output outputs/structured_light/structured_depth_green_only_new

.\.venv\Scripts\python.exe tools/fuse_structured_clouds.py `
  outputs/structured_light/structured_depth_green_only_new `
  --output outputs/structured_light/structured_fused_green_only_new

.\tools\colmap-4.2.1-cuda\bin\colmap.exe poisson_mesher `
  --input_path outputs/structured_light/structured_fused_green_only_new/dense.ply `
  --output_path outputs/structured_light/structured_fused_green_only_new/mesh.ply `
  --PoissonMeshing.depth 10 --PoissonMeshing.trim 5 `
  --PoissonMeshing.num_threads 4
```

输出目录必须不存在。掩膜按原分辨率逐像素扣除绿色：OpenCV H为35～90、S≥25，并要求G>1.08R且G>1.03B；内部孔隙及底座草同样去除。保留所有其他像素，不做连通域筛选、腐蚀、填孔或底部截断。另存黑底图和联系表供检查。非绿色桌面仍可能保留。

融合使用白光SIFT特征、邻域深度支持、三维刚体RANSAC和包含首尾视角的位姿图。每视图估计朝向相机的法线，变换后按0.5mm体素融合，要求至少两个视图支持并去除离群点。位姿优化误差、变换和补偿记录保存在`registration.json`；`dense.ply`含彩色点及法线。Poisson可能补出未观测到的曲面，需结合点云检查，不能把网格闭合视为测量完整。

```powershell
.\.venv\Scripts\python.exe tools/capture_structured_light.py `
  --serial 40031417 --display 5 `
  --settle-ms 800 --projector-level 220 --cell-size 1
```

- `--exposure-us`、`--gain`：显式设置初始曝光与增益；未提供时沿用相机当前值。
- `--settle-ms`：图案提交后的软件等待，默认600ms，不能视为投影仪硬件同步确认。
- `--projector-level`：投影最大灰度级，默认220；降低会减弱投影，也可能改善过曝。
- `--cell-size`：编码单元的投影像素边长，默认1。最细条纹无法分辨时可试2或4，分别降低编码分辨率和每组张数；细不等于一定更准确。
- `--focus-square`：对焦棋盘格大小，默认32个投影像素。
- `--min-sharpness`：绝对清晰度阈值，默认20，数值与ROI、成像尺寸和场景有关。
- `--focus-ratio`：每组复查相对参考的下限，默认0.4。
- `--max-clipped`：ROI中过曝灰度像素比例上限，默认0.05。
- `--output`：一个尚不存在的新会话目录。

参考：[Basler单帧采集接口](https://docs.baslerweb.com/pylonapi/cpp/pylon_programmingguide)、[OpenCV Gray code说明](https://docs.opencv.org/4.1.1/d1/dec/classcv_1_1structured__light_1_1GrayCodePattern.html)。
