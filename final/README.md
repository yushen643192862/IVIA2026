# 自适应双轴物体扫描：重建、质量分析与补拍规划

工作目录：`C:\Users\lenovo\Desktop\智能视觉信息采集\final`。

本程序提供六类结果：相机参数与相对姿态、稀疏点云、稠密点云、三角网格、质量分析，以及双轴补拍建议。当前没有实拍数据，已提供独立的**合成演示**；演示直接构造几何数据，不宣称已从真实照片完成重建。

## 先运行演示

在本目录打开 PowerShell：

```powershell
pip install -r requirements.txt
python scan.py doctor
python scan.py demo
```

演示是一个中空杯状物，故意去掉部分内壁网格，并让底部缺少稠密点。三个相机只从侧面观察。程序计算几何遮挡、邻近点支持和观察方向，识别低质量区域，再尝试改变物体姿态。

```text
outputs/demo/
  mesh.ply                  带凹陷和缺失内壁的示意网格
  dense.ply                 人工生成的表面点，包含刻意缺失区域
  sparse.ply                人工抽样点，仅展示稀疏点云形态
  cameras.json              合成相机内外参
  synthetic_phone*.png      几何渲染预览，不用于证明图像重建
  platform_demo.json        未标定的演示平台配置
  quality/
    quality_colored.ply     按质量着色的网格
    quality_preview.png    正面、侧面、顶面的质量预览图
    hole_boundaries.ply     红色开放边界点
    quality_regions.json   采样位置、法向、质量指标及边界信息
    quality_report.csv     可用 Excel 打开的采样统计
  planning/
    recommended_poses.json 全部候选评分与选中的补拍角度
    recommendations.md     可直接阅读的补拍建议
```

将 PLY 拖入 MeshLab 查看。网格选择面显示，边界点云选择点显示。绿色表示较有支持，黄色表示稀疏或视角不足，红色表示缺少邻近稠密点，灰色表示未采样；提供已配准的夹具模型时，紫色表示射线检测到夹具遮挡。

演示配置 `calibrated=false`，所有角度仅供说明，**不能直接作为电机控制指令**。

## 程序结构

```text
scan.py                 命令入口
scanner/io.py           ASCII/二进制 PLY 读写、JSON 输出
scanner/reconstruct.py  COLMAP 调用、相机参数导出、前景点云过滤
scanner/geometry.py     射线可见性、旋转与平台坐标变换
scanner/quality.py      邻域密度、数据支持、开放边界、多视角分析
scanner/planner.py      候选姿态评分与补拍选择
scanner/demo.py         合成几何与示意相机
tests/test_pipeline.py  几何、数据格式及演示流程验证
```

## 真实照片：第一步准备输入

建议每部手机单独一个目录，固定镜头、对焦和分辨率：

```text
data/images/phoneA/0001.jpg
data/images/phoneA/0002.jpg
data/images/phoneB/0001.jpg
data/masks/phoneA/0001.jpg.png
data/masks/phoneA/0002.jpg.png
data/masks/phoneB/0001.jpg.png
```

掩膜与对应照片尺寸一致，**白色是物体及随物体刚性运动的标记，黑色是背景、非刚性运动的平台和夹具**。固定相机拍旋转物体时，不能直接把固定背景和物体作为同一场景重建。本程序在旋转物体模式下要求提供掩膜。`tools/segment_green_screen.py` 提供纯颜色绿幕分割试验；这组照片中它会误删青蓝色帽檐、头发，或在放宽阈值后保留绿色阴影，因此 `tools/reconstruct_green.py` 默认使用模型预测作为主体范围，再用 `tools/refine_mask_green.py` 在原分辨率清除高置信度绿幕。必须检查预览中的细发丝、尾巴和底座一致性。

平台角度另存为采集日志，例如 `image,phone,yaw,pitch`，并记录灯光和装夹变化。当前重建通过 SfM 估计相对姿态，不直接将刻度角度作为已知相机外参；真实平台角度映射需要额外标定。

## 真实照片：第二步安装 COLMAP

`requirements.txt` 仅安装 Python 依赖，**不安装 COLMAP**。需要另行安装 COLMAP，并使 `colmap` 可从命令行运行，或用 `--colmap` 指定可执行文件。当前机器检查时没有找到 COLMAP。

- [官方安装说明](https://colmap.github.io/install.html)
- [官方命令行流程](https://colmap.github.io/cli.html)
- [官方模型格式](https://colmap.github.io/format.html)

程序会读取当前 COLMAP 的帮助文本，兼容新旧的特征提取/匹配 GPU 参数名称。特征提取与匹配使用 CPU；稠密重建调用 `patch_match_stereo`，需要安装版本支持的计算后端。**不能保证仅 CPU 的安装能运行稠密阶段**。失败时查看日志，不会伪造稠密结果。

```powershell
python scan.py doctor --colmap "C:\tools\COLMAP\COLMAP.exe"
```

## 真实照片：第三步重建

```powershell
python scan.py reconstruct --images data/images --masks data/masks --output outputs/scan01/reconstruction
```

如果 COLMAP 未加入 PATH，追加 `--colmap "实际可执行文件路径"`。

流程为：特征提取 → 全量匹配 → SfM → 导出参数和稀疏点云 → 图像去畸变 → 对齐前景掩膜并遮蔽背景 → 第一遍深度估计 → 清零背景深度 → 几何一致性检查 → 再清零背景深度 → 融合 → 前景投影过滤 → Poisson 网格。

提供 `--masks` 时，稠密阶段通过 `scanner/dense_masks.py` 将原始掩膜按相机畸变和稠密相机内参映射到去畸变坐标，不能用简单缩放代替。工作区 `dense/images` 的背景置黑，原始去畸变图保存在 `dense/images_unmasked`，输入照片不改动。PatchMatch 第一遍和几何检查后的背景深度、法线都清零，只有前景有效深度可以参与后续几何检查和融合。当前 PatchMatch 没有原生前景 ROI 参数，因此仍会计算全图网格；这里约束的是输入背景及可被后续使用的深度，不宣称跳过了所有背景像素运算。轮廓窗口可能混合前景背景，掩膜内的误分割和反光仍需检查。

`dense/masks` 保存对齐掩膜；`foreground_photometric_report.json`、`foreground_geometric_report.json` 记录各图删除的背景深度数。`dense/foreground_config.json` 记录掩膜哈希、相机映射和窗口半径。没有该配置的旧深度缓存，以及配置不一致的缓存会被拒用，应另开输出目录。该修改不追溯改写已启动任务或已有模型。

输出：

- `cameras.json`：各图像的 K、畸变、R、t、相机中心，统一表达在 SfM 物体坐标系中。
- `sparse.ply`、`sparse_text/`：稀疏重建和原始观测记录。
- `sparse_statistics.json`：注册图像数、注册比例、稀疏点数及平均点重投影误差；该平均值不是全角点 RMSE。
- `dense_raw.ply`：原始融合点云。
- `dense.ply`：至少被两个物体掩膜支持的点云，保留法向供网格生成。
- `mesh.ply`：Poisson 三角网格。
- `reconstruction_status.json`：阶段状态。
- `logs/`：命令记录与各阶段输出；出错时查看这里。

不同手机目录分别共享内参。若同一手机切换镜头或采集模式，应拆成不同目录。当前优先选择 `images.bin` 最大的子模型，输出子模型数量；不同子模型不会自动合并。输出路径中已有数据库时会拒绝继续，以免混入其他扫描；每轮使用新的扫描目录。

只测试稀疏重建可追加 `--sparse-only`。`--static-scene` 仅用于静止物体、移动相机的场景，不能用它绕过转台背景问题。

若默认设置只能注册少量图像，可为 `reconstruct` 追加 `--high-detail`：使用 CUDA SIFT、4096 最大图像边长、更低特征阈值和几何引导匹配。该选项要求支持 CUDA 的 COLMAP 与可用的 NVIDIA GPU。特征增加及注册成功不代表几何正确，仍需检查相机分布和点云；残留静止背景可能造成错误匹配。

使用 `--sparse-only` 停在稀疏阶段后，可先检查相机分布和物体形状，再用 `python scan.py resume-dense --images 原图目录 --masks 原掩膜目录 --output 同一个输出目录 --colmap 可执行文件路径` 从选定模型继续。续算不会重新运行特征提取、匹配或 SfM，且要求使用原来的掩膜目录。

`reconstruct`、`resume-dense` 和 `run` 均支持 `--patch-window-radius 10`，将所有参考图的稠密匹配窗口从默认半径 5（11×11）改为半径 10（21×21）。窗口为奇数边长，因此半径翻倍时边长是近似翻倍。窗口更大通常计算更慢，可能平滑细节；应对比几何与深度覆盖，不能只看新增点数。更改半径做对照时请使用新的输出目录，避免 COLMAP 跳过已有的旧参数深度图。

网格生成后会检查是否存在顶点和三角面。默认 Poisson 裁剪得到空网格，或剩余顶点不足输入点数的 1% 时，自动用深度 10、裁剪阈值 5 重试；仍为空时再用阈值 0，并在状态文件记录参数。该数量检查只检测严重裁剪，不保证形状正确。未裁剪表面可能延伸到没有观测支持的区域。

## 真实照片：第四步质量分析

```powershell
python scan.py analyze --mesh outputs/scan01/reconstruction/mesh.ply --cloud outputs/scan01/reconstruction/dense.ply --cameras outputs/scan01/reconstruction/cameras.json --output outputs/scan01/quality
```

实现的指标：

1. **附近是否有稠密点支持**：计算最近点距离，超过物体包围盒对角线的 2.5% 标为低支持。
2. **点云是否稀疏**：第 16 个近邻距离超过对角线的 9% 标为稀疏。这是局部间距指标，不是精确单位面积点密度。
3. **可见视角是否足够**：检查图像范围、前方深度、掩膜、网格射线遮挡和擦边视角；少于两个有效方向或最大方向差小于 10° 标为视角不足。
4. **开放边界**：找仅被一个三角面使用的边，组成连通边界。只有闭合环才计算局部平面投影面积，该面积是近似值。
5. **夹具遮挡**：若某观测相机记录带 `fixture_mesh`，程序读取该照片姿态下、已变换到物体坐标系的夹具网格并做射线检查。没有夹具模型时标为未知，不能将孔洞直接认定为夹具导致。

阈值是第一版启发式参数，仍需在实拍数据上验证。默认最多采样 1200 个三角面中心，CSV 和颜色不是每个顶点的独立精确测量。可用 `--max-samples` 调整。采样面的顶点颜色用于可视化，未采样区域的颜色不能视为逐点质量结论。

几何可见性只代表“从这个方向可能看到”，不证明图像匹配成功。稠密点也可能出错。没有深度真值时不能给出真实毫米误差。Poisson 会封补表面，因此无孔洞并不等于扫描完整。凹陷内部完全不可见时，程序只能提示未知和低支持，不能推断真实内部形状。

纯 NumPy 射线检测采用分块处理，适合小模型和第一版验证；大型网格、很多照片和候选姿态会较慢。正式扫描可先使用保留凹陷与边界的低面数模型分析，不应把简化产生的孔洞误当成采集缺陷。

## 真实照片：第五步补拍推荐

先复制演示配置作为模板，并完成实际坐标标定：

```powershell
Copy-Item outputs/demo/platform_demo.json platform.json
```

必须修改：

- `cameras`：固定相机在平台零位坐标系中的内外参；不能直接拿任意物体坐标系的 SfM 外参替代。
- `platform_from_model`：将当前 SfM 模型映射到平台零位坐标系的 4×4 矩阵，可包含统一正尺度。SfM 没有天然的毫米尺度，需要通过标记或已知尺寸配准。
- `yaw_axis`、`yaw_pivot`、`pitch_axis`、`pitch_pivot`：零位的轴方向与轴上点。
- `pitch_limits`、候选角度、`current_pose`、`captured_poses`：实际机构范围、当前角度和已拍角度。
- 可选 `fixture_mesh`：固定于平台世界坐标系的夹具遮挡模型。随物体或不同轴运动的夹具需要对应姿态变换，不能直接当成静止网格。

约定：内层 Pitch 后接外层 Yaw；变换为 `T_yaw @ T_pitch @ platform_from_model`。若机构轴顺序不同，需要修改变换定义。实测配置校验完成后才将 `calibrated` 改为 `true`；改标志本身不会完成标定。

```powershell
python scan.py plan --mesh outputs/scan01/reconstruction/mesh.ply --quality outputs/scan01/quality/quality_regions.json --platform platform.json --output outputs/scan01/planning
```

第一版评分优先让低支持区域与孔洞边界邻域朝向相机，并考虑多相机可见性和运动代价。随后逐次选择能补充尚未覆盖目标的姿态。已经拍过的姿态与超出俯仰范围的姿态会排除。

夹具模型可用于遮挡扣除，但没有模型时明确输出未知。没有电机控制、机构碰撞或载荷验证。所有实际运动仍由人确认执行。

## 一键真实流程

```powershell
python scan.py run --images data/images --masks data/masks --platform platform.json --output outputs/scan01
```

没有已标定平台配置时，先不传 `--platform`，完成重建与分析；不要将演示相机布局用于实际角度控制。

推荐补拍之后，将新旧照片合并到下一轮图像目录，制作对应掩膜，使用新的输出目录重新运行。本版采用整组重新重建，没有宣称已实现增量 SfM、跨轮网格配准或纹理烘焙。

## 验证与边界

```powershell
python -m unittest discover -s tests -v
```

当前 6 项测试通过：二进制 PLY 及法向读取、前景遮挡射线、COLMAP 文本参数和空观测行、绕指定轴心旋转、合成质量分析及未标定推荐保护，以及用模拟 COLMAP 进程检查阶段衔接和法向保留。模拟进程测试不代表真实 SfM/MVS 精度验证。真实调用路径已实现，但因没有实拍图像、未安装 COLMAP，尚未完成真实数据端到端验证。

当前不提供自动语义分割、物体类别识别、自动夹具识别、透明玻璃可靠重建、机械控制或纹理网格。建议先用不透明、刚性、有纹理的小物体验证真实流程。
