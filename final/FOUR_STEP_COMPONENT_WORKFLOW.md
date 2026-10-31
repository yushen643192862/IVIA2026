# 分组、组件与多照片拼接实验

运行前先确认 `outputs/structured_light/structured_depth_mask_inset6_v1` 和原始白光照片仍在原位置。此流程以 `group_0001` 为固定基准，输出在新目录，不修改原版点云。

```powershell
cd "C:\Users\lenovo\Desktop\智能视觉信息采集\final"

.\.venv\Scripts\python.exe tools/assemble_depth_components.py `
  outputs/structured_light/structured_depth_mask_inset6_v1 `
  --initial-registration outputs/structured_light/structured_depth_mask_inset6_v1/sequential_registration_v1/registration_closed.json `
  --output outputs/structured_light/structured_depth_mask_inset6_v1/component_four_step_new `
  --voxel-mm 3 `
  --min-component-points 200

.\.venv\Scripts\python.exe tools/fuse_component_assembly.py `
  outputs/structured_light/structured_depth_mask_inset6_v1/component_four_step_new `
  --output outputs/structured_light/structured_depth_mask_inset6_v1/component_four_step_new/fused_components.ply

.\.venv\Scripts\python.exe tools/preview_cloud.py `
  outputs/structured_light/structured_depth_mask_inset6_v1/component_four_step_new/fused_components.ply `
  outputs/structured_light/structured_depth_mask_inset6_v1/component_four_step_new/fused_components_preview.png
```

处理规则：

1. 在掩膜边缘 6 像素内寻找比附近内部点深 6～25 mm 的点。仅当至少 8 个内部邻点的深度和颜色都一致时，复制它们的中位颜色；三维坐标不变。
2. 先按亮度、饱和度和色相划粗颜色类，再在每类内按 3 mm 体素的 26 邻域划空间连通组件。少于 200 点的碎片保留原始组位姿。
3. 每个组件与已经加入模型的三维 SIFT 纹理点匹配。仅接受有两个参考组支持、匹配分布足够广、移动不超过 3 mm 的刚体变换。
4. 纹理不足时，把组件投影到至多五张更早的白光照片。要求至少两张照片同时有前景掩膜、接近的实测深度和足够点数；只允许最多 2 mm 平移和 0.6° 旋转。至少两张照片的颜色误差要明显下降，任何参考照片都不能明显变差。

`component_registration.json` 记录每组、每块组件的接受原因、变换和照片误差；`group_NNNN/component_XXX.ply` 可以逐块打开；`all_components_assembled.ply` 保留全部点；`fused_components.ply` 是简单体素融合的实验预览，不包含正式流程的法向一致性过滤。

本次 `component_four_step_v3`：31 组共 1120 个达到门槛的组件，1 个通过三维纹理配准，91 个通过多照片小幅调整，7311 个掩膜边缘点改色。融合预览约 675,308 点。手持细杆主要组件未通过调整；其中 group16 的一块 398 点组件只有 3 个纹理特征，且仅在一张较早照片中有足够的深度一致支持。预览未证明细杆重影得到解决，因此不替换原版 `dense.ply`。
