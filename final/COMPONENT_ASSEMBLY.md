# 按空间组件逐组拼接的实验

以 `group_0001` 为固定基准。后续每组先按 3 mm 三维体素的 26 邻域划分空间连通分量，只对至少 200 点的组件单独处理。每个组件的白光照片 SIFT 特征必须有有效结构光深度，并与已经组装的模型形成跨视角三维对应。只有两个以上参考组支持、匹配点分布足够广、刚体拟合改善明显、组件移动不超过 3 mm 时，才更新该组件的位置。证据不足时组件仍用原组位姿。

```powershell
cd "C:\Users\lenovo\Desktop\智能视觉信息采集\final"

.\.venv\Scripts\python.exe tools/assemble_depth_components.py `
  outputs/structured_light/structured_depth_mask_inset6_v1 `
  --initial-registration outputs/structured_light/structured_depth_mask_inset6_v1/sequential_registration_v1/registration_closed.json `
  --output outputs/structured_light/structured_depth_mask_inset6_v1/component_assembly_new `
  --voxel-mm 3 `
  --min-component-points 200

.\.venv\Scripts\python.exe tools/fuse_component_assembly.py `
  outputs/structured_light/structured_depth_mask_inset6_v1/component_assembly_new `
  --output outputs/structured_light/structured_depth_mask_inset6_v1/component_assembly_new/fused_components.ply
```

`component_registration.json` 记录每块的点数、匹配数、是否调整、拒绝原因和变换矩阵；`group_NNNN/component_XXX.ply` 可单独打开；`all_components_assembled.ply` 保留所有原始深度点；`fused_components.ply` 是实验性体素融合，要求同一体素至少两个原始拍摄组支持，但没有正式流程的法向一致性过滤。

本次 31 组共划出 589 个至少 200 点的组件，只有 5 个大组件通过跨视角纹理验证并发生小于约 1.1 mm 的调整。group16 中手持细杆候选点约 73% 落在连接手或身体的最大组件里；另有约 141 点落在一个 404 点的小组件，这块只有 3 个有效纹理特征、0 个可信的跨视角匹配。因此这轮没有解决手持物重影，实验结果不替换正式点云。
