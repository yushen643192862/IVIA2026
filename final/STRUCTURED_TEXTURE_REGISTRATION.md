# 结构光点云的纹理特征校验

这个实验从白光照片提取 SIFT 特征，只保留能关联到结构光三维点的特征。每组与已经配准的若干组建立描述子匹配，经三维距离门限和刚体 RANSAC 检验后，只有在至少两个参考视角支持、特征分布足够广、位姿移动较小、整体表面重合不退化时，才采用新的位姿。

```powershell
cd "C:\Users\lenovo\Desktop\智能视觉信息采集\final"

.\.venv\Scripts\python.exe tools/refine_texture_landmarks.py `
  outputs/structured_light/structured_depth_mask_inset6_v1 `
  --initial-registration outputs/structured_light/structured_depth_mask_inset6_v1/sequential_registration_v1/registration_closed.json `
  --output outputs/structured_light/structured_depth_mask_inset6_v1/texture_landmark_registration_new

.\.venv\Scripts\python.exe tools/fuse_structured_clouds.py `
  outputs/structured_light/structured_depth_mask_inset6_v1 `
  --output outputs/structured_light/structured_depth_mask_inset6_v1/fusion_texture_landmark_new `
  --dense-refine `
  --pose-override outputs/structured_light/structured_depth_mask_inset6_v1/texture_landmark_registration_new/registration.json
```

程序拒绝覆盖已有输出目录。纹理配准的逐组接受原因写在 `texture_report.json`；实验点云写在融合目录的 `dense.ply`。这一步校正的是整组刚体位姿，不能单独修复细杆局部测深或错误点，也不能保证没有可见纹理重影。

本次实验结果：31 组中 8 组通过；第 16 组只有一个参考视角支持，保持原位。实验融合输出 731,615 点，原版为 732,552 点。三视角预览没有显示手持物重影得到明确修复，所以当前实验结果不替换原版。
