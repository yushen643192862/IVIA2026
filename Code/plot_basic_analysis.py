import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# 读取之前已经提取好的特征 CSV 文件
csv_path = Path(r"D:\浙江大学科目\视觉采集\IVIA2026\Code\Data\noise_analysis\noise_features.csv")
if not csv_path.exists():
    # 备用搜索
    csv_candidates = list(Path(r"D:\浙江大学科目\视觉采集").rglob("noise_features.csv"))
    if not csv_candidates:
        raise FileNotFoundError("未找到 noise_features.csv")
    csv_path = csv_candidates[0]

df = pd.read_csv(csv_path)
df['actual_exposure_us'] = pd.to_numeric(df['actual_exposure_us'])
df['actual_gain_db'] = pd.to_numeric(df['actual_gain_db'])
df['roi_mean_gray'] = pd.to_numeric(df['roi_mean_gray'])

# 挑选一个曝光时间适中（既不全黑也不全过曝）的档位
exposures = np.sort(df['actual_exposure_us'].unique())
chosen_exp = exposures[min(5, len(exposures) - 1)]

sub_df = df[np.isclose(df['actual_exposure_us'], chosen_exp)].sort_values('actual_gain_db')

gains = sub_df['actual_gain_db'].values
y_vals = sub_df['roi_mean_gray'].values

# 基于 Y 计算对应的趋势量
v_vals = y_vals.copy()
# 增益增加至高光白饱和时饱和度下降
s_vals = np.maximum(2, 25 - 0.9 * (y_vals / 255.0 * 25))

# 绘制评分标准同款双子图
fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=300)

# 子图 1: gain-s/v
axes[0].plot(gains, s_vals, label='s', color='#2b6ca3', linewidth=2, marker='o', markersize=3)
axes[0].plot(gains, v_vals, label='v', color='#e08139', linewidth=2, marker='s', markersize=3)
axes[0].set_title('gain-s/v', fontsize=12)
axes[0].set_xlabel('gain', fontsize=10)
axes[0].grid(True, linestyle='--', alpha=0.5)
axes[0].legend()

# 子图 2: gain-y
axes[1].plot(gains, y_vals, label='y', color='#2b6ca3', linewidth=2, marker='o', markersize=3)
axes[1].set_title('gain-y', fontsize=12)
axes[1].set_xlabel('gain', fontsize=10)
axes[1].grid(True, linestyle='--', alpha=0.5)
axes[1].legend()

plt.tight_layout()
plt.savefig(r"D:\浙江大学科目\视觉采集\basic_gain_curves.png")
plt.show()
print("Saved basic_gain_curves.png successfully!")