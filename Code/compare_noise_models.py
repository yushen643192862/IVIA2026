"""复现二至五次模型，生成比较图表，并将结果分析写入最终四次模型报告。

运行：python compare_noise_models.py
依赖：numpy、matplotlib。输入与拟合方法来自 fit_noise_regression.py。
所有模型使用同一批筛选数据、同一随机种子和完全相同的训练/验证划分。
"""

import argparse
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile

# Matplotlib 的字体缓存写到临时目录，图表输出到实验数据目录。
os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'lab1_matplotlib'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from fit_noise_regression import DEFAULT_SEED, OUTPUT_FOLDERS, latest_features_csv, run_regression


def read_predictions(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def save_figure(fig, directory, name):
    fig.savefig(directory / (name + '.png'), dpi=180, facecolor='white')
    fig.savefig(directory / (name + '.svg'), facecolor='white')
    plt.close(fig)


def draw_comparison(models, predictions, directory):
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none'})
    degrees = sorted(models)
    colors = {'train': '#2c6eaa', 'validation': '#d06021'}
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    for ax, metric, title in zip(axes, ['RMSE', 'R2'], ['RMSE (lower is better)', 'R-squared (higher is better)']):
        for split in ['train', 'validation']:
            key = 'training_metrics' if split == 'train' else 'validation_metrics'
            values = [models[d][key][metric] for d in degrees]
            ax.plot(degrees, values, 'o-', color=colors[split], label=split.title())
            if split == 'validation':
                for degree, value in zip(degrees, values):
                    ax.annotate(f'{value:.4f}', (degree, value), xytext=(0, 10),
                                textcoords='offset points', ha='center', fontsize=9)
        ax.axvline(4, color='#618466', linestyle=':', alpha=0.75)
        ax.set(title=title, xlabel='Polynomial degree', xticks=degrees)
        ax.margins(y=0.25)
        ax.grid(alpha=0.2)
        ax.legend()
    axes[0].set_ylabel('Residual-noise standard deviation (gray levels)')
    save_figure(fig, directory, 'degree_comparison')

    validation = {d: [r for r in predictions[d] if r['split'] == 'validation'] for d in degrees}
    all_values = [float(r[key]) for rows in validation.values() for r in rows
                  for key in ['noise_std_gray', 'predicted_noise_std_gray']]
    lower, upper = min(0, min(all_values)), max(all_values) * 1.07
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), layout='constrained')
    for ax, degree in zip(axes.flat, degrees):
        rows = validation[degree]
        truth = np.array([float(r['noise_std_gray']) for r in rows])
        predicted = np.array([float(r['predicted_noise_std_gray']) for r in rows])
        ax.plot([lower, upper], [lower, upper], '--', color='#777777', linewidth=1)
        ax.scatter(truth, predicted, s=30, alpha=0.8, color='#2c6eaa', edgecolor='white', linewidth=0.4)
        metric = models[degree]['validation_metrics']
        ax.set(title=f'Degree {degree}: RMSE={metric["RMSE"]:.4f}, R2={metric["R2"]:.4f}',
               xlabel='Observed residual-noise estimate', ylabel='Predicted noise',
               xlim=(lower, upper), ylim=(lower, upper))
        ax.set_aspect('equal', adjustable='box')
        ax.grid(alpha=0.18)
    fig.suptitle(f'Same {len(validation[4])} validation samples; dashed line: prediction = observation')
    save_figure(fig, directory, 'validation_predictions')

    rows = validation[4]
    error = np.array([float(r['prediction_minus_observed']) for r in rows])
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), layout='constrained')
    for ax, key, label in zip(axes, ['actual_exposure_us', 'actual_gain_db'],
                               ['Exposure (us, logarithmic axis)', 'Gain (dB)']):
        values = np.array([float(r[key]) for r in rows])
        ax.scatter(values, error, s=35, color='#2c6eaa', alpha=0.8)
        ax.axhline(0, color='#777777', linestyle='--', linewidth=1)
        ax.set(xlabel=label, ylabel='Prediction - observed estimate (gray levels)')
        ax.grid(alpha=0.2)
    axes[0].set_xscale('log')
    fig.suptitle('Selected quartic model: validation residuals')
    save_figure(fig, directory, 'quartic_residuals')


def build_analysis(models, predictions, seed):
    degrees = sorted(models)
    validation = {d: [r for r in predictions[d] if r['split'] == 'validation'] for d in degrees}
    max_error = {d: max(float(r['absolute_error']) for r in validation[d]) for d in degrees}
    best = min(degrees, key=lambda d: models[d]['validation_metrics']['RMSE'])
    q = models[4]
    count_train, count_val = q['training_metrics']['n'], q['validation_metrics']['n']
    lines = [
        '## 阶数比较与模型选择', '',
        '### 比较方法', '',
        f'四个模型均使用同一批 {q["accepted_records"]} 条通过筛选的数据，随机种子固定为 {seed}。',
        f'训练集为 {count_train} 条，验证集为 {count_val} 条；程序逐文件核对四个模型的划分完全一致。',
        '所有模型采用普通最小二乘法，标准化只使用训练集。模型阶数改变时，没有重新抽取验证样本。',
        '预测目标均为白墙 ROI 高斯平滑残差的标准差；两个自变量均为实际曝光时间和实际增益。', '',
        '本次选择以验证集 RMSE 为主要指标，同时检查 MAE、最大绝对误差和参数数量。',
        '在同一验证集上，MSE、RMSE 与 $R^2$ 都基于同一个误差平方和，因此其排序一致，不能当作三项独立证据。', '',
        '### 各模型结果', '',
        '| 阶数 | 参数数 | 训练 RMSE | 训练 $R^2$ | 验证 MAE | 验证 RMSE | 验证 $R^2$ | 验证最大绝对误差 |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
    ]
    for degree in degrees:
        model = models[degree]
        tr, va = model['training_metrics'], model['validation_metrics']
        name = f'**{degree} 次（最终选择）**' if degree == 4 else f'{degree} 次'
        lines.append(f'| {name} | {len(model["coefficients_original_units"])} | {tr["RMSE"]:.4f} | '
                     f'{tr["R2"]:.4f} | {va["MAE"]:.4f} | {va["RMSE"]:.4f} | {va["R2"]:.4f} | {max_error[degree]:.4f} |')
    lines += ['', '![各阶模型的训练与验证指标](../model_comparison/degree_comparison.png)', '',
              '图 1：训练 RMSE 随阶数增加而下降；验证指标用于判断增加复杂度是否改善当前未参与拟合样本的预测。', '',
              '### 为什么选择四次', '']
    quartic_rmse = q['validation_metrics']['RMSE']
    for degree in [2, 3]:
        old_rmse = models[degree]['validation_metrics']['RMSE']
        change = (old_rmse - quartic_rmse) / old_rmse * 100
        lines.append(f'- 相比{degree}次，四次验证 RMSE 从 {old_rmse:.4f} 降至 {quartic_rmse:.4f}，'
                     f'下降 {change:.2f}%。这说明在本次划分中，增加到四次能够描述更多曝光与增益的非线性关系。')
    f = models[5]
    five_rmse = f['validation_metrics']['RMSE']
    increase = (five_rmse / quartic_rmse - 1) * 100
    lines += [
        f'- 从四次到五次，参数由 15 个增加到 21 个，训练 RMSE 从 {q["training_metrics"]["RMSE"]:.4f} '
        f'降至 {f["training_metrics"]["RMSE"]:.4f}，但验证 RMSE 升至 {five_rmse:.4f}（增加 {increase:.2f}%）。'
        '更好的训练拟合没有转化为更好的验证 RMSE，表现出过拟合迹象。',
        f'- 五次并非所有指标都更差：验证 MAE 为 {f["validation_metrics"]["MAE"]:.4f}，'
        f'略低于四次的 {q["validation_metrics"]["MAE"]:.4f}；'
        f'但最大绝对误差从 {max_error[4]:.4f} 增加到 {max_error[5]:.4f}。'
        '这说明五次减小了平均绝对误差，却加重了部分较大误差，平方误差指标对此更敏感。', '',
        (f'**在本次考察的二至五次模型、当前数据和固定验证划分下，四次的验证 RMSE 最低（{quartic_rmse:.4f}），'
         f'验证 $R^2$ 最高（{q["validation_metrics"]["R2"]:.4f}），且比五次少 6 个参数，因此选用四次。**'
         if best == 4 else f'当前重新计算的最低验证 RMSE 来自 {best} 次；此前选四次的结论需要重新评估。'), '',
        '### 预测与观测值的比较', '',
        '![四个模型在相同验证集上的预测与观测值](../model_comparison/validation_predictions.png)', '',
        '图 2：横轴为单图残差法得到的噪声估计值，纵轴为模型预测值；四幅子图使用相同坐标范围。'
        '点越接近虚线 $y=x$，预测越准确。这里的观测值是噪声代理量，不是独立标定的传感器真实噪声。', '',
        '### 四次模型的误差分析', '',
        '![四次模型残差与曝光、增益的关系](../model_comparison/quartic_residuals.png)', '',
        '图 3：残差定义为“预测值减观测值”。正值表示高估，负值表示低估；曝光轴使用对数尺度以展示不同数量级。', '',
    ]
    errors = np.array([float(r['prediction_minus_observed']) for r in validation[4]])
    lines += [f'四次模型验证残差的平均值为 {errors.mean():.4f} 灰度级，范围为 '
              f'[{errors.min():.4f}, {errors.max():.4f}]。以下列出绝对误差最大的 5 个验证样本，便于回看原图：', '',
              '| 图片 | 曝光（μs） | 增益（dB） | 观测噪声 | 预测噪声 | 残差 |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for row in sorted(validation[4], key=lambda r: float(r['absolute_error']), reverse=True)[:5]:
        lines.append(f'| `{row["filename"]}` | {float(row["actual_exposure_us"]):g} | '
                     f'{float(row["actual_gain_db"]):.4f} | {float(row["noise_std_gray"]):.4f} | '
                     f'{float(row["predicted_noise_std_gray"]):.4f} | {float(row["prediction_minus_observed"]):+.4f} |')
    short = np.array([float(r['actual_exposure_us']) <= 1000 for r in validation[4]])
    if short.any() and (~short).any():
        squared_error_share = float(np.sum(errors[short] ** 2) / np.sum(errors ** 2))
        lines += ['', '为定位误差来源，另按曝光是否超过 1000 μs 对验证样本作事后分组。'
                  '这个分组只用于诊断，没有重新筛选样本或修改拟合结果。', '',
                  '| 验证样本分组 | 数量 | RMSE | 平均残差 |', '| --- | ---: | ---: | ---: |']
        for label, mask in [('曝光 ≤ 1000 μs', short), ('曝光 > 1000 μs', ~short)]:
            lines.append(f'| {label} | {int(mask.sum())} | '
                         f'{np.sqrt(np.mean(errors[mask] ** 2)):.4f} | {errors[mask].mean():+.4f} |')
        lines += ['', f'短曝光组占验证样本的 {short.mean():.2%}，却贡献了 {squared_error_share:.2%} 的验证误差平方和。'
                  '结合上面的最大误差样本与残差图，可以定位四次模型仍需改进的参数区域；'
                  '因此较高的总体 $R^2$ 并不表示所有曝光/增益组合都拟合良好。']
    lines += ['',
              '误差可能同时来自多项式近似不足、单图残差估计中残留的墙面纹理、空间相关性及相机图像处理。'
              '当前结果不能单独判定各因素的贡献，也不能用低残差证明物理噪声测量准确。', '',
              '### 结论的适用范围', '',
              '- “四次最好”仅指当前比较的模型在本次验证划分上的 RMSE 最低，不是对所有模型或新场景的普遍证明。',
              '- 验证集已用于选择阶数，不能再称为未使用过的独立测试集。若要检验选择的稳定性，可进一步使用交叉验证或新的独立数据。',
              '- 过曝、截黑等样本已根据预先的图像质量规则排除；结论不能外推到全部 696 组条件。',
              '- 当前噪声指标是白墙 ROI 的高频残差标准差，不是时间噪声，模型也不是传感器物理参数标定。', '',
              '### 可复现结果', '',
              '- [完整指标表](../model_comparison/metrics.csv)',
              '- [二次模型报告](../regression/report.md)',
              '- [三次模型报告](../regression_cubic/report.md)',
              '- [四次模型报告](../regression_quartic/report.md)',
              '- [五次模型报告](../regression_quintic/report.md)', '',
              '复现命令：`python compare_noise_models.py`。PNG 用于报告预览，同名 SVG 可用于清晰排版。',
              '本节代码、图表和分析文字在 OpenAI Codex 协助下生成；全部数值由本地实验数据计算。']
    return '\n'.join(lines) + '\n', max_error


def compare_models(input_csv, seed=DEFAULT_SEED):
    input_csv = Path(input_csv)
    root = input_csv.parent
    directory = root / 'model_comparison'
    directory.mkdir(parents=True, exist_ok=True)
    models, predictions = {}, {}
    for degree, folder in OUTPUT_FOLDERS.items():
        with contextlib.redirect_stdout(io.StringIO()):
            models[degree] = run_regression(input_csv, root / folder, seed, degree)
        predictions[degree] = read_predictions(root / folder / 'predictions.csv')
        print(f'Degree {degree}: validation RMSE={models[degree]["validation_metrics"]["RMSE"]:.6f}', flush=True)
    reference = [(r['filename'], r['split'], r['noise_std_gray'],
                  r['actual_exposure_us'], r['actual_gain_db']) for r in predictions[4]]
    for degree in models:
        current = [(r['filename'], r['split'], r['noise_std_gray'],
                    r['actual_exposure_us'], r['actual_gain_db']) for r in predictions[degree]]
        if current != reference or models[degree]['input_sha256'] != models[4]['input_sha256']:
            raise ValueError('Input data or train/validation split differs between models.')
    draw_comparison(models, predictions, directory)
    section, max_error = build_analysis(models, predictions, seed)
    (directory / 'analysis_section.md').write_text(section, encoding='utf-8')
    (directory / 'report.md').write_text(
        '# 噪声模型的阶数比较与选择\n\n' + section.replace('../model_comparison/', ''), encoding='utf-8')
    with (directory / 'metrics.csv').open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['degree', 'parameters', 'split', 'n',
                                                   'MAE', 'MSE', 'RMSE', 'R2', 'negative_predictions',
                                                   'max_absolute_error'])
        writer.writeheader()
        for degree, model in models.items():
            for split, key in [('train', 'training_metrics'), ('validation', 'validation_metrics')]:
                maximum = max(float(r['absolute_error']) for r in predictions[degree] if r['split'] == split)
                writer.writerow(dict(degree=degree, parameters=len(model['coefficients_original_units']),
                                     split=split, max_absolute_error=maximum, **model[key]))
    summary = {
        'input_sha256': models[4]['input_sha256'], 'seed': seed,
        'quartic_coefficients': models[4]['coefficients_original_units'],
        'selected_degree': 4,
        'lowest_validation_rmse_degree': min(models, key=lambda d: models[d]['validation_metrics']['RMSE']),
        'validation_max_absolute_errors': max_error,
        'same_split_verified': True,
    }
    (directory / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False), encoding='utf-8')
    # 本节可由回归脚本在同一输入和参数下自动保留；重复执行不重复追加。
    report_path = root / OUTPUT_FOLDERS[4] / 'report.md'
    marker = '\n<!-- MODEL_COMPARISON -->'
    base = report_path.read_text(encoding='utf-8').split(marker, 1)[0].rstrip()
    report_path.write_text(base + '\n' + marker + '\n\n' + section, encoding='utf-8')
    print(f'Comparison and analysis saved: {report_path}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path)
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    args = parser.parse_args()
    try:
        compare_models(args.input if args.input is not None else latest_features_csv(), args.seed)
        return 0
    except (OSError, ValueError, np.linalg.LinAlgError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
