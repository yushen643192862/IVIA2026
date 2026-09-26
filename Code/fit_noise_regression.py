"""默认拟合最终四次模型，同时支持二、三、五次对照模型。

模型：sigma = a0 + a1*e + a2*g + a3*e*g + a4*e**2 + a5*g**2
             + a6*e**3 + a7*e**2*g + a8*e*g**2 + a9*g**3
             + a10*e**4 + a11*e**3*g + a12*e**2*g**2 + a13*e*g**3 + a14*g**4
e 使用 actual_exposure_us（微秒），g 使用 actual_gain_db（dB），
目标 sigma 使用 noise_std_gray（高频残差的标准差，不是方差）。

运行：python fit_noise_regression.py
对照模型：python fit_noise_regression.py --degree 2（也支持 3、5）
完整对比及图表：python compare_noise_models.py
指定输入：python fit_noise_regression.py --input "路径/noise_features.csv"

只用 fit_candidate=1 且 status=ok 的记录。固定种子随机抽取 20%
作为验证集（向上取整），剩余 80% 训练。标准化参数和回归参数只从
训练集学习；不会在评估后使用全量数据重新拟合。
依赖：numpy。输出 report.md、model.json 和 predictions.csv。
"""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import sys

import numpy as np
from data_paths import latest_features


VALIDATION_FRACTION = 0.20
DEFAULT_SEED = 42
TERMS = ['1', 'e', 'g', 'e*g', 'e^2', 'g^2',
         'e^3', 'e^2*g', 'e*g^2', 'g^3',
         'e^4', 'e^3*g', 'e^2*g^2', 'e*g^3', 'g^4',
         'e^5', 'e^4*g', 'e^3*g^2', 'e^2*g^3', 'e*g^4', 'g^5']
POWERS = [(0, 0), (1, 0), (0, 1), (1, 1), (2, 0), (0, 2),
          (3, 0), (2, 1), (1, 2), (0, 3),
          (4, 0), (3, 1), (2, 2), (1, 3), (0, 4),
          (5, 0), (4, 1), (3, 2), (2, 3), (1, 4), (0, 5)]
DEGREE_NAMES = {2: '二', 3: '三', 4: '四', 5: '五'}
OUTPUT_FOLDERS = {2: 'regression', 3: 'regression_cubic',
                  4: 'regression_quartic', 5: 'regression_quintic'}


def polynomial_terms(degree=4):
    if degree not in DEGREE_NAMES:
        raise ValueError('Degree must be 2, 3, 4 or 5.')
    return TERMS[:(degree + 1) * (degree + 2) // 2]


def formula_markdown(degree):
    terms = []
    for index, (i, j) in enumerate(POWERS[:len(polynomial_terms(degree))]):
        term = f'a_{{{index}}}'
        for symbol, power in [('e', i), ('g', j)]:
            if power:
                term += symbol if power == 1 else f'{symbol}^{{{power}}}'
        terms.append(term)
    groups = [terms[:6], terms[6:10], terms[10:15], terms[15:21]]
    groups = [group for group in groups if group]
    lines = ['$$', r'\begin{aligned}']
    for index, group in enumerate(groups):
        prefix = r'\hat{\sigma} ={}& ' if index == 0 else '&+'
        suffix = r' \\' if index < len(groups) - 1 else ''
        lines.append(prefix + '+'.join(group) + suffix)
    return lines + [r'\end{aligned}', '$$']


def latest_features_csv():
    return latest_features(Path(__file__).resolve().parent)


def load_filtered_data(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        required = {'filename', 'actual_exposure_us', 'actual_gain_db',
                    'noise_std_gray', 'fit_candidate', 'status'}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f'CSV must contain {sorted(required)}')
        all_rows = list(reader)
    rows = [row for row in all_rows
            if row['fit_candidate'].strip() == '1' and row['status'].strip() == 'ok']
    if len(rows) < 10:
        raise ValueError('Need at least 10 accepted records for training and validation.')
    if len({row['filename'] for row in rows}) != len(rows):
        raise ValueError('Duplicate accepted filenames; check the input data.')
    x = np.array([[float(row['actual_exposure_us']), float(row['actual_gain_db'])]
                  for row in rows], dtype=np.float64)
    y = np.array([float(row['noise_std_gray']) for row in rows], dtype=np.float64)
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError('Accepted rows contain non-finite numbers.')
    if np.any(x[:, 0] <= 0) or np.any(y < 0):
        raise ValueError('Exposure must be positive and noise standard deviation nonnegative.')
    return rows, x, y, len(all_rows)


def design_matrix(x, degree=4):
    """构造完整多项式特征；模型对待求系数仍然是线性的。"""
    e, g = x[:, 0], x[:, 1]
    return np.column_stack([e ** i * g ** j for i, j in POWERS[:len(polynomial_terms(degree))]])


def original_unit_coefficients(b, center, scale, degree=4):
    """按二项式定理把标准化多项式展开回 e（us）、g（dB）。"""
    me, mg = center
    se, sg = scale
    powers = POWERS[:len(polynomial_terms(degree))]
    indices = {power: index for index, power in enumerate(powers)}
    a = np.zeros(len(powers), dtype=np.float64)
    for coefficient, (i, j) in zip(b, powers):
        for p in range(i + 1):
            for q in range(j + 1):
                a[indices[(p, q)]] += (
                    coefficient * math.comb(i, p) * math.comb(j, q)
                    * (-me) ** (i - p) * (-mg) ** (j - q) / (se ** i * sg ** j)
                )
    return a


def evaluate(y, prediction):
    error = prediction - y
    mse = float(np.mean(error ** 2))
    total_variation = float(np.sum((y - np.mean(y)) ** 2))
    return {
        'n': len(y),
        'MAE': float(np.mean(np.abs(error))),
        'MSE': mse,
        'RMSE': math.sqrt(mse),
        'R2': 1.0 - float(np.sum(error ** 2)) / total_variation
              if total_variation > 0 else None,
        'negative_predictions': int(np.sum(prediction < 0)),
    }


def fit_model(x, y, seed=DEFAULT_SEED, degree=4):
    terms = polynomial_terms(degree)
    n_coefficients = len(terms)
    n_validation = math.ceil(len(y) * VALIDATION_FRACTION)
    indices = np.random.default_rng(seed).permutation(len(y))
    validation_idx, train_idx = indices[:n_validation], indices[n_validation:]
    if len(train_idx) < n_coefficients or len(validation_idx) < 2:
        raise ValueError(f'Not enough samples for {n_coefficients} coefficients and validation.')
    # 验证集不参与标准化参数的计算，防止数据泄漏。
    center = x[train_idx].mean(axis=0)
    scale = x[train_idx].std(axis=0)
    if np.any(scale <= 0):
        raise ValueError('Training exposure and gain must each vary.')
    train_design = design_matrix((x[train_idx] - center) / scale, degree)
    b, _, rank, singular_values = np.linalg.lstsq(train_design, y[train_idx], rcond=None)
    if rank != n_coefficients:
        raise ValueError(f'Training matrix has rank {rank}, not {n_coefficients}; coefficients are not identifiable.')
    a = original_unit_coefficients(b, center, scale, degree)
    prediction = design_matrix((x - center) / scale, degree) @ b
    # 检查展开后的公式与标准化计算等价。
    if not np.allclose(design_matrix(x, degree) @ a, prediction, rtol=1e-8, atol=1e-10):
        raise ValueError('Original-unit coefficient conversion failed verification.')
    model = {
        'formula': 'sigma = a0 + ' + ' + '.join(f'a{i}*{term}' for i, term in enumerate(terms) if i),
        'polynomial_degree': degree,
        'selected_model': degree == 4,
        'method': 'Ordinary least squares, no regularization, no clipping of predictions',
        'features': {'e': 'actual_exposure_us (microseconds)', 'g': 'actual_gain_db (dB)'},
        'target': 'noise_std_gray: single-image high-frequency residual standard deviation',
        'filter': 'fit_candidate == 1 and status == ok',
        'seed': seed, 'validation_fraction_requested': VALIDATION_FRACTION,
        'validation_rounding': 'ceil',
        'validation_fraction_actual': len(validation_idx) / len(y),
        'coefficients_original_units': {f'a{i}': float(v) for i, v in enumerate(a)},
        'coefficient_term_order': terms,
        'standardization_training_only': {'center_e_g': center.tolist(), 'scale_e_g': scale.tolist()},
        'coefficients_standardized': b.tolist(),
        'training_matrix_rank': int(rank),
        'training_matrix_condition_number': float(singular_values[0] / singular_values[-1]),
        'training_ranges': {
            'exposure_us': [float(x[train_idx, 0].min()), float(x[train_idx, 0].max())],
            'gain_db': [float(x[train_idx, 1].min()), float(x[train_idx, 1].max())],
        },
        'training_metrics': evaluate(y[train_idx], prediction[train_idx]),
        'validation_metrics': evaluate(y[validation_idx], prediction[validation_idx]),
        'limitations': 'Empirical fit to a spatial residual proxy, not temporal noise. '
                       'Random holdout evaluates this dataset; it does not establish extrapolation '
                       'to other scenes or exposure/gain ranges. Negative predictions are not clipped. '
                       'The validation set was used for degree selection; it is not an independent test set.',
    }
    return model, prediction, train_idx, validation_idx


def run_regression(input_csv, output_dir, seed=DEFAULT_SEED, degree=4):
    rows, x, y, source_count = load_filtered_data(input_csv)
    model, prediction, train_idx, validation_idx = fit_model(x, y, seed, degree)
    model.update(input_csv=str(Path(input_csv).resolve()), source_records=source_count,
                 accepted_records=len(rows), excluded_records=source_count - len(rows))
    model['input_sha256'] = hashlib.sha256(Path(input_csv).read_bytes()).hexdigest()
    output_dir = Path(output_dir)
    predictions_path = output_dir / 'predictions.csv'
    if predictions_path.resolve() == Path(input_csv).resolve():
        raise ValueError('Output must not overwrite the source CSV.')
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'model.json').write_text(
        json.dumps(model, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    split = np.full(len(rows), 'train', dtype='<U10')
    split[validation_idx] = 'validation'
    with predictions_path.open('w', encoding='utf-8-sig', newline='') as stream:
        fields = ['filename', 'split', 'actual_exposure_us', 'actual_gain_db',
                  'noise_std_gray', 'predicted_noise_std_gray', 'prediction_minus_observed',
                  'absolute_error', 'squared_error']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for i, row in enumerate(rows):
            error = float(prediction[i] - y[i])
            writer.writerow({
                'filename': row['filename'], 'split': split[i],
                'actual_exposure_us': x[i, 0], 'actual_gain_db': x[i, 1],
                'noise_std_gray': y[i], 'predicted_noise_std_gray': prediction[i],
                'prediction_minus_observed': error,
                'absolute_error': abs(error), 'squared_error': error ** 2,
            })

    lines = [
        f'# 二元{DEGREE_NAMES[degree]}次噪声模型回归报告', '',
        ('最终选用模型。' if degree == 4 else '用于与最终四次模型进行对照。')
        + f'采用普通最小二乘法求解完整二元{DEGREE_NAMES[degree]}次多项式的 {len(polynomial_terms(degree))} 个系数。', '',
        '## 数据与划分', '',
        '| 项目 | 数值 |', '| --- | ---: |',
        f'| 输入总数 | {source_count} |',
        f'| 通过筛选 | {len(rows)} |',
        f'| 排除 | {source_count - len(rows)} |',
        f'| 训练集 | {len(train_idx)} |',
        f'| 验证集 | {len(validation_idx)} |',
        f'| 随机种子 | {seed} |', '',
        '仅使用 `fit_candidate=1` 且 `status=ok` 的记录。随机抽取 20%（向上取整）',
        '作为验证集，其余作为训练集；标准化参数和模型系数均只使用训练集求解。', '',
        '## 模型与变量', '',
        '| 变量 | 数据字段 | 含义及单位 |', '| --- | --- | --- |',
        '| $e$ | `actual_exposure_us` | 实际曝光时间，μs |',
        '| $g$ | `actual_gain_db` | 实际增益，dB |',
        r'| $\sigma$ | `noise_std_gray` | 白墙高频残差标准差，灰度级 |', '',
        *formula_markdown(degree), '',
        '## 原始单位下的系数', '',
        '以下系数已从标准化坐标转换回原始单位；代入公式时曝光使用微秒，增益使用 dB。', '',
        '| 系数 | 对应项 | 数值 |', '| --- | --- | ---: |',
    ]
    lines += [f'| `{name}` | `{term}` | {value:.16g} |'
              for (name, value), term in
              zip(model['coefficients_original_units'].items(), model['coefficient_term_order'])]
    lines += ['', '## 拟合效果', '',
              '| 指标 | 训练集 | 验证集 |', '| --- | ---: | ---: |']
    for name in ('MAE', 'MSE', 'RMSE', 'R2', 'negative_predictions'):
        label = {'R2': '$R^2$', 'negative_predictions': '负预测值数量'}.get(name, name)
        values = [model[key][name] for key in ('training_metrics', 'validation_metrics')]
        formatted = ['未定义' if value is None else f'{value:.16g}' for value in values]
        lines.append(f'| {label} | {formatted[0]} | {formatted[1]} |')
    lines += ['', 'MAE 和 RMSE 的单位为灰度级，MSE 的单位为灰度级平方；$R^2$ 无量纲。',
              '', '## 说明与适用范围', '',
              '- 验证集不参与参数求解；它曾用于阶数比较，因此其指标不是独立测试集结果。',
              '- 预测值未强制截为非负数，负预测值数量见上表。',
              '- 模型针对当前场景中通过筛选的数据，不应直接外推到全部曝光/增益组合。',
              '- 拟合目标是单图空间高频残差噪声代理值，不是严格的传感器时间噪声。',
              '', '## 结果文件', '',
              '- [模型参数与配置](model.json)',
              '- [逐样本预测、误差及训练/验证划分](predictions.csv)',
              '- [噪声与亮度提取数据](../noise_features.csv)']
    # 同一数据和划分重跑四次模型时保留比较分析，避免引用旧数据的比较结论。
    comparison_dir = output_dir.parent / 'model_comparison'
    summary_path = comparison_dir / 'summary.json'
    section_path = comparison_dir / 'analysis_section.md'
    if degree == 4 and summary_path.is_file() and section_path.is_file():
        summary = json.loads(summary_path.read_text(encoding='utf-8'))
        if (summary.get('input_sha256') == model['input_sha256']
                and summary.get('seed') == seed
                and summary.get('quartic_coefficients') == model['coefficients_original_units']):
            lines += ['', '<!-- MODEL_COMPARISON -->', '', section_path.read_text(encoding='utf-8')]
    (output_dir / 'report.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Accepted: {len(rows)} / {source_count}; train={len(train_idx)}, validation={len(validation_idx)}')
    for name, value in model['coefficients_original_units'].items():
        print(f'{name} = {value:.12g}')
    for key in ('training_metrics', 'validation_metrics'):
        print(key + ': ' + json.dumps(model[key]))
    print(f'Saved: {output_dir}', flush=True)
    return model


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--input', type=Path, help='noise_features.csv path')
    parser.add_argument('--output-dir', type=Path, help='Output directory')
    parser.add_argument('--seed', type=int, default=DEFAULT_SEED)
    parser.add_argument('--degree', type=int, choices=(2, 3, 4, 5), default=4)
    args = parser.parse_args()
    try:
        input_csv = args.input if args.input is not None else latest_features_csv()
        output_dir = args.output_dir if args.output_dir is not None else input_csv.parent / OUTPUT_FOLDERS[args.degree]
        run_regression(input_csv, output_dir, args.seed, args.degree)
        return 0
    except (OSError, ValueError, np.linalg.LinAlgError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
