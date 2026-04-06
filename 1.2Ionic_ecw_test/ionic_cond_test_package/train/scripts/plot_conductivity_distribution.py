"""
生成Li基卤化物电导率分布图
修改: 所有字体改为加粗的 Arial
"""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# 设置matplotlib字体和样式 - 使用 Arial 加粗
plt.rcParams['font.sans-serif'] = ['Arial']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 14
plt.rcParams['font.weight'] = 'bold'  # 全局字体加粗

# 路径配置（电导率数据 CSV 文件）
INPUT_DIR = Path(r"D:\workspace\test_dagang\test_ionic_0304_final\ionic_cond_20feat\data\processed\features_20feat.csv")
OUTPUT_DIR = Path(r"D:\workspace\test_dagang\test_ionic_0304_final\ionic_cond_20feat\reports")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_ionic_conductivity_data():
    """从features_20feat.csv中提取离子电导率数据"""
    print("=" * 80)
    print("开始提取离子电导率数据...")
    print("=" * 80)

    all_data = []
    csv_file = INPUT_DIR
    print(f"读取CSV文件: {csv_file.name}\n")

    try:
        df_raw = pd.read_csv(csv_file)

        for _, row in df_raw.iterrows():
            formula = row.get('formula', '')
            log10_cond = row.get('log10_ionic_conductivity')

            if pd.isna(log10_cond):
                continue

            # log10(σ / mS·cm⁻¹) → mS/cm
            value_in_mScm = 10 ** float(log10_cond)

            if value_in_mScm <= 0:
                continue

            all_data.append({
                'paper_file': str(row.get('record_id', '')),
                'title': '',
                'formula': formula,
                'space_group': '',
                'conductivity_mScm': value_in_mScm,
                'original_value': log10_cond,
                'original_unit': 'log10(mS/cm)',
                'temperature': 'RT',
                'measurement_conditions': ''
            })

    except Exception as e:
        print(f"[WARNING] 处理文件 {csv_file.name} 时出错: {e}")

    df = pd.DataFrame(all_data)
    print(f"\n[SUCCESS] 成功提取 {len(df)} 条电导率数据")
    return df


def convert_to_mScm(value, unit):
    """将不同单位的电导率值转换为 mS/cm"""
    if value is None:
        return None

    # 处理字符串格式的科学计数法
    if isinstance(value, str):
        value = value.strip()
        value = value.replace('x10^', 'e').replace('×10^', 'e')
        value = value.replace('×10', 'e10')

    try:
        value = float(value)
    except:
        return None

    unit = unit.strip().lower()

    # S/cm -> mS/cm: 乘以1000
    if 's/cm' in unit and 'ms' not in unit and 'μs' not in unit and 'us' not in unit:
        return value * 1000
    elif 's cm-1' in unit and 'ms' not in unit and 'μs' not in unit and 'us' not in unit:
        return value * 1000
    elif 's·cm-1' in unit and 'ms' not in unit and 'μs' not in unit and 'us' not in unit:
        return value * 1000
    elif 's cm^-1' in unit and 'ms' not in unit and 'μs' not in unit and 'us' not in unit:
        return value * 1000
    elif 'ω^{-1} cm^{-1}' in unit or 'ω⁻¹ cm⁻¹' in unit:
        return value * 1000

    # mS/cm -> mS/cm: 不变
    elif 'ms/cm' in unit or 'ms cm-1' in unit or 'ms·cm-1' in unit or 'ms cm^-1' in unit:
        return value

    # μS/cm -> mS/cm: 乘以0.001
    elif 'μs/cm' in unit or 'us/cm' in unit or 'μs cm-1' in unit or 'us cm-1' in unit:
        return value * 0.001

    # S/m -> mS/cm: 乘以10
    elif 's/m' in unit or 's m-1' in unit or 's·m-1' in unit:
        return value * 10

    # mS/m -> mS/cm: 乘以0.01
    elif 'ms/m' in unit:
        return value * 0.01

    # 默认假设为 S/cm
    else:
        return value * 1000


def plot_conductivity_distribution_new(df, output_dir):
    """
    绘制电导率分布图（原始值，不取log）
    修改: 所有字体改为加粗的 Arial
    """
    print("\n" + "=" * 80)
    print("绘制电导率分布图（新版 - 加粗 Arial 字体）...")
    print("=" * 80)

    conductivities = df['conductivity_mScm'].values

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制直方图
    n, bins, patches = ax.hist(
        conductivities,
        bins=30,
        color='#3498db',
        edgecolor='black',
        alpha=0.7,
        linewidth=1.2
    )

    # 让柱子"更高"（视觉上占满更多y轴区域）：收紧y轴上限
    if len(n) > 0:
        y_max = max(n) * 1.10  # 预留10%顶端空间，避免标注被裁剪
        ax.set_ylim(0, y_max)

    # 在每个柱上添加频数标签 - 加粗字体
    for i, (count, patch) in enumerate(zip(n, patches)):
        if count > 0:
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + (max(n) * 0.01 if len(n) > 0 else 0),
                f'{int(count)}',
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold',  # 加粗
                fontfamily='Arial'  # Arial 字体
            )

    # 设置标签 - 加粗 Arial 字体
    ax.set_xlabel('Ionic Conductivity (mS/cm) at RT',
                   fontsize=20,
                   fontweight='bold',  # 加粗
                   fontfamily='Arial')  # Arial 字体
    ax.set_ylabel('Count',
                   fontsize=20,
                   fontweight='bold',  # 加粗
                   fontfamily='Arial')  # Arial 字体

    # 设置刻度字体 - 加粗 Arial 字体
    ax.tick_params(axis='both', labelsize=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')  # 加粗
        label.set_fontfamily('Arial')  # Arial 字体

    # 右上角添加总数标注
    ax.text(
        0.97, 0.95,
        f'Total = {len(conductivities)}',
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=16,
        fontweight='bold',
        fontfamily='Arial'
    )

    # 取消网格，并让x轴从0开始
    ax.grid(False)
    ax.set_xlim(left=0)

    plt.tight_layout()

    # 保存图片
    output_path = output_dir / 'Li_halide_conductivity_distribution.png'
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"[SUCCESS] 图片已保存: {output_path}")
    print(f"[INFO] 图片尺寸: 12 x 6 英寸 (宽 x 高)")
    print(f"[INFO] 字体: 加粗 Arial")
    print(f"[INFO] 分辨率: 400 DPI")

    plt.close()

    # 打印统计信息
    print("\n" + "=" * 80)
    print("电导率统计信息:")
    print("=" * 80)
    print(f"数据点总数: {len(conductivities)}")
    print(f"最小值: {conductivities.min():.6f} mS/cm")
    print(f"最大值: {conductivities.max():.6f} mS/cm")
    print(f"平均值: {conductivities.mean():.6f} mS/cm")
    print(f"中位数: {np.median(conductivities):.6f} mS/cm")


def plot_conductivity_distribution_log10(df, output_dir):
    """
    绘制电导率分布图（取log10后的值）
    修改: 所有字体改为加粗的 Arial
    """
    print("\n" + "=" * 80)
    print("绘制电导率分布图（log10版 - 加粗 Arial 字体）...")
    print("=" * 80)

    conductivities = np.log10(df['conductivity_mScm'].values)

    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 6))

    # 绘制直方图
    n, bins, patches = ax.hist(
        conductivities,
        bins=30,
        color='#3498db',
        edgecolor='black',
        alpha=0.7,
        linewidth=1.2
    )

    # 让柱子"更高"（视觉上占满更多y轴区域）：收紧y轴上限
    if len(n) > 0:
        y_max = max(n) * 1.10  # 预留10%顶端空间，避免标注被裁剪
        ax.set_ylim(0, y_max)

    # 在每个柱上添加频数标签 - 加粗字体
    for i, (count, patch) in enumerate(zip(n, patches)):
        if count > 0:
            height = patch.get_height()
            ax.text(
                patch.get_x() + patch.get_width() / 2,
                height + (max(n) * 0.01 if len(n) > 0 else 0),
                f'{int(count)}',
                ha='center',
                va='bottom',
                fontsize=14,
                fontweight='bold',  # 加粗
                fontfamily='Arial'  # Arial 字体
            )

    # 设置标签 - 加粗 Arial 字体
    #ax.set_xlabel(r'$\log_{10}(\sigma)$', fontsize=14, fontweight='bold')
    ax.set_xlabel(r'$\log_{10}(\sigma)$)',
                   fontsize=20,
                   fontweight='bold',  # 加粗
                   fontfamily='Arial')  # Arial 字体
    ax.set_ylabel('Count',
                   fontsize=20,
                   fontweight='bold',  # 加粗
                   fontfamily='Arial')  # Arial 字体

    # 设置刻度字体 - 加粗 Arial 字体
    ax.tick_params(axis='both', labelsize=18)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')  # 加粗
        label.set_fontfamily('Arial')  # Arial 字体

    # 右上角添加总数标注
    ax.text(
        0.97, 0.95,
        f'Total = {len(conductivities)}',
        transform=ax.transAxes,
        ha='right', va='top',
        fontsize=16,
        fontweight='bold',
        fontfamily='Arial'
    )

    # 取消网格
    ax.grid(False)

    plt.tight_layout()

    # 保存图片（新文件名，不覆盖原图）
    output_path = output_dir / 'Li_halide_conductivity_distribution_log10.png'
    plt.savefig(output_path, dpi=400, bbox_inches='tight')
    print(f"[SUCCESS] 图片已保存: {output_path}")
    print(f"[INFO] 图片尺寸: 12 x 6 英寸 (宽 x 高)")
    print(f"[INFO] 字体: 加粗 Arial")
    print(f"[INFO] 分辨率: 400 DPI")

    plt.close()

    # 打印统计信息
    print("\n" + "=" * 80)
    print("电导率统计信息 (log10):")
    print("=" * 80)
    print(f"数据点总数: {len(conductivities)}")
    print(f"最小值: {conductivities.min():.4f} log10(mS/cm)")
    print(f"最大值: {conductivities.max():.4f} log10(mS/cm)")
    print(f"平均值: {conductivities.mean():.4f} log10(mS/cm)")
    print(f"中位数: {np.median(conductivities):.4f} log10(mS/cm)")


def main():
    print("\n" + "=" * 80)
    print("生成Li基卤化物电导率分布图")
    print("=" * 80)

    # 提取数据
    df = extract_ionic_conductivity_data()

    if len(df) == 0:
        print("\n[ERROR] 未找到有效的电导率数据")
        return

    # 生成图表
    plot_conductivity_distribution_new(df, OUTPUT_DIR)
    plot_conductivity_distribution_log10(df, OUTPUT_DIR)

    print("\n" + "=" * 80)
    print("[SUCCESS] 完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()
