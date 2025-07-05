import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import matplotlib.font_manager as fm
from matplotlib import rcParams
import numpy as np

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 延迟导入，确保路径已设置
from evaluate_poems import evaluate


# ==================== 中文字体设置 - 更可靠的解决方案 ====================
def set_chinese_font():
    """设置中文字体，返回字体属性对象"""
    # 1. 尝试查找系统中可用的中文字体
    font_paths = [
        'C:/Windows/Fonts/simhei.ttf',  # Windows 黑体
        'C:/Windows/Fonts/simsun.ttc',  # Windows 宋体
        'C:/Windows/Fonts/msyh.ttc',  # Windows 微软雅黑
        '/System/Library/Fonts/PingFang.ttc',  # macOS 苹方
        '/System/Library/Fonts/STHeiti.ttf',  # macOS 华文黑体
        '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'  # Linux 通用中文字体
    ]

    # 2. 添加项目字体目录
    project_font_dir = os.path.join(os.path.dirname(__file__), '..', 'fonts')
    if os.path.exists(project_font_dir):
        for font_file in os.listdir(project_font_dir):
            if font_file.lower().endswith(('.ttf', '.ttc', '.otf')):
                font_paths.append(os.path.join(project_font_dir, font_file))

    # 3. 尝试加载可用字体
    font_prop = None
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                font_prop = fm.FontProperties(fname=font_path)
                print(f"成功加载中文字体: {font_path}")
                break
            except Exception as e:
                print(f"加载字体失败: {font_path} - {e}")

    # 4. 设置全局字体
    if font_prop:
        # 添加字体到字体管理器
        fm.fontManager.addfont(font_path)

        # 设置全局字体
        rcParams['font.family'] = 'sans-serif'
        rcParams['font.sans-serif'] = [font_prop.get_name()]
        rcParams['axes.unicode_minus'] = False

        # 设置默认字体大小
        rcParams['font.size'] = 12
        rcParams['axes.titlesize'] = 14
        rcParams['axes.labelsize'] = 12
        rcParams['xtick.labelsize'] = 10
        rcParams['ytick.labelsize'] = 10
        rcParams['legend.fontsize'] = 10

        print(f"已设置中文字体: {font_prop.get_name()}")
    else:
        print("警告: 未找到中文字体文件，图表可能无法正确显示中文")

    return font_prop


# 设置中文字体
chinese_font = set_chinese_font()


# ==================== 以下代码保持不变 ====================

def collect_all_results():
    """收集所有生成诗歌的评估结果"""
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # 构建正确的数据路径（上一级目录中的data/generated_poems）
    base_path = os.path.join(script_dir, '..', 'data', 'generated_poems')
    base_path = os.path.normpath(base_path)  # 规范化路径

    print(f"正在从目录收集数据: {base_path}")

    if not os.path.exists(base_path):
        print(f"错误: 目录不存在 - {base_path}")
        return pd.DataFrame()

    records = []
    for folder in os.listdir(base_path):
        folder_path = os.path.join(base_path, folder)
        if os.path.isdir(folder_path):
            print(f"处理策略: {folder}")
            try:
                stats = evaluate(folder_path)
                stats['strategy'] = folder
                records.append(stats)
            except Exception as e:
                print(f"评估策略 {folder} 时出错: {e}")

    return pd.DataFrame(records)


def plot_heatmap(df):
    """绘制热图展示评估结果"""
    if df.empty:
        print("没有数据可绘制")
        return

    # 重命名列以更友好的名称显示
    column_mapping = {
        'avg_repeat_rate': '平均重复率',
        'avg_length': '平均长度',
        'unique_ratio': '独特词比例'
    }
    df_plot = df.rename(columns=column_mapping)

    # 提取需要绘制的列
    plot_columns = list(column_mapping.values())
    df_plot = df_plot.set_index('strategy')[plot_columns]

    # 创建热图
    plt.figure(figsize=(12, 8))
    ax = sns.heatmap(df_plot, annot=True, fmt=".2f", cmap="YlGnBu", linewidths=0.5)

    # 设置中文标题和标签 - 显式使用字体
    if chinese_font:
        plt.title("不同采样策略下的诗歌生成质量指标对比", fontsize=14, fontproperties=chinese_font)
        plt.xlabel('评估指标', fontsize=12, fontproperties=chinese_font)
        plt.ylabel('采样策略', fontsize=12, fontproperties=chinese_font)
    else:
        plt.title("Poem Generation Metrics by Strategy", fontsize=14)
        plt.xlabel('Metrics', fontsize=12)
        plt.ylabel('Strategy', fontsize=12)

    # 设置坐标轴标签旋转
    plt.xticks(rotation=45, ha='right', fontsize=10)
    plt.yticks(fontsize=10)

    plt.tight_layout()

    # 保存图像
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'poem_metrics_heatmap.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"热图已保存至: {output_path}")
    plt.show()


def plot_bar_charts(df):
    """绘制条形图展示评估结果 - 完全重写以确保中文显示"""
    if df.empty:
        print("没有数据可绘制")
        return

    # 设置绘图风格
    sns.set(style="whitegrid")

    # 创建图形和子图
    fig, axes = plt.subplots(3, 1, figsize=(15, 18))

    # 为策略名称添加换行符，避免重叠
    df['strategy_short'] = df['strategy'].apply(lambda x: '\n'.join([x[i:i + 15] for i in range(0, len(x), 15)]))

    # 设置颜色
    colors = sns.color_palette("muted", n_colors=len(df))

    # 图1: 平均重复率
    ax1 = axes[0]
    bars1 = ax1.bar(df['strategy_short'], df['avg_repeat_rate'], color=colors)
    if chinese_font:
        ax1.set_title('不同策略的平均重复率', fontsize=16, fontproperties=chinese_font)
        ax1.set_xlabel('采样策略', fontsize=14, fontproperties=chinese_font)
        ax1.set_ylabel('平均重复率', fontsize=14, fontproperties=chinese_font)
    else:
        ax1.set_title('Average Repetition Rate by Strategy', fontsize=16)
        ax1.set_xlabel('Strategy', fontsize=14)
        ax1.set_ylabel('Average Repetition Rate', fontsize=14)

    # 添加数值标签
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=10)

    # 图2: 平均长度
    ax2 = axes[1]
    bars2 = ax2.bar(df['strategy_short'], df['avg_length'], color=colors)
    if chinese_font:
        ax2.set_title('不同策略的平均长度', fontsize=16, fontproperties=chinese_font)
        ax2.set_xlabel('采样策略', fontsize=14, fontproperties=chinese_font)
        ax2.set_ylabel('平均长度', fontsize=14, fontproperties=chinese_font)
    else:
        ax2.set_title('Average Length by Strategy', fontsize=16)
        ax2.set_xlabel('Strategy', fontsize=14)
        ax2.set_ylabel('Average Length', fontsize=14)

    # 添加数值标签
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2., height + 2,
                 f'{height:.0f}',
                 ha='center', va='bottom', fontsize=10)

    # 图3: 独特词比例
    ax3 = axes[2]
    bars3 = ax3.bar(df['strategy_short'], df['unique_ratio'], color=colors)
    if chinese_font:
        ax3.set_title('不同策略的独特词比例', fontsize=16, fontproperties=chinese_font)
        ax3.set_xlabel('采样策略', fontsize=14, fontproperties=chinese_font)
        ax3.set_ylabel('独特词比例', fontsize=14, fontproperties=chinese_font)
    else:
        ax3.set_title('Unique Word Ratio by Strategy', fontsize=16)
        ax3.set_xlabel('Strategy', fontsize=14)
        ax3.set_ylabel('Unique Word Ratio', fontsize=14)

    # 添加数值标签
    for bar in bars3:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width() / 2., height + 0.005,
                 f'{height:.3f}',
                 ha='center', va='bottom', fontsize=10)

    # 旋转X轴标签
    for ax in axes:
        plt.sca(ax)
        plt.xticks(rotation=45, ha='right')

    plt.tight_layout(pad=5.0)

    # 保存图像
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments')
    output_path = os.path.join(output_dir, 'poem_metrics_barchart.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"条形图已保存至: {output_path}")
    plt.show()


def save_results_to_csv(df):
    """将结果保存为CSV文件"""
    if df.empty:
        print("没有数据可保存")
        return

    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'experiments')
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'poem_metrics_results.csv')

    df.to_csv(output_path, index=False, encoding='utf-8-sig')
    print(f"结果已保存至CSV文件: {output_path}")


if __name__ == '__main__':
    print("开始收集诗歌评估结果...")
    df = collect_all_results()

    if df.empty:
        print("未收集到任何结果")
    else:
        print("\n评估结果摘要:")
        print(df)

        # 保存结果到CSV
        save_results_to_csv(df)

        # 绘制图表
        plot_heatmap(df)
        plot_bar_charts(df)

        print("\n分析完成！")