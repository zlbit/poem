import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from evaluate_poems import evaluate
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def collect_all_results(base_path='data/generated_poems'):
    records = []
    for folder in os.listdir(base_path):
        path = os.path.join(base_path, folder)
        if os.path.isdir(path):
            stats = evaluate(path)
            stats['strategy'] = folder
            records.append(stats)
    return pd.DataFrame(records)


def plot_heatmap(df):
    df_plot = df.set_index('strategy')[['avg_repeat_rate', 'avg_length', 'unique_ratio']]
    plt.figure(figsize=(10, 6))
    sns.heatmap(df_plot, annot=True, fmt=".2f", cmap="YlGnBu")
    plt.title("不同采样策略下的生成指标对比")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    df = collect_all_results()
    print(df)
    plot_heatmap(df)
