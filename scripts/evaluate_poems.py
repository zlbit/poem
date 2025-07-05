import os
import sys
import json
from collections import Counter

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def calc_repeat_rate(text):
    """计算文本的字符重复率"""
    if not text:
        return 0.0
    unique_chars = len(set(text))
    return 1 - unique_chars / len(text)


def calc_unique_word_ratio(text):
    """计算文本的独特词比例"""
    if not text:
        return 0.0
    words = text.split()
    if not words:
        return 0.0
    unique_words = len(set(words))
    return unique_words / len(words)


def load_poems(path):
    """从指定目录加载所有诗歌文本"""
    poems = []
    if not os.path.exists(path):
        print(f"警告: 目录不存在 - {path}")
        return poems

    for fname in os.listdir(path):
        if not fname.endswith('.txt'):
            continue

        fpath = os.path.join(path, fname)
        if os.path.isfile(fpath):
            try:
                with open(fpath, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    if content:  # 跳过空文件
                        poems.append(content)
            except Exception as e:
                print(f"读取文件 {fname} 时出错: {e}")

    return poems


def evaluate(path):
    """
    评估指定目录下诗歌的质量
    返回包含以下指标的字典:
    - count: 诗歌数量
    - avg_repeat_rate: 平均字符重复率
    - avg_length: 平均长度
    - unique_ratio: 独特诗歌比例
    - avg_unique_word_ratio: 平均独特词比例
    - common_words: 最常见的10个词
    """
    poems = load_poems(path)
    if not poems:
        return {
            'count': 0,
            'avg_repeat_rate': 0,
            'avg_length': 0,
            'unique_ratio': 0,
            'avg_unique_word_ratio': 0,
            'common_words': []
        }

    # 计算基本指标
    repeat_rates = [calc_repeat_rate(p) for p in poems]
    lengths = [len(p) for p in poems]
    unique_ratios = [calc_unique_word_ratio(p) for p in poems]

    # 计算常见词
    all_words = []
    for poem in poems:
        all_words.extend(poem.split())
    word_counts = Counter(all_words)
    common_words = word_counts.most_common(10)

    return {
        'count': len(poems),
        'avg_repeat_rate': sum(repeat_rates) / len(repeat_rates),
        'avg_length': sum(lengths) / len(lengths),
        'unique_ratio': len(set(poems)) / len(poems),
        'avg_unique_word_ratio': sum(unique_ratios) / len(unique_ratios),
        'common_words': common_words
    }


def save_evaluation_results(results, output_path):
    """将评估结果保存到JSON文件"""
    try:
        # 将常见词列表转换为可序列化的格式
        serializable_results = results.copy()
        serializable_results['common_words'] = [
            [word, count] for word, count in results['common_words']
        ]

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"评估结果已保存至: {output_path}")
    except Exception as e:
        print(f"保存评估结果时出错: {e}")


if __name__ == '__main__':
    # 获取当前脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # 构建正确的数据路径（上一级目录中的data/generated_poems）
    base_path = os.path.join(script_dir, '..', 'data', 'generated_poems')
    base_path = os.path.normpath(base_path)  # 规范化路径

    print(f"正在评估目录: {base_path}")

    if not os.path.exists(base_path):
        print(f"错误: 目录不存在 - {base_path}")
        exit(1)

    # 创建输出目录
    output_dir = os.path.join(script_dir, '..', 'experiments')
    os.makedirs(output_dir, exist_ok=True)

    # 评估每个策略目录
    for folder in sorted(os.listdir(base_path)):
        full_path = os.path.join(base_path, folder)
        if os.path.isdir(full_path):
            print(f"\n评估策略: {folder}")
            stats = evaluate(full_path)

            # 打印结果
            print(f"诗歌数量: {stats['count']}")
            print(f"平均字符重复率: {stats['avg_repeat_rate']:.4f}")
            print(f"平均长度: {stats['avg_length']:.2f}")
            print(f"独特诗歌比例: {stats['unique_ratio']:.4f}")
            print(f"平均独特词比例: {stats['avg_unique_word_ratio']:.4f}")
            print("最常见10个词:")
            for word, count in stats['common_words']:
                print(f"  '{word}': {count}")

            # 保存结果
            output_path = os.path.join(output_dir, f'eval_{folder}.json')
            save_evaluation_results(stats, output_path)

    print("\n所有策略评估完成！")