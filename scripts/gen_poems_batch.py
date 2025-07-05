import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model import RNNModel
from paddlenlp.datasets import load_dataset
from poem_data_processing import process_poems_from_dataset
from utils.sampling import sample_word

# 显式设置使用CPU设备
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'


def generate_poem(model, word_to_idx, idx_to_word, start_token='B', end_token='E', begin_char='春',
                  strategy='greedy', temperature=1.0, top_k=0, top_p=0.0, max_len=50):
    device = next(model.parameters()).device
    hidden = None
    # 确保开始标记在词汇表中
    if start_token not in word_to_idx:
        start_token = 'B' if 'B' in word_to_idx else list(word_to_idx.keys())[0]

    x = torch.tensor([[word_to_idx[start_token]]], dtype=torch.long).to(device)
    result = ''
    with torch.no_grad():
        output, hidden = model(x, hidden)
        predict = torch.softmax(output, dim=1)
        word = begin_char
        i = 0
        while word != end_token and i < max_len:
            result += word
            i += 1
            if word not in word_to_idx:
                # 如果字符不在词汇表中，使用开始标记
                word = start_token
                continue
            x = torch.tensor([[word_to_idx[word]]], dtype=torch.long).to(device)
            output, hidden = model(x, hidden)
            predict = torch.softmax(output, dim=1)
            word = sample_word(predict, idx_to_word, strategy, temperature, top_k, top_p)
    return result


def main():
    # 强制使用CPU
    device = torch.device("cpu")

    # 先加载模型检查点
    model_path = '../model/torch-latest.pth'
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {os.path.abspath(model_path)}")
        exit(1)

    checkpoint = torch.load(model_path, map_location=device)

    # 加载词汇表
    if 'word_to_idx' in checkpoint and 'idx_to_word' in checkpoint:
        print("从检查点加载词汇表...")
        word_to_idx = checkpoint['word_to_idx']
        idx_to_word = checkpoint['idx_to_word']
        vocab_size = len(idx_to_word)
    else:
        print("从数据集重新创建词汇表...")
        _, _, train_dataset = load_dataset('poetry', splits=('test', 'dev', 'train'), lazy=False)
        _, word_to_idx, idx_to_word = process_poems_from_dataset(train_dataset)
        vocab_size = len(idx_to_word)

    print(f"词汇表大小: {vocab_size}")

    # 存储模型配置参数
    num_layers = 1  # 默认层数
    rnn_size = 256  # 默认RNN大小

    # 加载模型配置
    if 'config' in checkpoint:
        print("从检查点加载模型配置...")
        config = checkpoint['config']
        # 使用当前词汇表大小覆盖配置
        config['vocab_size'] = vocab_size
        # 获取层数和RNN大小
        num_layers = config.get('num_layers', num_layers)
        rnn_size = config.get('rnn_size', rnn_size)

        model = RNNModel(
            vocab_size=config['vocab_size'],
            rnn_size=rnn_size,
            num_layers=num_layers
        )
    else:
        print("使用当前词汇表大小创建模型...")
        # 使用实际词汇表大小创建模型
        model = RNNModel(
            vocab_size=vocab_size,
            rnn_size=rnn_size,
            num_layers=num_layers
        )

    # 加载模型权重
    try:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        print("模型权重加载成功（非严格模式）")
    except RuntimeError as e:
        print(f"警告: 部分权重加载失败: {e}")
        # 手动加载兼容的权重
        model_dict = model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                           if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        print(f"成功加载了 {len(pretrained_dict)}/{len(model_dict)} 个权重")

    model.to(device)
    model.eval()

    # 打印模型配置（不使用model.num_layers属性）
    print(f"模型配置 | 词汇表大小: {vocab_size} | RNN大小: {rnn_size} | 层数: {num_layers}")

    start_chars = ['春', '风', '月', '夜', '山']

    strategies = [
        ('greedy', {}),
        ('temperature', {'temperature': 0.7}),
        ('temperature', {'temperature': 1.0}),
        ('temperature', {'temperature': 1.3}),
        ('top_k', {'top_k': 5}),
        ('top_k', {'top_k': 10}),
        ('top_k', {'top_k': 30}),
        ('top_p', {'top_p': 0.8}),
        ('top_p', {'top_p': 0.9}),
        ('top_p', {'top_p': 0.95}),
    ]

    # 创建输出目录
    output_root = '../data/generated_poems'
    os.makedirs(output_root, exist_ok=True)

    # 生成诗歌
    for strategy_name, params in strategies:
        for begin_char in start_chars:
            for i in range(5):  # 生成5首
                poem = generate_poem(
                    model, word_to_idx, idx_to_word,
                    begin_char=begin_char,
                    strategy=strategy_name,
                    **params
                )
                param_str = "_".join([f"{k}{v}" for k, v in params.items()]) or "default"
                output_dir = os.path.join(output_root, f'{strategy_name}_{param_str}')
                os.makedirs(output_dir, exist_ok=True)
                filename = f'{begin_char}_{i}.txt'
                with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(poem)
                print(f"生成诗歌: {begin_char}_{i} | 策略: {strategy_name} | 参数: {params} | 长度: {len(poem)}")

    print("所有诗歌生成完成！")


if __name__ == '__main__':
    main()