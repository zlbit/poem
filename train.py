import os
import time
import math
import pickle
import matplotlib.pyplot as plt
import torch
from torch import nn
from paddlenlp.datasets import load_dataset

from model import RNNModel
from poem_data_processing import process_poems_from_dataset, generate_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

start_token = 'B'
end_token = 'E'


def process_poems_filtered(dataset):
    """
    过滤数据，只保留完整四句结构的诗。
    """
    poems = []

    for sample in dataset:
        content_raw = sample.get('tokens', '') or sample.get('labels', '')
        if not content_raw:
            continue

        # 按标点切句，保留逗号和句号
        content = ''.join([w for w in content_raw.split('\x02') if w.strip()])
        # 简单处理，保证有4句（以逗号和句号为断句）
        sentences = [s for s in content.replace('。', '。|').replace('，', '，|').split('|') if s.strip()]
        if len(sentences) != 4:
            continue

        # 拼接加起止标记
        poem = start_token + ''.join(sentences) + end_token
        poems.append(poem)

    print(f"Total poems collected after filtering: {len(poems)}")

    if len(poems) == 0:
        raise ValueError("No poems collected after filtering. Please check your dataset or filtering rules.")

    return poems


def process_poems_and_build_vocab(dataset):
    """
    重写，处理诗歌并构造词表
    """
    poems = process_poems_filtered(dataset)

    max_len = max(len(p) for p in poems)
    print(f"Max poem length: {max_len}")

    import collections
    all_words = [word for poem in poems for word in poem]
    counter = collections.Counter(all_words)
    words = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)
    words.append(' ')  # 补充空格词

    word_to_idx = {word: i for i, word in enumerate(words)}
    idx_to_word = words[:]

    poems_vector = [[word_to_idx[word] for word in poem] for poem in poems]

    return poems_vector, word_to_idx, idx_to_word


def evaluate(model, dataset, word_to_idx, batch_size=64):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0
    total_tokens = 0

    poems_vector, _, _ = process_poems_and_build_vocab(dataset)

    with torch.no_grad():
        for X, Y in generate_batch(batch_size, poems_vector, word_to_idx):
            X = X.to(device)
            Y = Y.to(device)

            outputs, _ = model(X, None)
            Y = Y.view(-1)

            loss = loss_fn(outputs, Y.long())

            total_loss += loss.item() * Y.shape[0]
            total_tokens += Y.shape[0]

    avg_loss = total_loss / total_tokens if total_tokens > 0 else float('inf')
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, ppl


def train_one_epoch(model, optimizer, loss_fn, poems_vector, word_to_idx, batch_size):
    model.train()
    loss_sum = 0
    total_tokens = 0

    for X, Y in generate_batch(batch_size, poems_vector, word_to_idx):
        X = X.to(device)
        Y = Y.to(device)

        outputs, _ = model(X, None)
        Y = Y.view(-1)

        loss = loss_fn(outputs, Y.long())

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        loss_sum += loss.item() * Y.shape[0]
        total_tokens += Y.shape[0]

    avg_loss = loss_sum / total_tokens if total_tokens > 0 else 0.0
    ppl = math.exp(avg_loss) if avg_loss < 100 else float('inf')
    return avg_loss, ppl


def plot_metrics(metrics, save_dir, exp_name):
    epochs = range(1, len(metrics['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, metrics['train_loss'], label='Train Loss')
    plt.plot(epochs, metrics['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'{exp_name} Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, metrics['train_ppl'], label='Train PPL')
    plt.plot(epochs, metrics['val_ppl'], label='Val PPL')
    plt.xlabel('Epoch')
    plt.ylabel('Perplexity')
    plt.title(f'{exp_name} Perplexity')
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{exp_name}_metrics.png'))
    plt.close()


def run_experiment(exp_config):
    exp_name = exp_config['name']
    output_dir = os.path.join('experiments', exp_name)
    os.makedirs(output_dir, exist_ok=True)

    # 确保model目录存在
    model_dir = './model/'
    os.makedirs(model_dir, exist_ok=True)

    print(f"\n=== Running experiment: {exp_name} ===")
    print("Parameters:", exp_config)

    test_dataset, dev_dataset, train_dataset = load_dataset(
        'poetry', splits=('test', 'dev', 'train'), lazy=False)
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Dev dataset size: {len(dev_dataset)}")
    print(f"Test dataset size: {len(test_dataset)}")

    poems_vector, word_to_idx, idx_to_word = process_poems_and_build_vocab(train_dataset)

    model = RNNModel(
        vocab_size=len(idx_to_word),
        rnn_size=exp_config['rnn_size'],
        num_layers=exp_config['num_layers'],
        dropout=exp_config['dropout']
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=exp_config['lr'])
    loss_fn = nn.CrossEntropyLoss()

    metrics = {
        'train_loss': [],
        'val_loss': [],
        'train_ppl': [],
        'val_ppl': [],
    }

    # 跟踪最佳模型性能
    best_val_loss = float('inf')

    for epoch in range(exp_config['num_epochs']):
        start_time = time.time()
        train_loss, train_ppl = train_one_epoch(
            model, optimizer, loss_fn, poems_vector, word_to_idx, exp_config['batch_size'])
        val_loss, val_ppl = evaluate(model, dev_dataset, word_to_idx, exp_config['batch_size'])
        end_time = time.time()

        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['train_ppl'].append(train_ppl)
        metrics['val_ppl'].append(val_ppl)

        print(f"Epoch {epoch + 1}/{exp_config['num_epochs']} "
              f"| Train Loss: {train_loss:.6f}, Train PPL: {train_ppl:.2f} "
              f"| Val Loss: {val_loss:.6f}, Val PPL: {val_ppl:.2f} "
              f"| Time: {end_time - start_time:.2f}s")

        # 如果当前模型是最佳模型，保存到model目录
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"Found new best model with val_loss: {val_loss:.6f}")

            # 创建保存的检查点字典
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'word_to_idx': word_to_idx,
                'idx_to_word': idx_to_word,
            }

            # 保存到model目录以供test.py使用
            torch.save(checkpoint, os.path.join(model_dir, 'torch-latest.pth'))
            print(f"Best model saved to {os.path.join(model_dir, 'torch-latest.pth')}")

    # 保留原有的保存逻辑 - 训练结束后保存最终模型
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'word_to_idx': word_to_idx,
        'idx_to_word': idx_to_word
    }, os.path.join(output_dir, 'model.pth'))

    with open(os.path.join(output_dir, 'vocab.pkl'), 'wb') as f:
        pickle.dump({'word_to_idx': word_to_idx, 'idx_to_word': idx_to_word}, f)

    with open(os.path.join(output_dir, 'metrics.pkl'), 'wb') as f:
        pickle.dump(metrics, f)

    plot_metrics(metrics, output_dir, exp_name)


if __name__ == "__main__":
    experiments = [
        {
            'name': 'A1_lr0.002_rnn128_bs64_dp0.0_l2',
            'lr': 0.002,
            'rnn_size': 128,
            'batch_size': 64,
            'dropout': 0.0,
            'num_layers': 2,
            'num_epochs': 50,
        },
        {
            'name': 'A2_lr0.001_rnn128_bs64_dp0.0_l2',
            'lr': 0.001,
            'rnn_size': 128,
            'batch_size': 64,
            'dropout': 0.0,
            'num_layers': 2,
            'num_epochs': 50,
        },
        {
            'name': 'B1_lr0.001_rnn256_bs64_dp0.3_l2',
            'lr': 0.001,
            'rnn_size': 256,
            'batch_size': 64,
            'dropout': 0.3,
            'num_layers': 2,
            'num_epochs': 50,
        },
        {
            'name': 'B2_lr0.001_rnn256_bs128_dp0.3_l2',
            'lr': 0.001,
            'rnn_size': 256,
            'batch_size': 128,
            'dropout': 0.3,
            'num_layers': 2,
            'num_epochs': 50,
        },
        {
            'name': 'C1_lr0.0005_rnn256_bs128_dp0.3_l3',
            'lr': 0.0005,
            'rnn_size': 256,
            'batch_size': 128,
            'dropout': 0.3,
            'num_layers': 3,
            'num_epochs': 50,
        },
        {
            'name': 'C2_lr0.0005_rnn512_bs64_dp0.5_l3',
            'lr': 0.0005,
            'rnn_size': 512,
            'batch_size': 64,
            'dropout': 0.5,
            'num_layers': 3,
            'num_epochs': 50,
        },
        {
            'name': 'D1_lr0.001_rnn256_bs64_dp0.5_l1',
            'lr': 0.001,
            'rnn_size': 256,
            'batch_size': 64,
            'dropout': 0.5,
            'num_layers': 1,
            'num_epochs': 50,
        },
    ]

    for exp in experiments:
        run_experiment(exp)
