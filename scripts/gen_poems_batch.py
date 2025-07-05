import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from model import RNNModel
from paddlenlp.datasets import load_dataset
from poem_data_processing import process_poems_from_dataset
from utils.sampling import sample_word


def generate_poem(model, word_to_idx, idx_to_word, start_token='B', end_token='E', begin_char='春',
                  strategy='greedy', temperature=1.0, top_k=0, top_p=0.0, max_len=50):
    device = next(model.parameters()).device
    hidden = None
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
                break
            x = torch.tensor([[word_to_idx[word]]], dtype=torch.long).to(device)
            output, hidden = model(x, hidden)
            predict = torch.softmax(output, dim=1)
            word = sample_word(predict, idx_to_word, strategy, temperature, top_k, top_p)
    return result


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, _, train_dataset = load_dataset('poetry', splits=('test', 'dev', 'train'), lazy=False)
    poems_vector, word_to_idx, idx_to_word = process_poems_from_dataset(train_dataset)

    model = RNNModel(vocab_size=len(idx_to_word), rnn_size=128, num_layers=2)
    checkpoint = torch.load('./model/torch-latest.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

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
                output_dir = f'data/generated_poems/{strategy_name}_{param_str}'
                os.makedirs(output_dir, exist_ok=True)
                filename = f'{begin_char}_{i}.txt'
                with open(os.path.join(output_dir, filename), 'w', encoding='utf-8') as f:
                    f.write(poem)


if __name__ == '__main__':
    main()