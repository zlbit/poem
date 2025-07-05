import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
def calc_repeat_rate(text):
    return 1 - len(set(text)) / len(text) if text else 0

def load_poems(path):
    poems = []
    for fname in os.listdir(path):
        fpath = os.path.join(path, fname)
        if os.path.isfile(fpath):
            with open(fpath, 'r', encoding='utf-8') as f:
                poems.append(f.read().strip())
    return poems

def evaluate(path):
    poems = load_poems(path)
    repeat_rates = [calc_repeat_rate(p) for p in poems if p]
    lengths = [len(p) for p in poems if p]
    return {
        'count': len(poems),
        'avg_repeat_rate': sum(repeat_rates) / len(repeat_rates) if repeat_rates else 0,
        'avg_length': sum(lengths) / len(lengths) if lengths else 0,
        'unique_ratio': len(set(poems)) / len(poems) if poems else 0
    }

if __name__ == '__main__':
    base_path = 'data/generated_poems'
    for folder in sorted(os.listdir(base_path)):
        full_path = os.path.join(base_path, folder)
        if os.path.isdir(full_path):
            stats = evaluate(full_path)
            print(f"{folder}: {stats}")