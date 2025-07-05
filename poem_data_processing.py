import collections
import numpy as np
import torch

start_token = 'B'
end_token = 'E'

def process_poems_from_dataset(dataset):
    poems = []

    for idx, sample in enumerate(dataset):

        # 先尝试用 'tokens' 字段，分割并拼接
        content_raw = sample.get('tokens', '')
        if not content_raw:
            content_raw = sample.get('labels', '')
        if not content_raw:
            continue

        # 按 '\x02' 分割，过滤空字符串，再合并成一句诗
        tokens_list = [token for token in content_raw.split('\x02') if token.strip()]
        content = ''.join(tokens_list).strip()

        if not content:
            continue

        # 过滤条件可先关闭调试
        # if len(content) < 5 or len(content) > 79:
        #     continue
        # if any(char in content for char in ['_', '(', '（', '《', '[', start_token, end_token]):
        #     continue

        content = start_token + content + end_token
        poems.append(content)

    print(f"Total poems collected after filtering: {len(poems)}")

    if len(poems) == 0:
        raise ValueError("No poems collected after filtering. Please check your dataset or filtering rules.")

    max_len = max(len(p) for p in poems)
    print(f"Max poem length: {max_len}")

    all_words = [word for poem in poems for word in poem]
    counter = collections.Counter(all_words)
    words = sorted(counter.keys(), key=lambda x: counter[x], reverse=True)
    words.append(' ')

    word_to_idx = {word: i for i, word in enumerate(words)}
    idx_to_word = words[:]

    poems_vector = [[word_to_idx[word] for word in poem] for poem in poems]

    return poems_vector, word_to_idx, idx_to_word


def generate_batch(batch_size, poems_vec, word_to_idx):
    """
    生成批量训练数据。

    :param batch_size: 批量大小
    :param poems_vec: 诗歌的数字序列列表
    :param word_to_idx: 词汇到索引的映射字典
    :return: yield (x_batches, y_batches) 的Tensor元组
    """
    import math
    num_examples = math.ceil(len(poems_vec) / batch_size)

    for i in range(num_examples):
        start_index = i * batch_size
        end_index = min(start_index + batch_size, len(poems_vec))
        batch_poems = poems_vec[start_index:end_index]
        current_batch_size = len(batch_poems)

        max_length = max(len(poem) for poem in batch_poems)

        x_data = np.full((current_batch_size, max_length), word_to_idx[' '], dtype=np.int64)
        for row, poem in enumerate(batch_poems):
            x_data[row, :len(poem)] = poem

        y_data = np.copy(x_data)
        y_data[:, :-1] = x_data[:, 1:]

        yield torch.tensor(x_data), torch.tensor(y_data)
