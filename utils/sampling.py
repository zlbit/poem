import numpy as np

def sample_word(predict, idx_to_word, strategy='greedy', temperature=1.0, top_k=0, top_p=0.0):
    """
    根据指定策略从预测分布中采样下一个词。
    """
    predict = predict.detach().cpu().numpy().flatten()
    predict = predict / np.sum(predict)

    if strategy == 'greedy':
        return idx_to_word[np.argmax(predict)]

    if strategy == 'temperature':
        predict = np.log(predict + 1e-8) / temperature
        predict = np.exp(predict) / np.sum(np.exp(predict))
        return idx_to_word[np.random.choice(len(predict), p=predict)]

    if strategy == 'top_k':
        top_k = min(top_k, len(predict))
        indices = np.argpartition(-predict, top_k)[:top_k]
        top_probs = predict[indices]
        top_probs = top_probs / np.sum(top_probs)
        return idx_to_word[np.random.choice(indices, p=top_probs)]

    if strategy == 'top_p':
        sorted_indices = np.argsort(predict)[::-1]
        cumulative_probs = np.cumsum(predict[sorted_indices])
        cutoff = np.searchsorted(cumulative_probs, top_p)
        selected = sorted_indices[:cutoff+1]
        selected_probs = predict[selected] / np.sum(predict[selected])
        return idx_to_word[np.random.choice(selected, p=selected_probs)]

    raise ValueError(f"Unknown sampling strategy: {strategy}")
