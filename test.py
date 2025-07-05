import torch
import re

from model import RNNModel

# 模型保存的目录
model_dir = './model/'
# 定义开始和结束标记
start_token = 'B'
end_token = 'E'


def is_chinese_char(char):
    """检查是否为中文字符"""
    if not char or char.isspace():
        return False
    return bool(re.match(r'[\u4e00-\u9fff]', char))


def gen_poem(begin_word=None, style='五言绝句'):
    """
    基于训练过的模型生成诗歌，起始字为 begin_word。
    style: 五言绝句 或 七言绝句
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 首先加载checkpoint
    checkpoint = torch.load(f'{model_dir}/torch-latest.pth', map_location=device)

    # 直接从checkpoint中获取词表
    word_to_idx = checkpoint['word_to_idx']
    idx_to_word = checkpoint['idx_to_word']

    # 检查起始字是否在词表中
    if begin_word and begin_word not in word_to_idx:
        print(f"输入的字 `{begin_word}` 不在词表中，请换一个字试试。")
        return "请换一个字试试。"

    # 根据正确的参数初始化模型 (1层而不是2层)
    model = RNNModel(vocab_size=len(idx_to_word), rnn_size=256, num_layers=1)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()

    # 确定每句诗的字数
    chars_per_line = 5 if style == '五言绝句' else 7

    # 使用固定的预选词集合作为备选
    backup_words = ["山", "水", "云", "花", "风", "月", "日", "天", "春", "秋", "冬", "夏"]

    # 手动生成四句诗
    poem_lines = []

    with torch.no_grad():
        # 初始化状态
        hidden = None
        x = torch.tensor([[word_to_idx[start_token]]], dtype=torch.long).to(device)
        output, hidden = model(x, hidden)

        # 为四句诗分别生成内容
        for line_idx in range(4):
            # 每行的内容
            line_chars = []

            # 如果是第一行且有起始字
            if line_idx == 0 and begin_word:
                line_chars.append(begin_word)
                x = torch.tensor([[word_to_idx[begin_word]]], dtype=torch.long).to(device)
                output, hidden = model(x, hidden)

            # 为当前行生成所需字符数
            chars_needed = chars_per_line - len(line_chars)

            for i in range(chars_needed):
                # 获取模型输出并应用softmax
                probs = torch.softmax(output, dim=1)

                # 从概率分布中采样
                next_char_idx = torch.multinomial(probs, 1).item()
                next_char = idx_to_word[next_char_idx]

                # 检查生成的字符
                if next_char == end_token or not is_chinese_char(next_char):
                    # 如果不是有效字符，使用备选词
                    import random
                    next_char = random.choice(backup_words)
                    next_char_idx = word_to_idx.get(next_char, word_to_idx[backup_words[0]])

                # 添加字符到当前行
                line_chars.append(next_char)

                # 更新下一步的输入
                x = torch.tensor([[next_char_idx]], dtype=torch.long).to(device)
                output, hidden = model(x, hidden)

            # 构建完整的行并添加标点
            line = "".join(line_chars)
            if line_idx < 3:
                line += "，"
            else:
                line += "。"

            poem_lines.append(line)

    # 返回完整的诗
    return "\n".join(poem_lines)


if __name__ == '__main__':
    begin_char = input("请输入第一个字 please input the first character: \n").strip()
    if not begin_char:
        print("请输入一个有效的汉字！")
    else:
        print("AI作诗 generating poem...\n")
        poem = gen_poem(begin_char)
        print(poem)