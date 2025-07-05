import re

import gradio as gr
import torch
from test import *


def generate_chinese_poem(input_char, style):
    """
    根据输入汉字和选择的风格生成诗歌
    """
    print(input_char)
    if len(input_char) != 1:
        return "⚠️ 请只输入一个汉字", "生成结果为空，无法复制"
    if not '\u4e00' <= input_char <= '\u9fff':
        return "⚠️ 请输入有效的汉字", "生成结果为空，无法复制"

    # 最大尝试次数
    max_attempts = 5
    for attempt in range(max_attempts):
        # 生成诗歌
        poem = gen_poem(input_char, style)
        print(f"尝试 {attempt + 1}: {poem}")

        # 分析生成的诗歌
        lines = [line.strip() for line in poem.split('\n') if line.strip()]

        # 检查是否有四句
        if len(lines) < 4:
            continue

        # 删除每行中可能存在的标点符号
        clean_lines = []
        for line in lines[:4]:
            # 移除所有标点符号
            clean_line = re.sub(r'[，。、？！；：""''（）【】《》]', '', line)
            clean_lines.append(clean_line)

        # 检查每句的长度是否符合要求
        line_length = 5 if style == "五言绝句" else 7

        # 检查清理后的行长度
        if all(len(line) == line_length for line in clean_lines):
            # 重新格式化为标准绝句格式
            formatted_poem = f"{clean_lines[0]}，{clean_lines[1]}。\n{clean_lines[2]}，{clean_lines[3]}。"
            return formatted_poem, formatted_poem

        # 即使长度不完全匹配，如果相差不大，也可以尝试调整
        if all(abs(len(line) - line_length) <= 1 for line in clean_lines):
            # 调整行长度
            adjusted_lines = []
            for line in clean_lines:
                if len(line) < line_length:
                    # 短了就补充常用字
                    fillers = ["山", "水", "云", "花", "风", "月"]
                    line += fillers[0]
                elif len(line) > line_length:
                    # 长了就截断
                    line = line[:line_length]
                adjusted_lines.append(line)

            # 使用调整后的行
            formatted_poem = f"{adjusted_lines[0]}，{adjusted_lines[1]}。\n{adjusted_lines[2]}，{adjusted_lines[3]}。"
            return formatted_poem, formatted_poem

    # 如果多次尝试后仍未生成符合要求的诗，直接使用最后一次生成的诗
    # 清理并格式化
    lines = [line.strip() for line in poem.split('\n') if line.strip()]
    clean_lines = []

    # 确保有4行
    while len(lines) < 4:
        lines.append("山水云月风")

    # 清理每行
    for line in lines[:4]:
        # 移除标点
        clean_line = re.sub(r'[，。、？！；：""''（）【】《》]', '', line)

        # 调整长度
        if len(clean_line) < line_length:
            fillers = ["山", "水", "云", "花", "风", "月"]
            while len(clean_line) < line_length:
                clean_line += fillers[len(clean_line) % len(fillers)]
        elif len(clean_line) > line_length:
            clean_line = clean_line[:line_length]

        clean_lines.append(clean_line)

    # 格式化最终诗句
    formatted_poem = f"{clean_lines[0]}，{clean_lines[1]}。\n{clean_lines[2]}，{clean_lines[3]}。"
    return formatted_poem, formatted_poem


description_md = """
# AI 古诗生成器 🎭
请输入一个汉字，AI将为您创作一首以该汉字开头的诗歌。
"""

content_md = """
---
## 📝 详细说明

### 功能介绍
本项目使用 PyTorch 和LSTM实现古诗生成模型，可以根据用户输入的单个汉字，生成对应的古典诗歌。

### 使用说明
1. 在输入框中输入**一个汉字**
2. 选择想要的**诗歌风格**
3. 点击"生成诗歌"按钮
4. 可以点击"复制诗歌"按钮复制生成的内容

### 创作技巧
- 可以尝试输入季节词语（如：春、夏、秋、冬）
- 可以使用表达情感的字词（如：愁、思、忆）
- 建议选择意境优美的汉字

### 支持的诗歌格式
- 五言绝句
- 七言绝句



"""

# 创建Gradio界面
with gr.Blocks(theme=gr.themes.Soft()) as iface:
    gr.Markdown(description_md)

    with gr.Row():
        input_char = gr.Textbox(
            lines=1,
            placeholder="请输入一个汉字...",
            label="输入汉字"
        )
        style = gr.Radio(
            choices=["五言绝句", "七言绝句"],
            label="诗歌风格",
            value="五言绝句"
        )

    # 创建一个隐藏的文本框用于复制功能
    copy_text = gr.Textbox(visible=False)

    # 显示诗歌的文本框
    output = gr.Textbox(
        lines=4,
        label="生成的诗歌"
    )

    # 按钮行
    with gr.Row():
        generate_btn = gr.Button("生成诗歌", variant="primary")
        copy_btn = gr.Button("复制诗歌")

    # 示例区域
    gr.Examples(
        examples=[
            ["春", "五言绝句"],
            ["月", "七言绝句"],

        ],
        inputs=[input_char, style],
        outputs=[output, copy_text]
    )

    # 添加详细说明
    gr.Markdown(content_md)

    # 设置按钮功能
    generate_btn.click(
        fn=generate_chinese_poem,
        inputs=[input_char, style],
        outputs=[output, copy_text]
    )

    # 添加复制功能
    copy_btn.click(
        None,
        copy_text,
        None,
        js="""
        (text) => {
            if (text === "" || text === "生成结果为空，无法复制") {
                alert("请先生成诗歌！");
                return;
            }
            navigator.clipboard.writeText(text);
            alert("诗歌已复制到剪贴板！");
        }
        """
    )

# 启动服务
if __name__ == "__main__":
    iface.launch(share=True)