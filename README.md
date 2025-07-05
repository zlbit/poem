# 中文古诗生成项目说明文档 🎉📝

------

## 项目简介 📜

本项目基于 **RNN 模型**（LSTM），结合多种采样策略（贪心 Greedy、温度采样 Temperature、Top-K、Top-P ）进行生成对比，实现中文古诗自动生成。
 支持批量生成诗歌，自动保存结果，并提供生成结果的统计评价和可视化分析，便于对比不同采样策略的表现。✨

------

## 目录结构示例 📂

```
RNN-poem/
├── gradio/ # Gradio相关资源
├── data/ # 数据目录
│ └── generated_poems/ # 批量生成的诗歌
├── experiments/ # 实验记录
├── model/ # 模型存储
│ └── torch-latest.pth # 最佳模型权重
├── scripts/ # 评估和生成脚本
│ ├── evaluate_poems.py # 诗歌质量评估
│ ├── gen_poems_batch.py # 批量生成诗歌
│ └── plot_results.py # 结果可视化
│ └── streamlit_rating.py # Streamlit测试页面
├── ui/ # 用户界面资源文件
├── utils/ # 工具函数
│ └── sampling.py # 采样策略实现
├── main.py # 主应用入口（Gradio）
├── model.py # 模型架构定义
├── poem_data_processing.py# 数据预处理脚本
├── test.py # 单首诗歌生成脚本
├── train.py # 模型训练脚本
├── .gitignore # Git忽略配置
├── LICENSE # 许可证
└── README.md # 项目文档
```

------

## 依赖环境 ⚙️

请确保 Python 版本 ≥ 3.7，建议使用虚拟环境。

**主要依赖包：**

| 库名称     | 说明            | 推荐版本 |
| ---------- | --------------- | -------- |
| torch      | PyTorch框架     | ≥1.8.0   |
| paddlenlp  | PaddleNLP工具库 | ≥2.0.0   |
| numpy      | 数值计算        | ≥1.19.0  |
| pandas     | 数据处理        | ≥1.1.0   |
| matplotlib | 绘图库          | ≥3.3.0   |
| seaborn    | 统计可视化      | ≥0.11.0  |



安装示例：

```python
pip install torch paddlenlp numpy pandas matplotlib seaborn
```

## 运行说明 🚀

###  1. 批量生成诗歌 `gen_poems_batch.py` 📚

- 🧠 自动加载训练好的模型

- 🎲 支持多种采样策略（贪婪/温度/top-k/top-p）

- 🌸 使用多个起始字（春/风/月/夜/山）批量生成

- 💾 结果保存在 data/generated_poems/

运行命令：

```
python scripts/gen_poems_batch.py
```

### 2. 生成结果评估 `evaluate_poems.py` 📊

- 🔢 统计诗歌数量、平均重复率、平均长度
- 🧐 计算唯一性比例（避免重复生成）
- 📈 支持不同策略生成结果的定量对比

运行命令：

```
python scripts/evaluate_poems.py
```

### 3. 结果可视化 `plot_results.py` 📈

- 🌈 汇总所有策略生成结果
- 🔥 用热力图直观展示指标差异
- 🏆 便于选择最佳采样策略

运行命令：

```
python scripts/plot_results.py
```

### 4. 交互式诗歌生成 `main.py` ✨

- 🖥️ 启动Gradio网页界面
- ✍️ 用户输入单个汉字即可生成诗歌
- 🌟 支持五言/七言绝句选择
- 📋 一键复制功能

运行命令：

```
python main.py
```

### 5. 模型训练 `train.py` 🏋️

- ⚙️ 运行多个预设实验配置
- 💾 自动保存最佳模型到 `model/torch-latest.pth`
- 📝 记录训练指标到 `experiments/`
- 📉 生成训练过程可视化图表

运行命令：

```
python train.py
```

### 6. Streamlit测试 `streamlit_rating.py` 🧪

- 🔍 验证数据目录结构
- ✅ 检查文件路径是否正确
- 🐞 调试辅助工具

运行命令：

```
streamlit run scripts/streamlit_rating.py
```

------



## 参数说明 ⚙️

| 参数名      | 说明                  | 默认值   | 示例                                      |
| ----------- | --------------------- | -------- | ----------------------------------------- |
| start_token | 句首特殊符号          | 'B'      | 'B'                                       |
| end_token   | 句尾特殊符号          | 'E'      | 'E'                                       |
| begin_char  | 诗句起始字            | '春'     | '春', '风', '月'                          |
| strategy    | 采样策略              | 'greedy' | 'greedy', 'temperature', 'top_k', 'top_p' |
| temperature | 温度采样温度          | 1.0      | 0.7, 1.3                                  |
| top_k       | Top-K采样保留数量     | 0        | 5, 10, 30                                 |
| top_p       | Top-P采样累计概率阈值 | 0.0      | 0.8, 0.9, 0.95                            |
| max_len     | 最大生成长度          | 50       | 50                                        |



------

## 采样策略简介 🎲

- **greedy**：选择概率最大字，最确定性
- **temperature**：调节概率分布温度，控制随机程度
- **top_k**：从概率最高的 K 个字中采样
- **top_p**：Nucleus采样，从累计概率达到阈值的字中采样

------

## 调参实验 🧪🔥

### 实验配置总表

| 实验ID | 名称                                | 学习率 | RNN大小 | Batch大小 | Dropout | 层数 | 50轮 | 100轮 |
| ------ | ----------------------------------- | ------ | ------- | --------- | ------- | ---- | ---- | ----- |
| A1     | `A1_lr0.002_rnn128_bs64_dp0.0_l2`   | 0.002  | 128     | 64        | 0.0     | 2    | ✅    | ✅     |
| A2     | `A2_lr0.001_rnn128_bs64_dp0.0_l2`   | 0.001  | 128     | 64        | 0.0     | 2    | ✅    | ✅     |
| B1     | `B1_lr0.001_rnn256_bs64_dp0.3_l2`   | 0.001  | 256     | 64        | 0.3     | 2    | ✅    | ✅     |
| B2     | `B2_lr0.001_rnn256_bs128_dp0.3_l2`  | 0.001  | 256     | 128       | 0.3     | 2    | ✅    | ✅     |
| C1     | `C1_lr0.0005_rnn256_bs128_dp0.3_l3` | 0.0005 | 256     | 128       | 0.3     | 3    | ✅    | ✅     |
| C2     | `C2_lr0.0005_rnn512_bs64_dp0.5_l3`  | 0.0005 | 512     | 64        | 0.5     | 3    | ✅    | ✅     |
| D1     | `D1_lr0.001_rnn256_bs64_dp0.5_l1`   | 0.001  | 256     | 64        | 0.5     | 1    | ✅    | ✅     |

> ✅ = 已执行实验 ❌ = 未执行实验

------

## 注意事项 ⚠️

- 请确保模型权重 `torch-latest.pth` 放在 `./model/` 目录下
- 生成诗歌文件均为 UTF-8 编码
- 可调参数灵活生成不同风格诗歌
- 本项目聚焦训练、推理与采样策略对比

------

## 联系与反馈 💬

欢迎反馈建议，共同完善项目！
 祝你生成优美古诗，诗意满满~ 🌸🖋️
