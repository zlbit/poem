# poem
2025暑假实习

项目简介
该项目使用基于PyTorch的循环神经网络(RNN)模型来生成中国古典诗歌。模型通过学习大量古诗的模式，能够根据用户提供的起始字生成五言绝句或七言绝句格式的诗歌。
依赖列表
在运行此项目前，请确保安装以下依赖库：
torch>=1.8.0
paddlenlp>=2.0.0
matplotlib>=3.3.0
numpy>=1.19.0
gradio>=3.0.0
可以通过以下命令安装依赖：
pip install torch paddlenlp matplotlib numpy gradio
项目结构
• train.py: 模型训练脚本
• test.py: 诗歌生成测试脚本
• main.py: Gradio Web界面启动脚本
• poem_data_processing.py: 诗歌数据处理工具
• model.py: RNN模型定义
运行说明

1. 训练模型
运行以下命令开始训练模型：
python train.py
训练过程将会：
• 加载诗歌数据集
• 构建词汇表
• 训练多个实验配置的模型
• 保存模型权重和训练指标
• 生成训练过程可视化图表
训练结果将保存在./experiments/和./model/目录中。
2. 生成诗歌（命令行）
训练完成后，可以使用以下命令生成诗歌：
python test.py
运行后，按照提示输入一个起始汉字，程序将生成一首以该字开头的诗歌。
3. 启动Web界面
   使用以下命令启动Gradio Web界面：
   python main.py
   启动后，可以通过浏览器访问本地服务（通常为http://127.0.0.1:7860），在界面中输入起始汉字并选择诗歌风格，生成诗歌。
   参数设置
   模型参数
   在model.py中，RNN模型的主要参数包括：
   • vocab_size: 词汇表大小
   • rnn_size: RNN隐藏层大小（256）
   • num_layers: RNN层数（2）
   • dropout: Dropout概率（0.0）
   训练参数
   train.py中包含多组实验配置，每组配置有不同的参数组合：
   {
    'name': '实验名称'
   ,
    'lr': 学习率,               # 取值范围：0.0005-0.002
    'rnn_size': RNN隐藏层大小,  # 取值：128, 256, 512
    'batch_size': 批量大小,     # 取值：64, 128
    'dropout': Dropout概率,     # 取值：0.0, 0.3, 0.5
    'num_layers': RNN层数,      # 取值：1, 2, 3
    'num_epochs': 训练轮次,     # 默认50
   }
   生成参数
   在test.py和main.py中，诗歌生成的参数包括：
   • begin_word: 诗歌的起始字
   • style: 诗歌风格，可选"五言绝句"或"七言绝句"
