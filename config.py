# config.py
import os

# 数据路径
DATA_PATH = "data/processed/traffic_08.csv"
MODEL_SAVE_PATH = "models/Traffic2LSTM_PeMS08_model.pth"

# 训练参数
BATCH_SIZE = 32
HIDDEN_SIZE = 128
DROPOUT = 0.35
LEARNING_RATE = 0.0005
EPOCHS = 100
WINDOW_SIZE = 24          # 滑动窗口大小（小时）
VAL_SPLIT = 0.2           # 验证集比例
TEST_SPLIT = 0.2          # 测试集比例

# 随机种子
RANDOM_SEED = 42