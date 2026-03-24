基于双层 LSTM 的交通流时空预测模型

本项目实现了一个基于双层 LSTM 和层归一化的交通流时空预测模型，在 PeMS08 公开数据集上验证了其有效性。代码采用模块化设计，便于扩展和复现。

------

## 📖 项目简介

针对城市交通流预测中多站点、非线性、强时序依赖的挑战，本项目基于 PeMS08 数据集（170 个检测器，3 个交通指标），设计并实现了一个双层 LSTM 网络，用于预测未来一小时的占用率。通过对比实验，验证了双层结构相较于单层 LSTM 和移动平均基线的优越性。

**技术栈**：Python, PyTorch, LayerNorm, Scikit-learn, Pandas, NumPy, Matplotlib

------

## ✨ 主要特点

- **双层 LSTM**：通过堆叠两层 LSTM 提取高阶时序特征，提升预测精度。
- **层归一化（LayerNorm）**：稳定训练过程，不破坏时间依赖性，优于 BatchNorm。
- **正则化技术**：Dropout（0.35）与梯度裁剪防止过拟合。
- **动态学习率**：ReduceLROnPlateau 自适应调整学习率。
- **多指标评估**：RMSE、MAE、MAPE、Spearman 相关系数全面评价。
- **模块化设计**：配置、数据加载、模型、训练、评估分离，易于维护与扩展。

------

## 📊 数据集

采用加州交通部公开数据集 **PeMS08**，包含 170 个传感器，时间跨度为 2016 年，采样频率为 5 分钟。原始数据包含流量（flow）、占有率（occupy）、速度（speed）三个指标。

**预处理步骤**：

1. 小时聚合（每 12 个 5 分钟点平均为 1 小时）。
2. 添加小时 one‑hot 编码（24 维），增强周期性特征。
3. 滑动窗口（24 小时）构造样本，窗口内倒序排列。
4. 将所有传感器特征拼接为 4590 维向量（27 维 × 170 传感器）。
5. MinMaxScaler 归一化，6:2:2 时间顺序划分训练/验证/测试集。

------

## 🔧 环境配置

```
# 克隆仓库
git clone https://github.com/yourname/traffic-prediction.git
cd traffic-prediction

# 安装依赖
pip install -r requirements.txt
```

`requirements.txt` 内容：

```
numpy
pandas
torch
scikit-learn
scipy
matplotlib
```

------

## 📁 项目结构

```
traffic-prediction/
├── config.py               # 超参数与路径配置
├── data_loader.py          # 数据加载与预处理
├── model.py                # 双层 LSTM 模型定义
├── train.py                # 训练循环
├── evaluate.py             # 模型评估与指标计算
├── utils.py                # 工具函数（Spearman 等）
├── main.py                 # 主入口
├── data/
│   └── processed/
│       └── traffic_08.csv  # 预处理后的数据（自动生成）
├── models/                 # 保存训练好的模型权重
└── README.md
```

------

## 🚀 使用方法

### 1. 准备数据

将原始 PeMS08 数据文件 `pems08.npz` 放入 `data/raw/` 目录，首次运行会自动生成 `data/processed/traffic_08.csv`。
（注：代码中已注释生成步骤，若已有 CSV 可直接使用）

### 2. 训练与评估

```
python main.py
```

脚本将自动完成：

- 数据加载与预处理
- 模型构建与训练（默认 100 个 epoch）
- 在训练集和测试集上计算 RMSE、MAE、MAPE、Spearman 相关系数
- 绘制第一个传感器的预测对比曲线
- 保存模型权重到 `models/` 目录

### 3. 修改超参数

编辑 `config.py` 即可调整训练参数（学习率、隐藏层大小、Dropout 率、训练轮数等）。

------

## 📈 实验结果

在 PeMS08 测试集上，双层 LSTM 模型取得以下性能：

| 指标     | 模型   | 移动平均基线 | 相对提升  |
| -------- | ------ | ------------ | --------- |
| RMSE     | 0.0209 | 0.0253       | **17.4%** |
| MAE      | 0.0085 | -            | -         |
| MAPE     | 14.82% | -            | -         |
| Spearman | 0.9452 | 0.8414       | **12.3%** |

与单层 LSTM 对比，双层结构在 RMSE、MAE、MAPE 上均有稳定提升，验证了其提取高阶时序特征的有效性。

## 🔮 未来展望

- **双向 LSTM**：利用窗口内前后文信息，增强特征提取。
- **图神经网络（GCN）**：显式建模路网空间结构，实现时空同步预测。
- **注意力机制**：聚焦关键时间步，提升对突发事件的响应能力。
- **多步预测**：扩展为未来多小时的滚动预测，更具实用价值。