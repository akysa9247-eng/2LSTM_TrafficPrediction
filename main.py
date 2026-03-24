# main.py
import torch
import matplotlib.pyplot as plt
import config
import numpy as np
from data_loader import load_data
from model import TrafficLSTM_model
from train import train_model
from evaluate import evaluate_model

def main():
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    # 加载数据
    (train_X, train_Y, val_X, val_Y, test_X, test_Y,
     scaler_X, scaler_Y, num_nodes) = load_data()

    # 创建模型
    input_size = train_X.shape[2]
    model = TrafficLSTM_model(input_size=input_size,
                              hidden_size=config.HIDDEN_SIZE,
                              output_size=num_nodes,
                              dropout=config.DROPOUT)
    print(model)

    # 训练
    model = train_model(model, train_X, train_Y, val_X, val_Y)

    # 评估并获取测试集预测值
    test_actual, test_pred = evaluate_model(model, train_X, train_Y, test_X, test_Y, scaler_Y)

    # 绘制预测曲线
    sensor_idx = 0
    plt.figure(figsize=(12, 5))
    plt.plot(test_actual[:, sensor_idx], label='True Values', color='blue', alpha=0.7)
    plt.plot(test_pred[:, sensor_idx], label='Predictions', color='red', alpha=0.7)
    plt.title(f'Predictions vs True Values for Sensor {sensor_idx}')
    plt.xlabel('Time Step (hour)')
    plt.ylabel('Occupancy')
    plt.legend()
    plt.grid(True)
    plt.show()

    # 保存模型权重
    torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
    print(f"\n模型已保存为 {config.MODEL_SAVE_PATH}")

if __name__ == "__main__":
    main()