# data_loader.py
import numpy as np
import pandas as pd
from numpy import load
from sklearn.preprocessing import MinMaxScaler
import config

def create_dataset(location, traffic, WINDOW_SIZE):
    """
    为单个位置创建带滞后特征的样本
    """
    location_current = traffic[traffic["location"] == location].reset_index(drop=True)
    location_current["hour"] = ((location_current["timestep"] - 1) // 12)
    grouped = location_current.groupby("hour").mean().reset_index()
    grouped['hour'] %= 24
    one_hot_hour = pd.get_dummies(grouped['hour'], prefix='hour')
    hour_grouped = pd.concat([grouped[["occupy", "flow", "speed"]], one_hot_hour], axis=1)
    hour_grouped = np.array(hour_grouped, dtype=np.float32)

    X, Y = [], []
    for i in range(len(hour_grouped) - WINDOW_SIZE):
        X.append(hour_grouped[i:i + WINDOW_SIZE][::-1])
        Y.append(hour_grouped[i + WINDOW_SIZE, 0])
    return X, Y

def load_data():
    """加载并预处理数据，返回训练/验证/测试集及归一化器"""
    data = load(config.ORIGN_DATA_PATH)
    lst = data.files
    traffic_data = data[lst[0]]
    print(f"原始数据形状: {traffic_data.shape}")

    # 将三维数组转换为 DataFrame（每个时间步、每个位置一行）
    data_dict = []
    for timestep in range(traffic_data.shape[0]):
        for location in range(traffic_data.shape[1]):
            data_dict.append({
                "timestep": timestep + 1,
                "location": location,
                "flow": traffic_data[timestep][location][0],
                "occupy": traffic_data[timestep][location][1],
                "speed": traffic_data[timestep][location][2]
            })

    df = pd.DataFrame(data_dict)
    df.to_csv(config.DATA_PATH, index=False)
    traffic = pd.read_csv(config.DATA_PATH)

    # 获取传感器数量
    num_nodes = traffic['location'].nunique()
    print(f"传感器数量: {num_nodes}")

    # 为所有传感器生成数据
    X_all, Y_all = [], []
    for loc in range(num_nodes):
        x_loc, y_loc = create_dataset(loc, traffic, config.WINDOW_SIZE)
        X_all.append(x_loc)
        Y_all.append(y_loc)

    X = np.moveaxis(np.array(X_all, dtype=object), 0, -1)
    Y = np.moveaxis(np.array(Y_all, dtype=object), 0, -1)
    X = np.array([x.tolist() for x in X], dtype=np.float32)  # (样本数, 24, 27, num_nodes)
    Y = np.array([y.tolist() for y in Y], dtype=np.float32)  # (样本数, num_nodes)

    # 合并特征与位置维度
    X = X.reshape(X.shape[0], X.shape[1], -1)  # (样本数, 24, 27 * num_nodes)

    # 按时间顺序划分数据集
    N = len(X)
    train_size = int(N * (1 - config.VAL_SPLIT - config.TEST_SPLIT))
    val_size = int(N * config.VAL_SPLIT)
    test_size = N - train_size - val_size

    train_X_raw = X[:train_size]
    val_X_raw = X[train_size:train_size+val_size]
    test_X_raw = X[train_size+val_size:]

    train_Y_raw = Y[:train_size]
    val_Y_raw = Y[train_size:train_size+val_size]
    test_Y_raw = Y[train_size+val_size:]

    print(f"训练集 X: {train_X_raw.shape}, Y: {train_Y_raw.shape}")
    print(f"验证集 X: {val_X_raw.shape}, Y: {val_Y_raw.shape}")
    print(f"测试集 X: {test_X_raw.shape}, Y: {test_Y_raw.shape}")

    # 归一化
    scaler_X = MinMaxScaler()
    scaler_Y = MinMaxScaler()

    def normalize_X(data, scaler, fit=False):
        shape = data.shape
        data_2d = data.reshape(-1, shape[-1])
        if fit:
            normalized = scaler.fit_transform(data_2d)
        else:
            normalized = scaler.transform(data_2d)
        return normalized.reshape(shape)

    train_X = normalize_X(train_X_raw, scaler_X, fit=True)
    val_X = normalize_X(val_X_raw, scaler_X, fit=False)
    test_X = normalize_X(test_X_raw, scaler_X, fit=False)

    train_Y = scaler_Y.fit_transform(train_Y_raw)
    val_Y = scaler_Y.transform(val_Y_raw)
    test_Y = scaler_Y.transform(test_Y_raw)

    return (train_X, train_Y, val_X, val_Y, test_X, test_Y,
            scaler_X, scaler_Y, num_nodes)