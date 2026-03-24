# evaluate.py
import numpy as np
import torch
import config
from utils import columnwise_spearman

def predict_and_calc_score(model, X, Y, scaler_Y):
    """
    返回反归一化后的真实值、预测值、移动平均基线，
    以及模型的 RMSE、MAE、MAPE、基线 RMSE
    """
    model.eval()
    with torch.no_grad():
        X_t = torch.tensor(X, dtype=torch.float32)
        pred = model(X_t).cpu().numpy()

    # 反归一化
    pred_scaled = scaler_Y.inverse_transform(pred)
    Y_scaled = scaler_Y.inverse_transform(Y)

    # 移动平均基线（窗口12小时）
    window_size = 12
    moving_avg = np.apply_along_axis(
        lambda x: np.convolve(x, np.ones(window_size)/window_size, mode='same'), axis=0, arr=Y_scaled)

    baseline_rmse = np.sqrt(np.mean((Y_scaled - moving_avg) ** 2))
    model_rmse = np.sqrt(np.mean((Y_scaled - pred_scaled) ** 2))
    model_mae = np.mean(np.abs(Y_scaled - pred_scaled))
    mask = np.abs(Y_scaled) > 1e-4
    model_mape = np.mean(np.abs((Y_scaled[mask] - pred_scaled[mask]) / Y_scaled[mask])) * 100

    return Y_scaled, pred_scaled, moving_avg, model_rmse, baseline_rmse, model_mae, model_mape

def evaluate_model(model, train_X, train_Y, test_X, test_Y, scaler_Y):
    """评估训练集和测试集，打印结果"""
    # 训练集评估
    train_actual, train_pred, train_ma, train_rmse, train_base_rmse, train_mae, train_mape = \
        predict_and_calc_score(model, train_X, train_Y, scaler_Y)
    print("\n====== 训练集结果 ======")
    print(f"移动平均基线 RMSE: {train_base_rmse:.4f}")
    print(f"模型预测 RMSE: {train_rmse:.4f}, MAE: {train_mae:.4f}, MAPE: {train_mape:.2f}%")

    train_spearman = columnwise_spearman(train_actual, train_pred)
    train_ma_spearman = columnwise_spearman(train_actual, train_ma)
    print(f"训练集 Spearman (模型预测): {train_spearman:.4f}")
    print(f"训练集 Spearman (移动平均基线): {train_ma_spearman:.4f}")

    # 测试集评估
    test_actual, test_pred, test_ma, test_rmse, test_base_rmse, test_mae, test_mape = \
        predict_and_calc_score(model, test_X, test_Y, scaler_Y)
    print("\n====== 测试集结果 ======")
    print(f"移动平均基线 RMSE: {test_base_rmse:.4f}")
    print(f"模型预测 RMSE: {test_rmse:.4f}, MAE: {test_mae:.4f}, MAPE: {test_mape:.2f}%")

    test_spearman = columnwise_spearman(test_actual, test_pred)
    test_ma_spearman = columnwise_spearman(test_actual, test_ma)
    print(f"测试集 Spearman (模型预测): {test_spearman:.4f}")
    print(f"测试集 Spearman (移动平均基线): {test_ma_spearman:.4f}")

    return test_actual, test_pred