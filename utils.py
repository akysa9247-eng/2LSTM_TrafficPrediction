# utils.py
import numpy as np
import scipy.stats as stats

def columnwise_spearman(actual, pred):
    """按列计算 Spearman 相关系数并取平均"""
    n = actual.shape[1]
    corr = [stats.spearmanr(actual[:, i], pred[:, i])[0] for i in range(n)]
    return np.mean(corr)