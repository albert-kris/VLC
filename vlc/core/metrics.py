import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score


def cluster_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """聚类 ACC：匈牙利算法对齐预测簇与真值簇后算准确率。"""
    y_true = y_true.astype(np.int64)
    y_pred = y_pred.astype(np.int64)
    n_clusters = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((n_clusters, n_clusters), dtype=np.int64)
    for i in range(len(y_true)):
        w[y_pred[i], y_true[i]] += 1
    row, col = linear_sum_assignment(w.max() - w)
    return w[row, col].sum() / len(y_true)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """返回 acc、ari、nmi 三指标字典。"""
    return {
        "acc": cluster_accuracy(y_true, y_pred),
        "ari": adjusted_rand_score(y_true, y_pred),
        "nmi": normalized_mutual_info_score(y_true, y_pred),
    }
