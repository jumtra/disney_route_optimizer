import lightgbm as lgb
import numpy as np

# カスタムメトリクスの定義


def rmsle(y_pred: np.ndarray, dataset: lgb.Dataset):
    """RMLSLEを算出"""
    y_true = dataset.get_label()  # 真の値を取得
    log_diff = np.log1p(y_pred) - np.log1p(y_true)
    squared_log_diff = np.square(log_diff)
    mean_squared_log_diff = np.mean(squared_log_diff)
    rmsle_score = np.sqrt(mean_squared_log_diff)
    return "rmsle", rmsle_score, False
