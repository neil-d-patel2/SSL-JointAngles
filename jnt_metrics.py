"""Regression metrics (mirrors Gait_DP-RGNet/src/utils.get_metrics)."""
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr


def get_metrics(y_true, y_pred):
    """-> (R, RMSE, NRMSE-over-range, CoD) on flattened arrays."""
    yt = np.asarray(y_true).flatten()
    yp = np.asarray(y_pred).flatten()
    rng = np.max(yt) - np.min(yt)
    r, _ = pearsonr(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    nrmse = rmse / rng if rng > 1e-8 else np.nan
    cod = r2_score(yt, yp, force_finite=True)
    return r, rmse, nrmse, cod


def aggregate_per_cycle(true_seqs, pred_seqs):
    """Per-cycle (N, T) in physical units -> mean (R, RMSE, NRMSE%) over cycles."""
    rs, rmses, nrmses = [], [], []
    for t, p in zip(true_seqs, pred_seqs):
        r, rmse, nrmse, _ = get_metrics(t, p)
        rs.append(r)
        rmses.append(rmse)
        nrmses.append(nrmse * 100)
    return np.nanmean(rs), np.nanmean(rmses), np.nanmean(nrmses)
