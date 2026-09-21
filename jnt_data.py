"""Load final_jnt from the processed .npz and build subject-stratified folds.

Global channel-wise normalization (matches how the main model normalizes its
joint-angle outputs), so the learned relationship stays transplantable.
"""
import numpy as np
from sklearn.model_selection import GroupKFold

# Channel order, proximal->distal sagittal-plane angles. Three separate joints.
CH_NAMES = ["Hip", "Knee", "Ankle"]


def load_jnt(npz_path):
    """-> (jnt (N, T, 3) float32, subject_ids (N,))."""
    data = np.load(npz_path)
    jnt = np.transpose(data["final_jnt"], (2, 0, 1)).astype(np.float32)  # (T,3,N)->(N,T,3)
    return jnt, data["subject_ids"]


def normalize_global(jnt):
    """Channel-wise normalize. -> (normed, mean, std) with mean/std shaped (1,1,3)."""
    mean = jnt.mean(axis=(0, 1), keepdims=True)
    std = jnt.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return (jnt - mean) / std, mean, std


def fit_normalizer(jnt):
    """Fit channel-wise normalization parameters on training samples only."""
    mean = jnt.mean(axis=(0, 1), keepdims=True)
    std = jnt.std(axis=(0, 1), keepdims=True)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_normalizer(jnt, mean, std):
    return ((jnt - mean) / std).astype(np.float32)


def subject_folds(subject_ids, n_splits=5):
    """Yield (train_idx, val_idx, val_subjects); no subject shared across a split."""
    n_splits = min(n_splits, len(np.unique(subject_ids)))
    gkf = GroupKFold(n_splits=n_splits)
    dummy = np.zeros(len(subject_ids))
    for train_idx, val_idx in gkf.split(dummy, groups=subject_ids):
        yield train_idx, val_idx, np.unique(subject_ids[val_idx])
