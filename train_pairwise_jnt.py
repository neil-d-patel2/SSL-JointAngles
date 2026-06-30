"""Design B: three directional regressors, one per held-out channel
(e.g. knee + ankle -> hip). Subject-stratified CV.

    python train_pairwise_jnt.py --data /path/to/gait_r_label_normalized+jnt.npz
"""
import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from jnt_data import load_jnt, normalize_global, subject_folds, CH_NAMES
from jnt_models import PairwiseJointModel
from jnt_metrics import aggregate_per_cycle
from jnt_engine import fit, get_device

CKPT_DIR = "checkpoints"


def mse_step(model, batch, device):
    src, tgt = batch
    return ((model(src.to(device)) - tgt.to(device)) ** 2).mean()


def split_src_tgt(arr, target_c, C):
    """(N, T, C) -> (src (N,T,C-1), tgt (N,T,1)) for predicting channel target_c."""
    src_ch = [c for c in range(C) if c != target_c]
    return arr[:, :, src_ch], arr[:, :, target_c:target_c + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="processed .npz with final_jnt")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    os.makedirs(CKPT_DIR, exist_ok=True)

    device = get_device()
    jnt, subject_ids = load_jnt(args.data)
    jnt_n, mean, std = normalize_global(jnt)
    C = jnt.shape[2]
    print(f"Device: {device} | {jnt.shape[0]} cycles, "
          f"{len(np.unique(subject_ids))} subjects, {C} channels")

    per_dir = {c: {"r": [], "rmse": [], "nrmse": []} for c in range(C)}

    for fold, (tr, va, val_subj) in enumerate(subject_folds(subject_ids, args.folds)):
        print(f"\n--- Fold {fold + 1} | held-out subjects: {val_subj.tolist()} ---")
        for target_c in range(C):
            src_tr, tgt_tr = split_src_tgt(jnt_n[tr], target_c, C)
            src_va, tgt_va = split_src_tgt(jnt_n[va], target_c, C)

            tr_ld = DataLoader(TensorDataset(torch.tensor(src_tr), torch.tensor(tgt_tr)),
                               batch_size=args.batch_size, shuffle=True)
            va_ld = DataLoader(TensorDataset(torch.tensor(src_va), torch.tensor(tgt_va)),
                               batch_size=args.batch_size)

            model = PairwiseJointModel(in_dim=C - 1, out_dim=1)
            fit(model, tr_ld, va_ld, mse_step, mse_step, device,
                epochs=args.epochs, verbose=False)
            torch.save(model.state_dict(),
                       f"{CKPT_DIR}/pairwise_jnt_ch{target_c}_fold{fold + 1}.pt")

            model.eval()
            with torch.no_grad():
                pred = model(torch.tensor(src_va).to(device)).cpu().numpy()[:, :, 0]
            pred = pred * std[0, 0, target_c] + mean[0, 0, target_c]
            true = tgt_va[:, :, 0] * std[0, 0, target_c] + mean[0, 0, target_c]
            r, rmse, nrmse = aggregate_per_cycle(true, pred)
            per_dir[target_c]["r"].append(r)
            per_dir[target_c]["rmse"].append(rmse)
            per_dir[target_c]["nrmse"].append(nrmse)
            src_names = "+".join(CH_NAMES[c] for c in range(C) if c != target_c)
            print(f"  {src_names} -> {CH_NAMES[target_c]}: "
                  f"R={r:.4f} RMSE={rmse:.4f} NRMSE={nrmse:.2f}%")

    print("\n" + "=" * 56)
    print("Pairwise models -- subject-stratified CV summary (mean +/- std)")
    print("=" * 56)
    for c in range(C):
        r, rmse, nr = per_dir[c]["r"], per_dir[c]["rmse"], per_dir[c]["nrmse"]
        src_names = "+".join(CH_NAMES[k] for k in range(C) if k != c)
        print(f"{src_names} -> {CH_NAMES[c]:>4} | R {np.nanmean(r):.4f}+/-{np.nanstd(r):.4f}"
              f" | RMSE {np.nanmean(rmse):.4f}+/-{np.nanstd(rmse):.4f}"
              f" | NRMSE {np.nanmean(nr):.2f}%+/-{np.nanstd(nr):.2f}%")


if __name__ == "__main__":
    main()
