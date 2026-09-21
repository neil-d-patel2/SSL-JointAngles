"""Design B: three directional regressors, one per held-out channel
(e.g. knee + ankle -> hip). Subject-stratified CV.

    python train_pairwise_jnt.py --data /path/to/gait_r_label_normalized+jnt.npz
"""
import argparse
import json
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from jnt_data import (
    apply_normalizer,
    fit_normalizer,
    load_jnt,
    subject_folds,
    CH_NAMES,
)
from jnt_models import PairwiseJointModel
from jnt_metrics import aggregate_per_cycle
from jnt_engine import fit, get_device

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
    ap.add_argument("--output_dir", default="checkpoints")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = get_device()
    jnt, subject_ids = load_jnt(args.data)
    C = jnt.shape[2]
    print(f"Device: {device} | {jnt.shape[0]} cycles, "
          f"{len(np.unique(subject_ids))} subjects, {C} channels")

    per_dir = {c: {"r": [], "rmse": [], "nrmse": []} for c in range(C)}

    for fold, (tr, va, val_subj) in enumerate(subject_folds(subject_ids, args.folds)):
        print(f"\n--- Fold {fold + 1} | held-out subjects: {val_subj.tolist()} ---")
        mean, std = fit_normalizer(jnt[tr])
        jnt_n = apply_normalizer(jnt, mean, std)
        norm_path = os.path.join(args.output_dir, f"pairwise_jnt_fold{fold + 1}_normalization.npz")
        np.savez(
            norm_path,
            mean=mean,
            std=std,
            channel_names=np.asarray(CH_NAMES),
            held_out_subjects=val_subj,
        )
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
                       os.path.join(
                           args.output_dir,
                           f"pairwise_jnt_ch{target_c}_fold{fold + 1}.pt",
                       ))

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

    summary = {
        "model": "pairwise_joint",
        "data": os.path.abspath(args.data),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "folds": args.folds,
        "seed": args.seed,
        "cycles": int(jnt.shape[0]),
        "subjects": int(len(np.unique(subject_ids))),
        "metrics": {
            CH_NAMES[c]: {
                metric: {
                    "fold_values": [float(x) for x in per_dir[c][metric]],
                    "mean": float(np.nanmean(per_dir[c][metric])),
                    "std": float(np.nanstd(per_dir[c][metric])),
                }
                for metric in ("r", "rmse", "nrmse")
            }
            for c in range(C)
        },
    }
    with open(os.path.join(args.output_dir, "pairwise_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
