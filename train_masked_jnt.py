"""Design A: mask one of the 3 joint channels and reconstruct it from the other two.
One model learns all 3 directions; subject-stratified CV tests cross-subject generality.

    python train_masked_jnt.py --data /path/to/gait_r_label_normalized+jnt.npz
"""
import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from jnt_data import (
    apply_normalizer,
    fit_normalizer,
    load_jnt,
    subject_folds,
    CH_NAMES,
)
from jnt_models import MaskedJointModel
from jnt_metrics import aggregate_per_cycle
from jnt_engine import fit, get_device

def masked_train_step(model, batch, device):
    """Random channel masked per sample; MSE on the masked channel only."""
    (x,) = batch
    x = x.to(device)
    B, T, C = x.shape
    mask = F.one_hot(torch.randint(0, C, (B,), device=device), C).bool()  # (B, C)
    pred = model(x, mask)
    return ((pred - x) ** 2)[mask.unsqueeze(1).expand(B, T, C)].mean()


def masked_val_step(model, batch, device):
    """Deterministic: mask each channel in turn, mean reconstruction loss."""
    (x,) = batch
    x = x.to(device)
    B, T, C = x.shape
    losses = []
    for c in range(C):
        mask = torch.zeros(B, C, dtype=torch.bool, device=device)
        mask[:, c] = True
        losses.append(((model(x, mask)[:, :, c] - x[:, :, c]) ** 2).mean())
    return torch.stack(losses).mean()


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
        tr_ld = DataLoader(TensorDataset(torch.tensor(jnt_n[tr])),
                           batch_size=args.batch_size, shuffle=True)
        va_ld = DataLoader(TensorDataset(torch.tensor(jnt_n[va])),
                           batch_size=args.batch_size)

        model = MaskedJointModel(n_channels=C)
        fit(model, tr_ld, va_ld, masked_train_step, masked_val_step, device,
            epochs=args.epochs)
        stem = os.path.join(args.output_dir, f"masked_jnt_fold{fold + 1}")
        torch.save(model.state_dict(), f"{stem}.pt")
        np.savez(
            f"{stem}_normalization.npz",
            mean=mean,
            std=std,
            channel_names=np.asarray(CH_NAMES),
            held_out_subjects=val_subj,
        )

        # Per-direction evaluation, denormalized to physical units.
        model.eval()
        xv = torch.tensor(jnt_n[va]).to(device)
        with torch.no_grad():
            for c in range(C):
                mask = torch.zeros(xv.shape[0], C, dtype=torch.bool, device=device)
                mask[:, c] = True
                pred = model(xv, mask)[:, :, c].cpu().numpy() * std[0, 0, c] + mean[0, 0, c]
                true = jnt_n[va][:, :, c] * std[0, 0, c] + mean[0, 0, c]
                r, rmse, nrmse = aggregate_per_cycle(true, pred)
                per_dir[c]["r"].append(r)
                per_dir[c]["rmse"].append(rmse)
                per_dir[c]["nrmse"].append(nrmse)
                print(f"  predict {CH_NAMES[c]} from others: "
                      f"R={r:.4f} RMSE={rmse:.4f} NRMSE={nrmse:.2f}%")

    print("\n" + "=" * 56)
    print("Masked model -- subject-stratified CV summary (mean +/- std)")
    print("=" * 56)
    for c in range(C):
        r, rmse, nr = per_dir[c]["r"], per_dir[c]["rmse"], per_dir[c]["nrmse"]
        print(f"predict {CH_NAMES[c]:>4} | R {np.nanmean(r):.4f}+/-{np.nanstd(r):.4f}"
              f" | RMSE {np.nanmean(rmse):.4f}+/-{np.nanstd(rmse):.4f}"
              f" | NRMSE {np.nanmean(nr):.2f}%+/-{np.nanstd(nr):.2f}%")

    summary = {
        "model": "masked_joint",
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
    with open(os.path.join(args.output_dir, "masked_metrics.json"), "w") as f:
        json.dump(summary, f, indent=2)


if __name__ == "__main__":
    main()
