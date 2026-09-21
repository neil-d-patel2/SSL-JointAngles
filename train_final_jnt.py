"""Train deployable masked and/or pairwise models on all available cycles.

Use cross-validation metrics from the other training scripts for evaluation.
This script intentionally uses every sample to produce final initialization
weights after model selection is complete.
"""

import argparse
import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from jnt_data import apply_normalizer, fit_normalizer, load_jnt, CH_NAMES
from jnt_engine import fit, get_device
from jnt_models import MaskedJointModel, PairwiseJointModel
from train_masked_jnt import masked_train_step, masked_val_step
from train_pairwise_jnt import mse_step, split_src_tgt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--output_dir", default="checkpoints/final")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--design", choices=("both", "masked", "pairwise"), default="both")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()

    jnt, subject_ids = load_jnt(args.data)
    mean, std = fit_normalizer(jnt)
    jnt_n = apply_normalizer(jnt, mean, std)
    np.savez(
        os.path.join(args.output_dir, "full_data_normalization.npz"),
        mean=mean,
        std=std,
        channel_names=np.asarray(CH_NAMES),
        subjects=np.unique(subject_ids),
    )
    print(
        f"Device: {device} | {jnt.shape[0]} cycles, "
        f"{len(np.unique(subject_ids))} subjects, {jnt.shape[2]} channels",
        flush=True,
    )

    if args.design in ("both", "masked"):
        print("\n--- Full-data masked model ---", flush=True)
        loader = DataLoader(
            TensorDataset(torch.tensor(jnt_n)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        eval_loader = DataLoader(
            TensorDataset(torch.tensor(jnt_n)), batch_size=args.batch_size
        )
        model = MaskedJointModel(n_channels=jnt.shape[2])
        best = fit(
            model,
            loader,
            eval_loader,
            masked_train_step,
            masked_val_step,
            device,
            epochs=args.epochs,
        )
        torch.save(model.state_dict(), os.path.join(args.output_dir, "masked_jnt_full.pt"))
        print(f"Saved masked_jnt_full.pt | reconstruction loss {best:.5f}", flush=True)

    if args.design in ("both", "pairwise"):
        for target_c in range(jnt.shape[2]):
            source, target = split_src_tgt(jnt_n, target_c, jnt.shape[2])
            loader = DataLoader(
                TensorDataset(torch.tensor(source), torch.tensor(target)),
                batch_size=args.batch_size,
                shuffle=True,
            )
            eval_loader = DataLoader(
                TensorDataset(torch.tensor(source), torch.tensor(target)),
                batch_size=args.batch_size,
            )
            model = PairwiseJointModel(in_dim=jnt.shape[2] - 1, out_dim=1)
            print(f"\n--- Full-data pairwise target: {CH_NAMES[target_c]} ---", flush=True)
            best = fit(
                model,
                loader,
                eval_loader,
                mse_step,
                mse_step,
                device,
                epochs=args.epochs,
            )
            path = os.path.join(args.output_dir, f"pairwise_jnt_ch{target_c}_full.pt")
            torch.save(model.state_dict(), path)
            print(f"Saved {os.path.basename(path)} | reconstruction loss {best:.5f}", flush=True)


if __name__ == "__main__":
    main()
