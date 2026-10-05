"""Train and evaluate joint-angle models on a fixed subject-level test split."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from jnt_data import apply_normalizer, fit_normalizer, load_jnt, CH_NAMES
from jnt_engine import fit, get_device
from jnt_metrics import aggregate_per_cycle
from jnt_models import MaskedJointModel, PairwiseJointModel
from train_masked_jnt import masked_train_step, masked_val_step
from train_pairwise_jnt import mse_step, split_src_tgt


def subject_split(subject_ids: np.ndarray, seed: int, test_fraction: float, val_fraction: float):
    """Return sample indices for deterministic, mutually exclusive subject splits."""
    subjects = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    subjects = subjects.copy()
    rng.shuffle(subjects)

    n_test = max(1, int(round(len(subjects) * test_fraction)))
    n_val = max(1, int(round(len(subjects) * val_fraction)))
    if n_test + n_val >= len(subjects):
        raise ValueError("test_fraction + val_fraction leaves no training subjects")

    test_subjects = np.sort(subjects[:n_test])
    val_subjects = np.sort(subjects[n_test : n_test + n_val])
    train_subjects = np.sort(subjects[n_test + n_val :])

    indices = {
        "train": np.flatnonzero(np.isin(subject_ids, train_subjects)),
        "validation": np.flatnonzero(np.isin(subject_ids, val_subjects)),
        "test": np.flatnonzero(np.isin(subject_ids, test_subjects)),
    }
    split_subjects = {
        "train": train_subjects,
        "validation": val_subjects,
        "test": test_subjects,
    }
    return indices, split_subjects


def metric_dict(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    r, rmse, nrmse = aggregate_per_cycle(true, predicted)
    return {"r": float(r), "rmse": float(rmse), "nrmse": float(nrmse)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = get_device()

    jnt, subject_ids = load_jnt(args.data)
    indices, split_subjects = subject_split(
        subject_ids, args.seed, args.test_fraction, args.val_fraction
    )
    train_idx, val_idx, test_idx = (
        indices["train"],
        indices["validation"],
        indices["test"],
    )
    mean, std = fit_normalizer(jnt[train_idx])
    normalized = apply_normalizer(jnt, mean, std)
    np.savez(
        args.output_dir / "test_split_normalization.npz",
        mean=mean,
        std=std,
        channel_names=np.asarray(CH_NAMES),
        **{f"{name}_subjects": values for name, values in split_subjects.items()},
    )

    print(f"Device: {device}")
    for name in ("train", "validation", "test"):
        print(
            f"{name:>10}: {len(split_subjects[name])} subjects, "
            f"{len(indices[name])} cycles"
        )

    test_true = jnt[test_idx]
    masked_predictions = np.empty_like(test_true)
    pairwise_predictions = np.empty_like(test_true)

    print("\n--- Masked model ---", flush=True)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(normalized[train_idx])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(normalized[val_idx])),
        batch_size=args.batch_size,
    )
    masked_model = MaskedJointModel(n_channels=jnt.shape[2])
    best_masked_loss = fit(
        masked_model,
        train_loader,
        val_loader,
        masked_train_step,
        masked_val_step,
        device,
        epochs=args.epochs,
    )
    torch.save(masked_model.state_dict(), args.output_dir / "masked_jnt_test_split.pt")
    masked_model.eval()
    test_x = torch.tensor(normalized[test_idx]).to(device)
    with torch.no_grad():
        for channel in range(jnt.shape[2]):
            mask = torch.zeros(test_x.shape[0], jnt.shape[2], dtype=torch.bool, device=device)
            mask[:, channel] = True
            prediction = masked_model(test_x, mask)[:, :, channel].cpu().numpy()
            masked_predictions[:, :, channel] = (
                prediction * std[0, 0, channel] + mean[0, 0, channel]
            )

    print("\n--- Pairwise models ---", flush=True)
    for target_channel in range(jnt.shape[2]):
        source_train, target_train = split_src_tgt(
            normalized[train_idx], target_channel, jnt.shape[2]
        )
        source_val, target_val = split_src_tgt(
            normalized[val_idx], target_channel, jnt.shape[2]
        )
        train_loader = DataLoader(
            TensorDataset(torch.tensor(source_train), torch.tensor(target_train)),
            batch_size=args.batch_size,
            shuffle=True,
        )
        val_loader = DataLoader(
            TensorDataset(torch.tensor(source_val), torch.tensor(target_val)),
            batch_size=args.batch_size,
        )
        model = PairwiseJointModel(in_dim=jnt.shape[2] - 1, out_dim=1)
        print(f"Target: {CH_NAMES[target_channel]}", flush=True)
        fit(
            model,
            train_loader,
            val_loader,
            mse_step,
            mse_step,
            device,
            epochs=args.epochs,
        )
        torch.save(
            model.state_dict(),
            args.output_dir / f"pairwise_jnt_ch{target_channel}_test_split.pt",
        )
        source_test, _ = split_src_tgt(
            normalized[test_idx], target_channel, jnt.shape[2]
        )
        model.eval()
        with torch.no_grad():
            prediction = model(torch.tensor(source_test).to(device)).cpu().numpy()[:, :, 0]
        pairwise_predictions[:, :, target_channel] = (
            prediction * std[0, 0, target_channel] + mean[0, 0, target_channel]
        )

    metrics = {"masked": {}, "pairwise": {}}
    for channel, name in enumerate(CH_NAMES):
        metrics["masked"][name] = metric_dict(
            test_true[:, :, channel], masked_predictions[:, :, channel]
        )
        metrics["pairwise"][name] = metric_dict(
            test_true[:, :, channel], pairwise_predictions[:, :, channel]
        )

    raw = np.load(args.data)
    payload = {
        "true": test_true,
        "masked": masked_predictions,
        "pairwise": pairwise_predictions,
        "subject_ids": subject_ids[test_idx],
    }
    if "sides" in raw:
        payload["sides"] = raw["sides"][test_idx]
    np.savez_compressed(args.output_dir / "test_predictions.npz", **payload)

    results = {
        "protocol": "subject_disjoint_train_validation_test",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "fractions": {
            "train": 1.0 - args.test_fraction - args.val_fraction,
            "validation": args.val_fraction,
            "test": args.test_fraction,
        },
        "subjects": {name: values.tolist() for name, values in split_subjects.items()},
        "cycles": {name: int(len(values)) for name, values in indices.items()},
        "best_masked_validation_loss": float(best_masked_loss),
        "test_metrics": metrics,
    }
    with open(args.output_dir / "test_results.json", "w") as output:
        json.dump(results, output, indent=2)

    print("\n=== Untouched test-set results ===")
    for design in ("masked", "pairwise"):
        print(f"\n{design.title()}")
        for name in CH_NAMES:
            values = metrics[design][name]
            print(
                f"  {name:>5}: R={values['r']:.4f} "
                f"RMSE={values['rmse']:.4f} degrees "
                f"NRMSE={values['nrmse']:.2f}%"
            )
    print(f"\nSaved results to {args.output_dir / 'test_results.json'}")


if __name__ == "__main__":
    main()
