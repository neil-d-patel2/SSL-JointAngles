"""Train temporal inpainting with 25% of time points masked across all joints.

Unlike whole-channel masking, every joint is unavailable at a masked instant.
The model must therefore use surrounding time points rather than a same-time
mapping from the other joint channels.
"""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from jnt_data import apply_normalizer, fit_normalizer, load_jnt, CH_NAMES
from jnt_engine import fit, get_device
from jnt_metrics import aggregate_per_cycle
from jnt_models import MaskedJointModel
from train_test_split_jnt import subject_split


def make_temporal_mask(
    batch_size: int,
    time_steps: int,
    channels: int,
    mask_ratio: float,
    span_length: int,
    device: torch.device,
    deterministic: bool = False,
) -> torch.Tensor:
    """Mask an exact fraction of time points in short blocks across all channels."""
    if not 0.0 < mask_ratio < 1.0:
        raise ValueError("mask_ratio must be between 0 and 1")
    if not 1 <= span_length <= time_steps:
        raise ValueError("span_length must be in [1, time_steps]")

    n_mask = max(1, int(round(time_steps * mask_ratio)))
    span_lengths = [span_length] * (n_mask // span_length)
    if n_mask % span_length:
        span_lengths.append(n_mask % span_length)
    time_mask = torch.zeros(batch_size, time_steps, dtype=torch.bool)

    for sample in range(batch_size):
        for span_index, length in enumerate(span_lengths):
            available_starts = [
                start
                for start in range(time_steps - length + 1)
                if not time_mask[sample, start : start + length].any()
            ]
            if not available_starts:
                raise RuntimeError("could not place non-overlapping temporal mask spans")
            if deterministic:
                choice = (sample * 7 + span_index * 13) % len(available_starts)
            else:
                choice = int(torch.randint(len(available_starts), (1,)).item())
            start = available_starts[choice]
            time_mask[sample, start : start + length] = True

    mask = time_mask.unsqueeze(-1).expand(batch_size, time_steps, channels)
    return mask.to(device)


def temporal_step(
    model,
    batch,
    device,
    mask_ratio: float,
    span_length: int,
    deterministic: bool,
):
    (x,) = batch
    x = x.to(device)
    mask = make_temporal_mask(
        x.shape[0],
        x.shape[1],
        x.shape[2],
        mask_ratio,
        span_length,
        device,
        deterministic,
    )
    prediction = model(x, mask)
    return ((prediction - x) ** 2)[mask].mean()


def masked_test_values(
    model: MaskedJointModel,
    normalized: np.ndarray,
    physical: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
    mask_ratio: float,
    span_length: int,
    device: torch.device,
):
    x = torch.tensor(normalized).to(device)
    mask = make_temporal_mask(
        x.shape[0],
        x.shape[1],
        x.shape[2],
        mask_ratio,
        span_length,
        device,
        deterministic=True,
    )
    model.eval()
    with torch.no_grad():
        prediction = model(x, mask).cpu().numpy()
    prediction = prediction * std + mean
    time_mask = mask[:, :, 0].cpu().numpy()
    true_masked = np.stack([physical[i, time_mask[i], :] for i in range(len(physical))])
    pred_masked = np.stack([prediction[i, time_mask[i], :] for i in range(len(physical))])
    return true_masked, pred_masked, time_mask


def interpolation_baseline(physical: np.ndarray, time_mask: np.ndarray) -> np.ndarray:
    """Linear interpolation using visible values only, returned at masked points."""
    baseline = np.empty((physical.shape[0], int(time_mask[0].sum()), physical.shape[2]))
    for sample in range(len(physical)):
        hidden = np.flatnonzero(time_mask[sample])
        visible = np.flatnonzero(~time_mask[sample])
        for channel in range(physical.shape[2]):
            baseline[sample, :, channel] = np.interp(
                hidden, visible, physical[sample, visible, channel]
            )
    return baseline.astype(np.float32)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_fraction", type=float, default=0.15)
    parser.add_argument("--val_fraction", type=float, default=0.15)
    parser.add_argument("--mask_ratio", type=float, default=0.25)
    parser.add_argument("--span_length", type=int, default=5)
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
        indices["train"], indices["validation"], indices["test"]
    )
    mean, std = fit_normalizer(jnt[train_idx])
    normalized = apply_normalizer(jnt, mean, std)

    np.savez(
        args.output_dir / "temporal_masked_normalization.npz",
        mean=mean,
        std=std,
        channel_names=np.asarray(CH_NAMES),
        **{f"{name}_subjects": values for name, values in split_subjects.items()},
    )
    print(f"Device: {device}")
    print(
        f"Mask: {args.mask_ratio:.0%} of time points, "
        f"span length {args.span_length}, all channels"
    )
    for name in ("train", "validation", "test"):
        print(
            f"{name:>10}: {len(split_subjects[name])} subjects, "
            f"{len(indices[name])} cycles"
        )

    train_loader = DataLoader(
        TensorDataset(torch.tensor(normalized[train_idx])),
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(normalized[val_idx])),
        batch_size=args.batch_size,
    )
    train_step = partial(
        temporal_step,
        mask_ratio=args.mask_ratio,
        span_length=args.span_length,
        deterministic=False,
    )
    val_step = partial(
        temporal_step,
        mask_ratio=args.mask_ratio,
        span_length=args.span_length,
        deterministic=True,
    )
    model = MaskedJointModel(n_channels=jnt.shape[2])
    best_validation_loss = fit(
        model,
        train_loader,
        val_loader,
        train_step,
        val_step,
        device,
        epochs=args.epochs,
    )
    torch.save(model.state_dict(), args.output_dir / "temporal_masked_test_split.pt")

    true_masked, pred_masked, test_mask = masked_test_values(
        model,
        normalized[test_idx],
        jnt[test_idx],
        mean,
        std,
        args.mask_ratio,
        args.span_length,
        device,
    )
    baseline_predictions = interpolation_baseline(jnt[test_idx], test_mask)
    metrics = {}
    baseline_metrics = {}
    for channel, name in enumerate(CH_NAMES):
        r, rmse, nrmse = aggregate_per_cycle(
            true_masked[:, :, channel], pred_masked[:, :, channel]
        )
        metrics[name] = {"r": float(r), "rmse": float(rmse), "nrmse": float(nrmse)}
        r, rmse, nrmse = aggregate_per_cycle(
            true_masked[:, :, channel], baseline_predictions[:, :, channel]
        )
        baseline_metrics[name] = {
            "r": float(r),
            "rmse": float(rmse),
            "nrmse": float(nrmse),
        }

    raw = np.load(args.data)
    predictions = {
        "true_masked": true_masked,
        "predicted_masked": pred_masked,
        "interpolation_baseline": baseline_predictions,
        "time_mask": test_mask,
        "subject_ids": subject_ids[test_idx],
    }
    if "sides" in raw:
        predictions["sides"] = raw["sides"][test_idx]
    np.savez_compressed(args.output_dir / "temporal_test_predictions.npz", **predictions)

    results = {
        "protocol": "subject_disjoint_temporal_masking",
        "seed": args.seed,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "mask_ratio": args.mask_ratio,
        "span_length": args.span_length,
        "masking": "same time points across all joint channels",
        "metric_scope": "masked test points only",
        "subjects": {name: values.tolist() for name, values in split_subjects.items()},
        "cycles": {name: int(len(values)) for name, values in indices.items()},
        "best_validation_loss": float(best_validation_loss),
        "test_metrics": metrics,
        "interpolation_baseline_metrics": baseline_metrics,
    }
    with open(args.output_dir / "temporal_test_results.json", "w") as output:
        json.dump(results, output, indent=2)

    print("\n=== Temporal masking: untouched test-set masked points ===")
    for name in CH_NAMES:
        values = metrics[name]
        print(
            f"  {name:>5}: R={values['r']:.4f} "
            f"RMSE={values['rmse']:.4f} degrees "
            f"NRMSE={values['nrmse']:.2f}%"
        )
    print("\n=== Linear interpolation baseline on the same masked points ===")
    for name in CH_NAMES:
        values = baseline_metrics[name]
        print(
            f"  {name:>5}: R={values['r']:.4f} "
            f"RMSE={values['rmse']:.4f} degrees "
            f"NRMSE={values['nrmse']:.2f}%"
        )
    print(f"\nSaved results to {args.output_dir / 'temporal_test_results.json'}")


if __name__ == "__main__":
    main()
