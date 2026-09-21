"""Generate evaluation figures from saved CV metrics and checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from jnt_data import apply_normalizer, load_jnt, subject_folds, CH_NAMES
from jnt_models import MaskedJointModel, PairwiseJointModel


COLORS = {"Observed": "#222222", "Masked": "#4477AA", "Pairwise": "#EE6677"}
DESIGNS = (("Masked", "masked_metrics.json"), ("Pairwise", "pairwise_metrics.json"))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
        }
    )


def read_metrics(metrics_dir: Path) -> dict[str, dict]:
    return {
        design: json.loads((metrics_dir / filename).read_text())
        for design, filename in DESIGNS
    }


def save_figure(fig: plt.Figure, path: Path) -> None:
    fig.savefig(path, dpi=220, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_metric_summary(metrics: dict[str, dict], output_dir: Path) -> None:
    specs = (("r", "Correlation (R)", (0.88, 1.005)), ("rmse", "RMSE (degrees)", None),
             ("nrmse", "NRMSE (%)", None))
    x = np.arange(len(CH_NAMES))
    width = 0.34
    fig, axes = plt.subplots(1, 3, figsize=(12.8, 3.8))
    for ax, (key, label, ylim) in zip(axes, specs):
        upper_extent = 0.0
        for design_index, (design, _) in enumerate(DESIGNS):
            means = [metrics[design]["metrics"][joint][key]["mean"] for joint in CH_NAMES]
            stds = [metrics[design]["metrics"][joint][key]["std"] for joint in CH_NAMES]
            offset = (design_index - 0.5) * width
            bars = ax.bar(
                x + offset,
                means,
                width,
                yerr=stds,
                capsize=3,
                color=COLORS[design],
                alpha=0.92,
                label=design,
                linewidth=0,
            )
            label_padding = 0.002 if key == "r" else (0.08 if key == "rmse" else 0.18)
            for bar, value, uncertainty in zip(bars, means, stds):
                precision = 3 if key == "r" else 2
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    value + uncertainty + label_padding,
                    f"{value:.{precision}f}",
                    ha="center",
                    va="bottom",
                    fontsize=8,
                    color="#222222",
                )
            upper_extent = max(upper_extent, max(np.asarray(means) + np.asarray(stds)))
        ax.set_xticks(x, CH_NAMES)
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(axis="x", visible=False)
        if ylim:
            ax.set_ylim(*ylim)
        else:
            ax.set_ylim(0, upper_extent * 1.12)
    axes[0].legend(loc="lower left")
    fig.suptitle("Five-fold held-out-subject performance", fontsize=14, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output_dir / "cv_metrics_summary.png")


def load_oof_predictions(
    data_path: Path,
    masked_dir: Path,
    pairwise_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    jnt, subject_ids = load_jnt(data_path)
    raw = np.load(data_path)
    sides = raw["sides"] if "sides" in raw else np.full(jnt.shape[0], "")
    masked_pred = np.empty_like(jnt)
    pairwise_pred = np.empty_like(jnt)

    for fold, (_, val_indices, _) in enumerate(subject_folds(subject_ids, 5), start=1):
        masked_norm = np.load(masked_dir / f"masked_jnt_fold{fold}_normalization.npz")
        normalized = apply_normalizer(jnt[val_indices], masked_norm["mean"], masked_norm["std"])
        x = torch.tensor(normalized)
        masked_model = MaskedJointModel(n_channels=jnt.shape[2])
        masked_model.load_state_dict(
            torch.load(masked_dir / f"masked_jnt_fold{fold}.pt", map_location="cpu", weights_only=True)
        )
        masked_model.eval()
        with torch.no_grad():
            for channel in range(jnt.shape[2]):
                mask = torch.zeros(x.shape[0], jnt.shape[2], dtype=torch.bool)
                mask[:, channel] = True
                prediction = masked_model(x, mask)[:, :, channel].numpy()
                masked_pred[val_indices, :, channel] = (
                    prediction * masked_norm["std"][0, 0, channel]
                    + masked_norm["mean"][0, 0, channel]
                )

        pairwise_norm = np.load(pairwise_dir / f"pairwise_jnt_fold{fold}_normalization.npz")
        normalized = apply_normalizer(jnt[val_indices], pairwise_norm["mean"], pairwise_norm["std"])
        for channel in range(jnt.shape[2]):
            source_channels = [i for i in range(jnt.shape[2]) if i != channel]
            model = PairwiseJointModel(in_dim=jnt.shape[2] - 1, out_dim=1)
            model.load_state_dict(
                torch.load(
                    pairwise_dir / f"pairwise_jnt_ch{channel}_fold{fold}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            model.eval()
            with torch.no_grad():
                prediction = model(torch.tensor(normalized[:, :, source_channels])).numpy()[:, :, 0]
            pairwise_pred[val_indices, :, channel] = (
                prediction * pairwise_norm["std"][0, 0, channel]
                + pairwise_norm["mean"][0, 0, channel]
            )

    return jnt, masked_pred, pairwise_pred, subject_ids, sides


def plot_oof_curves(
    true: np.ndarray,
    masked: np.ndarray,
    pairwise: np.ndarray,
    output_dir: Path,
) -> None:
    gait = np.linspace(0, 100, true.shape[1])
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.8), sharex=True)
    for channel, ax in enumerate(axes):
        observed_mean = true[:, :, channel].mean(axis=0)
        lower, upper = np.percentile(true[:, :, channel], [10, 90], axis=0)
        ax.fill_between(gait, lower, upper, color="#BBBBBB", alpha=0.25, label="Observed 10–90%")
        ax.plot(gait, observed_mean, color=COLORS["Observed"], linewidth=2.1, label="Observed mean")
        ax.plot(gait, masked[:, :, channel].mean(axis=0), color=COLORS["Masked"], linewidth=1.9, label="Masked")
        ax.plot(gait, pairwise[:, :, channel].mean(axis=0), color=COLORS["Pairwise"], linewidth=1.9, label="Pairwise")
        ax.set_title(CH_NAMES[channel])
        ax.set_xlabel("Gait cycle (%)")
        ax.set_ylabel("Joint angle (degrees)")
        ax.set_xlim(0, 100)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("Out-of-fold joint-angle reconstruction", fontsize=14, fontweight="bold", y=1.12)
    fig.tight_layout()
    save_figure(fig, output_dir / "oof_reconstruction_curves.png")


def plot_error_distributions(
    true: np.ndarray,
    masked: np.ndarray,
    pairwise: np.ndarray,
    output_dir: Path,
) -> None:
    masked_rmse = np.sqrt(np.mean((masked - true) ** 2, axis=1))
    pairwise_rmse = np.sqrt(np.mean((pairwise - true) ** 2, axis=1))
    positions = []
    values = []
    colors = []
    for channel in range(3):
        positions.extend([channel * 3 + 1, channel * 3 + 2])
        values.extend([masked_rmse[:, channel], pairwise_rmse[:, channel]])
        colors.extend([COLORS["Masked"], COLORS["Pairwise"]])

    fig, ax = plt.subplots(figsize=(8.5, 4.7))
    boxes = ax.boxplot(
        values,
        positions=positions,
        widths=0.72,
        whis=(5, 95),
        showfliers=False,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 1.6},
        whiskerprops={"color": "#666666"},
        capprops={"color": "#666666"},
    )
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.92)
    for position, distribution in zip(positions, values):
        median = np.median(distribution)
        ax.text(
            position,
            median + 0.12,
            f"{median:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="white",
            fontweight="bold",
        )
    ax.set_xticks([1.5, 4.5, 7.5], CH_NAMES)
    ax.set_ylabel("Per-cycle RMSE (degrees)")
    ax.set_title(
        "Out-of-fold reconstruction error (whiskers: 5th–95th percentile)",
        fontsize=13,
        fontweight="bold",
    )
    ax.grid(axis="x", visible=False)
    legend_handles = [
        plt.Line2D([0], [0], color=COLORS[name], linewidth=8, label=name)
        for name in ("Masked", "Pairwise")
    ]
    ax.legend(handles=legend_handles, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_dir / "oof_error_distributions.png")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics_dir", type=Path, default=Path("results"))
    parser.add_argument("--output_dir", type=Path, default=Path("results/figures"))
    parser.add_argument("--data", type=Path)
    parser.add_argument("--masked_checkpoints", type=Path)
    parser.add_argument("--pairwise_checkpoints", type=Path)
    args = parser.parse_args()

    configure_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics = read_metrics(args.metrics_dir)
    plot_metric_summary(metrics, args.output_dir)

    prediction_args = (args.data, args.masked_checkpoints, args.pairwise_checkpoints)
    if any(prediction_args) and not all(prediction_args):
        parser.error("--data, --masked_checkpoints, and --pairwise_checkpoints must be used together")
    if all(prediction_args):
        true, masked, pairwise, _, _ = load_oof_predictions(*prediction_args)
        plot_oof_curves(true, masked, pairwise, args.output_dir)
        plot_error_distributions(true, masked, pairwise, args.output_dir)

    for path in sorted(args.output_dir.glob("*")):
        print(path)


if __name__ == "__main__":
    main()
