"""Convert Plug-in Gait C3D trials into the NPZ format used by this project.

Each example is an ipsilateral foot-strike-to-foot-strike gait cycle. The
sagittal (X) components of the Vicon hip, knee, and ankle angle outputs are
resampled to a fixed number of gait-percent samples.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import ezc3d
import numpy as np


JOINTS = ("Hip", "Knee", "Ankle")
SIDE_CONTEXT = {"L": "left", "R": "right"}


def event_seconds(times: np.ndarray, index: int) -> float:
    """Return a C3D event time, whose two rows are minutes and seconds."""
    return 60.0 * float(times[0, index]) + float(times[1, index])


def resample_cycle(cycle: np.ndarray, n_points: int) -> np.ndarray:
    old_x = np.linspace(0.0, 1.0, cycle.shape[0])
    new_x = np.linspace(0.0, 1.0, n_points)
    return np.stack(
        [np.interp(new_x, old_x, cycle[:, channel]) for channel in range(3)],
        axis=1,
    ).astype(np.float32)


def build_dataset(
    data_dir: Path,
    sides: tuple[str, ...],
    n_points: int,
    min_cycle_seconds: float,
    max_cycle_seconds: float,
) -> tuple[dict[str, np.ndarray], Counter]:
    cycles: list[np.ndarray] = []
    subject_ids: list[str] = []
    cycle_sides: list[str] = []
    source_files: list[str] = []
    cycle_starts: list[float] = []
    cycle_ends: list[float] = []
    counts: Counter = Counter()

    for path in sorted(data_dir.rglob("*.c3d")):
        counts["c3d_files"] += 1
        c3d = ezc3d.c3d(str(path))
        point_labels = c3d["parameters"]["POINT"]["LABELS"]["value"]
        point_index = {label: index for index, label in enumerate(point_labels)}
        event_group = c3d["parameters"].get("EVENT", {})
        n_events = int(event_group.get("USED", {}).get("value", [0])[0])
        if n_events == 0:
            counts["files_without_events"] += 1
            continue

        labels = event_group["LABELS"]["value"]
        contexts = event_group["CONTEXTS"]["value"]
        times = event_group["TIMES"]["value"]
        rate = float(c3d["header"]["points"]["frame_rate"])
        first_frame = int(c3d["header"]["points"]["first_frame"])
        points = c3d["data"]["points"]
        file_produced_cycle = False

        for side in sides:
            angle_labels = [f"{side}{joint}Angles" for joint in JOINTS]
            if not all(label in point_index for label in angle_labels):
                counts["side_trials_without_angles"] += 1
                continue

            strikes = sorted(
                event_seconds(times, i)
                for i in range(n_events)
                if labels[i].strip().lower() == "foot strike"
                and contexts[i].strip().lower() == SIDE_CONTEXT[side]
            )
            if len(strikes) < 2:
                counts["side_trials_without_two_strikes"] += 1
                continue

            angles = np.stack(
                [points[0, point_index[label], :] for label in angle_labels], axis=1
            )
            for start_seconds, end_seconds in zip(strikes, strikes[1:]):
                duration = end_seconds - start_seconds
                if not min_cycle_seconds <= duration <= max_cycle_seconds:
                    counts["cycles_bad_duration"] += 1
                    continue

                start = int(round(start_seconds * rate)) - first_frame
                end = int(round(end_seconds * rate)) - first_frame
                if start < 0 or end >= angles.shape[0] or end <= start:
                    counts["cycles_outside_trial"] += 1
                    continue

                cycle = angles[start : end + 1]
                if not np.isfinite(cycle).all():
                    counts["cycles_nonfinite"] += 1
                    continue

                cycles.append(resample_cycle(cycle, n_points))
                subject_ids.append(path.parent.name)
                cycle_sides.append(side)
                source_files.append(str(path.relative_to(data_dir)))
                cycle_starts.append(start_seconds)
                cycle_ends.append(end_seconds)
                counts[f"{side}_cycles"] += 1
                file_produced_cycle = True

        if file_produced_cycle:
            counts["files_used"] += 1

    if not cycles:
        raise RuntimeError(f"No usable gait cycles found under {data_dir}")

    stacked = np.stack(cycles).astype(np.float32)  # (N, T, 3)
    dataset = {
        # The existing loader expects (T, 3, N).
        "final_jnt": np.transpose(stacked, (1, 2, 0)),
        "subject_ids": np.asarray(subject_ids),
        "sides": np.asarray(cycle_sides),
        "source_files": np.asarray(source_files),
        "cycle_start_seconds": np.asarray(cycle_starts, dtype=np.float32),
        "cycle_end_seconds": np.asarray(cycle_ends, dtype=np.float32),
        "channel_names": np.asarray(JOINTS),
    }
    return dataset, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--side", choices=("both", "left", "right"), default="both")
    parser.add_argument("--points", type=int, default=101)
    parser.add_argument("--min_cycle_seconds", type=float, default=0.5)
    parser.add_argument("--max_cycle_seconds", type=float, default=2.0)
    args = parser.parse_args()

    sides = {"both": ("L", "R"), "left": ("L",), "right": ("R",)}[args.side]
    dataset, counts = build_dataset(
        args.data_dir,
        sides,
        args.points,
        args.min_cycle_seconds,
        args.max_cycle_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **dataset)

    print(f"Saved {dataset['subject_ids'].size} cycles to {args.output}")
    print(f"Subjects: {np.unique(dataset['subject_ids']).size}")
    for key in sorted(counts):
        print(f"{key}: {counts[key]}")


if __name__ == "__main__":
    main()
