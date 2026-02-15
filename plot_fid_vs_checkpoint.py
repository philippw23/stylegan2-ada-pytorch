#!/usr/bin/env python3
"""Plot FID (`fid50k_full`) against training checkpoint from a JSONL log."""

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt


SNAP_RE = re.compile(r"network-snapshot-(\d+)\.pkl$")


def parse_jsonl(path: Path):
    """Extract `(checkpoint_step, fid50k_full)` pairs from a metrics JSONL file.

    Only entries with a `snapshot_pkl` matching `network-snapshot-<step>.pkl`
    and a numeric `results.fid50k_full` value are returned.
    """
    points = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}: invalid JSON on line {line_no}") from exc
            snap = obj.get("snapshot_pkl", "")
            match = SNAP_RE.search(snap)
            if not match:
                continue
            step = int(match.group(1))
            fid = obj.get("results", {}).get("fid50k_full")
            if fid is None:
                continue
            points.append((step, float(fid)))
    return sorted(points, key=lambda x: x[0])


def main():
    """Parse CLI arguments, build the FID-vs-checkpoint plot, and save it."""
    parser = argparse.ArgumentParser(
        description="Plot FID (fid50k_full) vs checkpoint from a metric JSONL file."
    )
    parser.add_argument(
        "jsonl",
        type=Path,
        help="Path to metric-fid50k_full.jsonl",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output image path (default: <jsonl stem>_fid_vs_checkpoint.png)",
    )
    parser.add_argument(
        "--title",
        default="FID vs Checkpoint",
        help="Plot title",
    )
    args = parser.parse_args()

    points = parse_jsonl(args.jsonl)
    if not points:
        raise SystemExit("No valid FID points found in JSONL.")

    steps, fids = zip(*points)

    out_path = args.out
    if out_path is None:
        out_path = args.jsonl.with_name(args.jsonl.stem + "_fid_vs_checkpoint.png")

    plt.figure(figsize=(10, 5))
    plt.plot(steps, fids, marker="o", linewidth=1.5, markersize=3)
    plt.xlabel("Checkpoint")
    plt.ylabel("FID (fid50k_full)")
    plt.title(args.title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
