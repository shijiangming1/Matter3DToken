#!/usr/bin/env python3
"""Generate a checkpoint-backed progress table for the A/B/C study."""

import argparse
import re
from pathlib import Path

import torch


def checkpoint_row(name: str, path: Path) -> str:
    if not path.is_file():
        return f"| {name} | pending | - | {path} |"
    state = torch.load(path, map_location="cpu", weights_only=False)
    return (
        f"| {name} | {float(state['best_miou']):.6f} | "
        f"{state.get('epoch', 'unknown')} | {path} |"
    )


def fusion_row(path: Path) -> str:
    if not path.is_file():
        return f"| ABC fusion | pending | - | {path} |"
    records = []
    for line in path.read_text().splitlines():
        match = re.match(r"^weights=(\S+) mIoU=([0-9.]+)$", line)
        if match:
            records.append((float(match.group(2)), match.group(1)))
    if not records:
        return f"| ABC fusion | no valid metric | - | {path} |"
    score, weights = max(records)
    return f"| ABC fusion | {score:.6f} | weights {weights} | {path} |"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--b", type=Path, required=True)
    parser.add_argument("--c", type=Path, required=True)
    parser.add_argument("--fusion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    text = "\n".join([
        "# SemanticKITTI A/B/C Progress",
        "",
        "| Candidate | best mIoU (%) | best epoch | Artifact |",
        "| --- | ---: | --- | --- |",
        checkpoint_row("A Z-Prototype", args.a),
        checkpoint_row("B Sparse-Batch", args.b),
        checkpoint_row("C Adaptive", args.c),
        fusion_row(args.fusion),
        "",
    ])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
