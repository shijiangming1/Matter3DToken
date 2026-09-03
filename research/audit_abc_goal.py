#!/usr/bin/env python3
"""Audit the strict A/B/C and fusion mIoU targets from saved artifacts."""

import argparse
import re
from pathlib import Path

import torch


MODULE_TARGET = 68.5
FUSION_TARGET = 69.5


def read_checkpoint(path):
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "best_miou" not in state:
        raise KeyError(f"{path} does not contain best_miou")
    return float(state["best_miou"]), state.get("epoch", "unknown")


def read_fusion(path):
    pattern = re.compile(
        r"^weights=(\S+)(?: temperatures=\S+)? mIoU=([0-9.]+)$"
    )
    candidates = []
    for line in Path(path).read_text().splitlines():
        match = pattern.match(line)
        if match:
            candidates.append((float(match.group(2)), match.group(1)))
    if not candidates:
        raise ValueError(f"No fusion mIoU rows found in {path}")
    return max(candidates)


def read_full_training_log(path):
    """Verify that a 20-epoch candidate completed without a runtime failure."""
    text = Path(path).read_text(errors="replace")
    return (
        "Training: 19/20 epochs" in text
        and "Validation: 19/20 epochs" in text
        and "Finished Training" in text
        and "Traceback" not in text
        and "RuntimeError" not in text
    )


def main(args):
    modules = {
        "A_z_aware_prototype": (args.checkpoint_a, args.log_a),
        "B_sparse_batch_reindex": (args.checkpoint_b, args.log_b),
        "C_budgeted_router": (args.checkpoint_c, args.log_c),
    }
    rows = []
    passed = True
    for name, (checkpoint, log_path) in modules.items():
        miou, epoch = read_checkpoint(checkpoint)
        complete = read_full_training_log(log_path)
        ok = miou > MODULE_TARGET and complete
        evidence = f"best epoch {epoch}; 20 epochs {'complete' if complete else 'incomplete'}"
        rows.append((name, miou, evidence, ok))
        passed &= ok

    fusion_miou, weights = read_fusion(args.fusion_metrics)
    fusion_ok = fusion_miou > FUSION_TARGET
    rows.append(("ABC_fusion", fusion_miou, f"weights {weights}", fusion_ok))
    passed &= fusion_ok

    report = [
        "# SemanticKITTI A/B/C Goal Audit",
        "",
        f"Module threshold: strictly > {MODULE_TARGET:.1f}%",
        f"Fusion threshold: strictly > {FUSION_TARGET:.1f}%",
        "",
        "| Candidate | mIoU (%) | Source | Status |",
        "| --- | ---: | --- | --- |",
    ]
    for name, miou, source, ok in rows:
        report.append(
            f"| {name} | {miou:.6f} | {source} | {'PASS' if ok else 'FAIL'} |"
        )
    report.extend(["", f"Overall: {'GOAL ACHIEVED' if passed else 'NOT YET ACHIEVED'}"])
    text = "\n".join(report) + "\n"
    if args.output:
        Path(args.output).write_text(text)
    print(text, end="")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--checkpoint_b", required=True)
    parser.add_argument("--checkpoint_c", required=True)
    parser.add_argument("--log_a", required=True)
    parser.add_argument("--log_b", required=True)
    parser.add_argument("--log_c", required=True)
    parser.add_argument("--fusion_metrics", required=True)
    parser.add_argument("--output")
    main(parser.parse_args())
