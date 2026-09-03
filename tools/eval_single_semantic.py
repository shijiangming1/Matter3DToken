#!/usr/bin/env python3
"""Evaluate one SemanticKITTI checkpoint with an optional scalar override."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Collate, SemanticKITTI
from research.cache_semantic_logits import build_model
from utils.metrics import fast_hist, per_class_iu


def main(args):
    config = yaml.safe_load(open(args.config))
    dataset = SemanticKITTI(
        rootdir=args.dataset,
        phase="val",
        input_feat=config["embedding"]["input_feat"],
        voxel_size=config["embedding"]["voxel_size"],
        num_neighbors_emb=config["embedding"]["neighbors"],
        bev_size=config["vit"]["bev_size"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=Collate(sparse_batch_reindex=config["vit"].get("sparse_batch_reindex", False)),
    )
    device = torch.device(args.device)
    model = build_model(config, args.checkpoint, device)
    if args.parameter:
        parameters = dict(model.named_parameters())
        if args.parameter not in parameters:
            raise KeyError(f"unknown parameter: {args.parameter}")
        parameter = parameters[args.parameter]
        if parameter.numel() != 1:
            raise ValueError("only scalar parameters can be overridden")
        parameter.data.fill_(args.value)

    hist = np.zeros((args.classes, args.classes), dtype=np.int64)
    with torch.inference_mode():
        for index, batch in enumerate(loader):
            idx_pad = batch["idx_pad"]
            inputs = (
                batch["feat"].to(device, non_blocking=True),
                batch["xyz"].to(device, non_blocking=True),
                batch["neighbors"].to(device, non_blocking=True),
                batch["xy_bev"].to(device, non_blocking=True),
                batch["idx_bev"].to(device, non_blocking=True),
                batch["idx_occ"].to(device, non_blocking=True),
                None if idx_pad is None else idx_pad.to(device, non_blocking=True),
                batch["splits_bev"].to(device, non_blocking=True),
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(*inputs)[batch["upsample"].to(device)]
            prediction = logits.argmax(1).cpu()
            labels = batch["labels"]
            valid = labels != 255
            hist += fast_hist(
                prediction[valid], labels[valid], args.classes
            ).numpy()
            if index % 50 == 0:
                print(f"evaluated {index + 1}/{len(loader)}", flush=True)

    class_iou = 100 * per_class_iu(hist)
    result = {
        "checkpoint": args.checkpoint,
        "parameter": args.parameter,
        "value": args.value if args.parameter else None,
        "miou": float(np.nanmean(class_iou)),
        "class_iou": class_iou.tolist(),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--parameter")
    parser.add_argument("--value", type=float, default=0.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--classes", type=int, default=19)
    main(parser.parse_args())
