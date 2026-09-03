#!/usr/bin/env python3
"""Cache one SemanticKITTI model's validation logits for offline fusion search."""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Collate, SemanticKITTI
from matter3dtoken.model import Matter3DToken


def build_model(config, checkpoint, device):
    v = config["vit"]
    model = Matter3DToken(
        config["embedding"]["cin"], config["embedding"]["chidden"],
        config["embedding"]["num_layers"], v["dim"], v["depth"],
        v["num_heads"], v["expansion"], config["merge_head"]["chidden"],
        config["merge_head"]["cout"], config["classif"]["nb_class"],
        v["drop_prob"], density_encoding=v.get("density_encoding", False),
        z_aware_prototypes=v.get("z_aware_prototypes"),
        budgeted_adaptive_router=v.get("budgeted_adaptive_router"),
        sparse_batch_reindex=v.get("sparse_batch_reindex", False),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)["net"]
    state = {k.removeprefix("module."): value for k, value in state.items()}
    model.load_state_dict(state, strict=True)
    return model.eval()


def main(args):
    config = yaml.safe_load(open(args.config))
    dataset = SemanticKITTI(
        rootdir=args.dataset, phase="val",
        input_feat=config["embedding"]["input_feat"],
        voxel_size=config["embedding"]["voxel_size"],
        num_neighbors_emb=config["embedding"]["neighbors"],
        bev_size=config["vit"]["bev_size"],
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
        collate_fn=Collate(sparse_batch_reindex=config["vit"].get("sparse_batch_reindex", False)),
    )
    device = torch.device("cuda:0")
    model = build_model(config, args.checkpoint, device)
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
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
            torch.save(
                {"filename": tuple(batch["filename"]),
                 "logits": logits.cpu().to(torch.float16),
                 "labels": batch["labels"].cpu()},
                output / f"{index:05d}.pt",
            )
            if index % 25 == 0:
                print(f"cached {index + 1}/{len(loader)}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    main(parser.parse_args())
