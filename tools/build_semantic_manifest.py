#!/usr/bin/env python3
"""Record validation scan order and point counts for cached-logit alignment."""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from datasets import Collate, SemanticKITTI


parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--batch_size", type=int, default=2)
parser.add_argument("--output", required=True)
args = parser.parse_args()
config = yaml.safe_load(open(args.config))
dataset = SemanticKITTI(
    rootdir=args.dataset, phase="val",
    input_feat=config["embedding"]["input_feat"],
    voxel_size=config["embedding"]["voxel_size"],
    num_neighbors_emb=config["embedding"]["neighbors"],
    bev_size=config["vit"]["bev_size"],
)
loader = DataLoader(
    dataset, batch_size=args.batch_size, shuffle=False, num_workers=2,
    collate_fn=Collate(sparse_batch_reindex=config["vit"].get("sparse_batch_reindex", False)),
)
manifest = []
for batch in loader:
    manifest.extend(zip(batch["filename"], batch["splits_pc"].tolist()))
Path(args.output).parent.mkdir(parents=True, exist_ok=True)
torch.save(manifest, args.output)
print(f"wrote {len(manifest)} scans to {args.output}")
