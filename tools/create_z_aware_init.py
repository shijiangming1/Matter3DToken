#!/usr/bin/env python3
"""Create a strict-loadable Z-Prototype initialization from a Matter3DToken checkpoint.

Only matching parameters are transferred.  The prototype assignment layer and
the 3D RoPE frequency buffer intentionally retain the Z-Prototype model's
initialization, so the resulting checkpoint can be loaded with train.py's
strict loader while preserving the baseline backbone.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matter3dtoken.model import Matter3DToken


def build_model(config):
    return Matter3DToken(
        emb_cin=config["embedding"]["cin"],
        emb_chidden=config["embedding"]["chidden"],
        emb_num_layers=config["embedding"]["num_layers"],
        vit_dim=config["vit"]["dim"],
        vit_depth=config["vit"]["depth"],
        vit_num_heads=config["vit"]["num_heads"],
        vit_expansion=config["vit"]["expansion"],
        merge_chidden=config["merge_head"]["chidden"],
        merge_cout=config["merge_head"]["cout"],
        nb_class=config["classif"]["nb_class"],
        drop_path=config["vit"]["drop_prob"],
        density_encoding=config["vit"].get("density_encoding", False),
        z_aware_prototypes=config["vit"].get("z_aware_prototypes"),
        budgeted_adaptive_router=config["vit"].get("budgeted_adaptive_router"),
        sparse_batch_reindex=config["vit"].get("sparse_batch_reindex", False),
    )


def load_model_config(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = build_model(load_model_config(args.config))
    target = model.state_dict()
    source = torch.load(args.source, map_location="cpu", weights_only=False)["net"]
    source = {key.removeprefix("module."): value for key, value in source.items()}

    transferred = []
    for name, value in source.items():
        if name in target and target[name].shape == value.shape:
            target[name] = value
            transferred.append(name)
    model.load_state_dict(target, strict=True)

    state = {"module." + name: value for name, value in model.state_dict().items()}
    torch.save({"net": state, "source": args.source, "transferred": transferred}, args.output)
    print(f"Transferred {len(transferred)}/{len(target)} tensors to {args.output}")
    skipped = sorted(set(source) - set(transferred))
    print("Skipped source tensors:", ", ".join(skipped))


if __name__ == "__main__":
    main()
