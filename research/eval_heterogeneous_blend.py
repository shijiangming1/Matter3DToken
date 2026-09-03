#!/usr/bin/env python3
"""Evaluate and blend SemanticKITTI models with different BEV collations.

Sparse-Batch and padded tokenizers cannot safely share a collated batch.  This
script evaluates one scan at a time through each model's native collator and
blends logits after each model has restored the original scan points.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from datasets import Collate, SemanticKITTI
from utils.metrics import fast_hist, per_class_iu
from matter3dtoken.model import Matter3DToken


def read_config(path):
    with open(path) as handle:
        return yaml.safe_load(handle)


def build_model(config, checkpoint, device):
    model = Matter3DToken(
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
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)["net"]
    if next(iter(state)).startswith("module."):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


def make_loader(config, dataset_root, batch_size, num_workers):
    dataset = SemanticKITTI(
        rootdir=dataset_root,
        phase="val",
        input_feat=config["embedding"]["input_feat"],
        voxel_size=config["embedding"]["voxel_size"],
        num_neighbors_emb=config["embedding"]["neighbors"],
        bev_size=config["vit"]["bev_size"],
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=Collate(sparse_batch_reindex=config["vit"].get("sparse_batch_reindex", False)),
    )


def forward(model, batch, device):
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
        logits = model(*inputs)
    return logits[batch["upsample"].to(device, non_blocking=True)]


def main(args):
    paths = [args.config_a, args.config_b, args.config_c]
    checkpoints = [args.checkpoint_a, args.checkpoint_b, args.checkpoint_c]
    configs = [read_config(path) for path in paths]
    raw_weight_sets = args.weight_sets.split(",") if args.weight_sets else [args.weights]
    weight_sets = []
    for raw_weights in raw_weight_sets:
        weights = [float(value) for value in raw_weights.split(":")]
        if len(weights) != 3 or sum(weights) <= 0:
            raise ValueError("Each weight set must contain three positive-sum values")
        weight_sets.append([weight / sum(weights) for weight in weights])
    temperature_sets = []
    raw_temperatures = (
        args.temperature_sets.split(",") if args.temperature_sets else ["1:1:1"]
    )
    for raw_temperature in raw_temperatures:
        temperatures = [float(value) for value in raw_temperature.split(":")]
        if len(temperatures) != 3 or any(value <= 0 for value in temperatures):
            raise ValueError("Each temperature set must contain three positive values")
        temperature_sets.append(temperatures)
    device = torch.device("cuda:0")
    models = [build_model(config, checkpoint, device) for config, checkpoint in zip(configs, checkpoints)]
    loaders = [
        make_loader(config, args.dataset, args.batch_size, args.num_workers)
        for config in configs
    ]
    num_classes = configs[0]["classif"]["nb_class"]
    combinations = [
        (weights, temperatures)
        for weights in weight_sets
        for temperatures in temperature_sets
    ]
    confusion = [torch.zeros((num_classes,) * 2, dtype=torch.int64) for _ in combinations]

    with torch.inference_mode():
        for batches in zip(*loaders):
            filenames = [tuple(batch["filename"]) for batch in batches]
            if len(set(filenames)) != 1:
                raise RuntimeError(f"Loader order mismatch: {filenames}")
            logits = [forward(model, batch, device) for model, batch in zip(models, batches)]
            labels = batches[0]["labels"].to(device, non_blocking=True)
            if any(not torch.equal(labels, batch["labels"].to(device)) for batch in batches[1:]):
                raise RuntimeError(f"Label mismatch for {filenames[0]}")
            valid = labels != 255
            for index, (weights, temperatures) in enumerate(combinations):
                prediction = sum(
                    weight * value / temperature
                    for weight, temperature, value in zip(weights, temperatures, logits)
                ).argmax(dim=1)
                confusion[index] += fast_hist(prediction[valid], labels[valid], num_classes).cpu()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as handle:
        for (weights, temperatures), hist in zip(combinations, confusion):
            iou = per_class_iu(hist.numpy())
            miou = 100.0 * np.nanmean(iou)
            label = ":".join(f"{weight:.6f}" for weight in weights)
            temperature_label = ":".join(f"{value:.4f}" for value in temperatures)
            handle.write(
                f"weights={label} temperatures={temperature_label} mIoU={miou:.6f}\n"
            )
            for index, value in enumerate(iou):
                handle.write(
                    f"weights={label} temperatures={temperature_label} "
                    f"class_{index}_iou={100.0 * value:.6f}\n"
                )
            print(f"weights={label} temperatures={temperature_label} mIoU={miou:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--config_a", required=True)
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--config_b", required=True)
    parser.add_argument("--checkpoint_b", required=True)
    parser.add_argument("--config_c", required=True)
    parser.add_argument("--checkpoint_c", required=True)
    parser.add_argument("--weights", default="0.333333:0.333333:0.333334")
    parser.add_argument("--weight_sets", default="")
    parser.add_argument("--temperature_sets", default="")
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
