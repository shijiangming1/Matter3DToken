import argparse
import os

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from datasets import Collate, SemanticKITTI
from utils.metrics import fast_hist, per_class_iu
from matter3dtoken.model import Matter3DToken


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
        z_aware_prototypes=config["vit"].get("z_aware_prototypes", None),
        budgeted_adaptive_router=config["vit"].get("budgeted_adaptive_router", None),
        sparse_batch_reindex=config["vit"].get("sparse_batch_reindex", False),
    ).to(device)
    state = torch.load(checkpoint, map_location=device, weights_only=False)["net"]
    if state and next(iter(state)).startswith("module."):
        state = {key[len("module.") :]: value for key, value in state.items()}
    model.load_state_dict(state)
    model.eval()
    return model


def main(args):
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    with open(args.config_b or args.config) as stream:
        config_b = yaml.safe_load(stream)
    device = torch.device("cuda:0")
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
        batch_size=config["dataloader"]["batch_size"],
        shuffle=False,
        num_workers=config["dataloader"]["num_workers"],
        pin_memory=True,
        collate_fn=Collate(sparse_batch_reindex=config["vit"].get("sparse_batch_reindex", False)),
    )
    model_a = build_model(config, args.checkpoint_a, device)
    model_b = build_model(config_b, args.checkpoint_b, device)
    model_c = None
    model_d = None
    weights = None
    if args.checkpoint_c:
        with open(args.config_c or args.config) as stream:
            config_c = yaml.safe_load(stream)
        model_c = build_model(config_c, args.checkpoint_c, device)
        if args.checkpoint_d:
            with open(args.config_d or args.config) as stream:
                config_d = yaml.safe_load(stream)
            model_d = build_model(config_d, args.checkpoint_d, device)
        weights = [
            tuple(float(value) for value in item.split(":"))
            for item in args.weights.split(",")
        ]
        expected_weights = 4 if model_d is not None else 3
        if any(len(item) != expected_weights for item in weights):
            raise ValueError(
                f"Each --weights item must contain {expected_weights} values"
            )
    else:
        ratios = [float(value) for value in args.ratios.split(",")]
        weights = [(ratio, 1.0 - ratio) for ratio in ratios]
    confusion = [torch.zeros((config["classif"]["nb_class"],) * 2, dtype=torch.int64) for _ in weights]

    with torch.inference_mode():
        for batch in loader:
            inputs = (
                batch["feat"].to(device, non_blocking=True),
                batch["xyz"].to(device, non_blocking=True),
                batch["neighbors"].to(device, non_blocking=True),
                batch["xy_bev"].to(device, non_blocking=True),
                batch["idx_bev"].to(device, non_blocking=True),
                batch["idx_occ"].to(device, non_blocking=True),
                None,
                batch["splits_bev"].to(device, non_blocking=True),
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits_a = model_a(*inputs)
                logits_b = model_b(*inputs)
                logits_c = model_c(*inputs) if model_c is not None else None
                logits_d = model_d(*inputs) if model_d is not None else None
                if args.flip_y:
                    flipped = list(inputs)
                    flipped[0] = inputs[0].clone()
                    flipped[0][:, 2].neg_()
                    flipped[1] = inputs[1].clone()
                    flipped[1][:, 1].neg_()
                    flipped[3] = inputs[3].clone()
                    flipped[3][:, 1].neg_()
                    logits_a = 0.5 * (logits_a + model_a(*flipped))
                    logits_b = 0.5 * (logits_b + model_b(*flipped))
                    if model_c is not None:
                        logits_c = 0.5 * (logits_c + model_c(*flipped))
                    if model_d is not None:
                        logits_d = 0.5 * (logits_d + model_d(*flipped))
            labels = batch["labels"].to(device, non_blocking=True)
            upsample = batch["upsample"].to(device, non_blocking=True)
            for i, blend_weights in enumerate(weights):
                logits = blend_weights[0] * logits_a + blend_weights[1] * logits_b
                if logits_c is not None:
                    logits = logits + blend_weights[2] * logits_c
                if logits_d is not None:
                    logits = logits + blend_weights[3] * logits_d
                pred = logits[upsample].argmax(dim=1)
                valid = labels != 255
                confusion[i] += fast_hist(pred[valid], labels[valid], logits.shape[1]).cpu()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w") as stream:
        for blend_weights, hist in zip(weights, confusion):
            iou = per_class_iu(hist.numpy())
            miou = 100.0 * np.nanmean(iou)
            label = ":".join(f"{value:.3f}" for value in blend_weights)
            stream.write(f"weights={label} mIoU={miou:.6f}\n")
            print(f"weights={label} mIoU={miou:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--config_b", default="")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--checkpoint_a", required=True)
    parser.add_argument("--checkpoint_b", required=True)
    parser.add_argument("--checkpoint_c", default="")
    parser.add_argument("--config_c", default="")
    parser.add_argument("--checkpoint_d", default="")
    parser.add_argument("--config_d", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--ratios", default="0,0.25,0.5,0.75,1")
    parser.add_argument("--weights", default="0.25:0.25:0.5")
    parser.add_argument("--flip_y", action="store_true")
    main(parser.parse_args())
