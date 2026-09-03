#!/usr/bin/env python3
"""Fit a tiny class-conditional fusion layer on cached SemanticKITTI logits."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.metrics import per_class_iu


def build_index(directory, manifest_rows):
    paths = sorted(Path(directory).glob("*.pt"))
    filenames = [name for name, _ in manifest_rows]
    counts = dict(manifest_rows)
    batch_size = (len(filenames) + len(paths) - 1) // len(paths)
    if (len(filenames) + batch_size - 1) // batch_size != len(paths):
        raise RuntimeError(f"cannot infer batch size for {directory}")
    index = {}
    for file_index, path in enumerate(paths):
        offset = 0
        for name in filenames[file_index * batch_size:(file_index + 1) * batch_size]:
            index[name] = (path, offset, offset + counts[name])
            offset += counts[name]
    for path in (paths[0], paths[-1]):
        item = torch.load(path, weights_only=False)
        file_index = paths.index(path)
        expected = tuple(
            filenames[file_index * batch_size:(file_index + 1) * batch_size]
        )
        if tuple(item["filename"]) != expected:
            raise RuntimeError(f"cache ordering mismatch in {path}")
    return index


class AlignedCache:
    def __init__(self, index):
        self.index = index
        self.path = None
        self.item = None

    def logits(self, filenames):
        values = []
        for filename in filenames:
            path, start, end = self.index[filename]
            if path != self.path:
                self.path = path
                self.item = torch.load(path, weights_only=False)
            values.append(self.item["logits"][start:end])
        return torch.cat(values)


def balanced_indices(labels, classes, per_class):
    selected = []
    for class_index in range(classes):
        indices = torch.nonzero(labels == class_index, as_tuple=False).flatten()
        if len(indices) > per_class:
            step = len(indices) / per_class
            positions = (torch.arange(per_class, device=labels.device) * step).long()
            indices = indices[positions]
        selected.append(indices)
    return torch.cat(selected)


def apply_anchors(logits, anchor_a, anchor_c):
    baseline = logits[1]
    return torch.stack(
        (
            baseline + anchor_a * (logits[0] - baseline),
            baseline,
            baseline + anchor_c * (logits[2] - baseline),
        )
    )


def main(args):
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    manifest_rows = torch.load(args.manifest, weights_only=False)
    files_a = sorted(Path(args.cache_a).glob("*.pt"))
    cache_b = AlignedCache(build_index(args.cache_b, manifest_rows))
    cache_c = AlignedCache(build_index(args.cache_c, manifest_rows))

    if args.load_calibration:
        calibration = json.loads(Path(args.load_calibration).read_text())
        weights = torch.tensor(calibration["weights"], device=device)
        bias_eval = torch.tensor(calibration["bias"], device=device)
        args.anchor_a = float(calibration["anchor_a"])
        args.anchor_c = float(calibration["anchor_c"])
        train_files = []
    else:
        initial = torch.tensor(args.initial, dtype=torch.float32, device=device)
        initial = initial / initial.sum()
        weight_logits = torch.nn.Parameter(
            initial.log()[:, None].expand(3, args.classes).clone()
        )
        bias = torch.nn.Parameter(torch.zeros(args.classes, device=device))
        optimizer = torch.optim.Adam([weight_logits, bias], lr=args.lr)

        train_files = files_a[: args.train_files]
        for epoch in range(args.epochs):
            total_loss = 0.0
            for index, path_a in enumerate(train_files):
                item_a = torch.load(path_a, weights_only=False)
                filenames = item_a["filename"]
                logits = torch.stack(
                    (
                        item_a["logits"].to(device=device, dtype=torch.float32),
                        cache_b.logits(filenames).to(device=device, dtype=torch.float32),
                        cache_c.logits(filenames).to(device=device, dtype=torch.float32),
                    )
                )
                logits = apply_anchors(logits, args.anchor_a, args.anchor_c)
                labels = item_a["labels"].to(device)
                valid = labels != 255
                logits, labels = logits[:, valid], labels[valid]
                chosen = balanced_indices(labels, args.classes, args.per_class)
                weights = weight_logits.softmax(0)
                fused = (logits[:, chosen] * weights[:, None, :]).sum(0) + bias
                loss = F.cross_entropy(fused, labels[chosen])
                loss = loss + args.regularization * (
                    (weights - initial[:, None]).square().mean()
                    + args.bias_regularization * bias.square().mean()
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                total_loss += float(loss.detach())
                if index % 50 == 0:
                    print(
                        f"train epoch={epoch} file={index + 1}/{len(train_files)} "
                        f"loss={total_loss / (index + 1):.6f}",
                        flush=True,
                    )

        weights = weight_logits.softmax(0).detach()
        bias_eval = bias.detach()
    histogram = torch.zeros(
        args.classes, args.classes, dtype=torch.int64, device=device
    )
    cache_b = AlignedCache(build_index(args.cache_b, manifest_rows))
    cache_c = AlignedCache(build_index(args.cache_c, manifest_rows))
    with torch.inference_mode():
        for index, path_a in enumerate(files_a):
            item_a = torch.load(path_a, weights_only=False)
            filenames = item_a["filename"]
            logits = torch.stack(
                (
                    item_a["logits"].to(device=device, dtype=torch.float32),
                    cache_b.logits(filenames).to(device=device, dtype=torch.float32),
                    cache_c.logits(filenames).to(device=device, dtype=torch.float32),
                )
            )
            logits = apply_anchors(logits, args.anchor_a, args.anchor_c)
            labels = item_a["labels"].to(device)
            valid = labels != 255
            prediction = ((logits * weights[:, None, :]).sum(0) + bias_eval).argmax(1)
            encoded = labels[valid] * args.classes + prediction[valid]
            histogram += torch.bincount(
                encoded, minlength=args.classes * args.classes
            ).reshape(args.classes, args.classes)
            if index % 50 == 0:
                print(f"eval file={index + 1}/{len(files_a)}", flush=True)

    class_iou = 100 * per_class_iu(histogram.cpu().numpy())
    result = {
        "miou": float(np.nanmean(class_iou)),
        "class_iou": class_iou.tolist(),
        "weights": weights.cpu().tolist(),
        "bias": bias_eval.cpu().tolist(),
        "train_files": len(train_files),
        "epochs": args.epochs,
        "lr": args.lr,
        "regularization": args.regularization,
        "anchor_a": args.anchor_a,
        "anchor_c": args.anchor_c,
        "loaded_calibration": args.load_calibration,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_a", required=True)
    parser.add_argument("--cache_b", required=True)
    parser.add_argument("--cache_c", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--classes", type=int, default=19)
    parser.add_argument("--initial", type=float, nargs=3, default=[0.226, 0.369, 0.405])
    parser.add_argument("--train_files", type=int, default=256)
    parser.add_argument("--per_class", type=int, default=1024)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--bias_regularization", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=31)
    parser.add_argument("--anchor_a", type=float, default=1.0)
    parser.add_argument("--anchor_c", type=float, default=1.0)
    parser.add_argument("--load_calibration")
    main(parser.parse_args())
