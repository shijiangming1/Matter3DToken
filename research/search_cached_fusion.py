#!/usr/bin/env python3
"""Search weights and temperatures from cached validation logits."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.metrics import fast_hist, per_class_iu


def parse_sets(raw, expected, positive=True, raw_weights=False):
    result = []
    for item in raw.split(","):
        values = [float(value) for value in item.split(":")]
        invalid = len(values) != 3
        if expected == "weights":
            if raw_weights:
                invalid = invalid or all(value == 0 for value in values)
            else:
                invalid = invalid or any(value < 0 for value in values) or sum(values) <= 0
        elif positive:
            invalid = invalid or any(value <= 0 for value in values)
        if invalid:
            raise ValueError(f"invalid set: {item}")
        if expected == "weights":
            if not raw_weights:
                total = sum(values)
                values = [value / total for value in values]
        result.append(values)
    return result


def main(args):
    device = torch.device(args.device)
    weights = parse_sets(args.weight_sets, "weights", raw_weights=args.raw_weights)
    temperatures = parse_sets(args.temperature_sets, "temperatures")
    combinations = [(w, t) for w in weights for t in temperatures]
    coefficients = torch.tensor(
        [[w_i / t_i for w_i, t_i in zip(weight, temperature)]
         for weight, temperature in combinations], dtype=torch.float32,
        device=device,
    )
    histograms = torch.zeros(
        (len(combinations), args.classes, args.classes), dtype=torch.int64,
        device=device,
    )
    manifest_rows = torch.load(args.manifest, weights_only=False)
    manifest = dict(manifest_rows)
    manifest_filenames = [filename for filename, _ in manifest_rows]
    files = sorted(Path(args.cache_a).glob("*.pt"))
    def make_index(directory):
        paths = sorted(Path(directory).glob("*.pt"))
        if not paths:
            raise RuntimeError(f"no cache files in {directory}")
        batch_size = (len(manifest_rows) + len(paths) - 1) // len(paths)
        expected_files = (len(manifest_rows) + batch_size - 1) // batch_size
        if expected_files != len(paths):
            raise RuntimeError(f"cannot infer cache batch size for {directory}")
        result = {}
        for file_index, path in enumerate(paths):
            offset = 0
            start = file_index * batch_size
            filenames = manifest_filenames[start:start + batch_size]
            for filename in filenames:
                count = manifest[filename]
                result[filename] = (path, offset, offset + count)
                offset += count
        for path in (paths[0], paths[-1]):
            item = torch.load(path, weights_only=False)
            file_index = paths.index(path)
            start = file_index * batch_size
            expected = tuple(manifest_filenames[start:start + batch_size])
            if tuple(item["filename"]) != expected:
                raise RuntimeError(f"cache ordering mismatch in {path}")
            expected_points = sum(manifest[name] for name in expected)
            if len(item["logits"]) != expected_points:
                raise RuntimeError(f"point count mismatch in {path}")
        return result

    index_b = make_index(args.cache_b)
    index_c = make_index(args.cache_c)
    # Keep only the most recently used cache file.  A batch-size mismatch can
    # make one source file serve two A batches, so this small cache still
    # avoids duplicate reads without retaining tens of GB in RAM.
    loaded = {}

    def aligned_logits(index, filenames):
        state = loaded.setdefault(id(index), {"path": None, "item": None})
        values = []
        for filename in filenames:
            path, start, end = index[filename]
            if path != state["path"]:
                state["path"] = path
                state["item"] = torch.load(path, weights_only=False)
            values.append(state["item"]["logits"][start:end])
        return torch.cat(values, dim=0)

    for index, pa in enumerate(files):
        a = torch.load(pa, weights_only=False)
        filenames = a["filename"]
        b = {"logits": aligned_logits(index_b, filenames)}
        c = {"logits": aligned_logits(index_c, filenames)}
        labels = a["labels"].to(device)
        valid = labels != 255
        logits = [a["logits"].to(device=device, dtype=torch.float32),
                  b["logits"].to(device=device, dtype=torch.float32),
                  c["logits"].to(device=device, dtype=torch.float32)]
        stacked_logits = torch.stack(logits)
        labels_valid = labels[valid].long()
        chunk_size = args.candidate_chunk or len(combinations)
        for start in range(0, len(combinations), chunk_size):
            end = min(start + chunk_size, len(combinations))
            predictions = torch.einsum(
                "kt,tnc->knc", coefficients[start:end], stacked_logits
            ).argmax(2)
            combo_ids = torch.arange(end - start, device=device).view(-1, 1)
            encoded = combo_ids * args.classes * args.classes
            encoded = encoded + labels_valid.view(1, -1) * args.classes
            encoded = encoded + predictions[:, valid]
            histograms[start:end] += torch.bincount(
                encoded.reshape(-1),
                minlength=(end - start) * args.classes * args.classes,
            ).reshape(end - start, args.classes, args.classes)
        if index % 50 == 0:
            print(f"scored {index + 1}/{len(files)}", flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    with output.open("w") as handle:
        for (weight, temperature), hist in zip(combinations, histograms):
            miou = 100 * np.nanmean(per_class_iu(hist.cpu().numpy()))
            w = ":".join(f"{v:.6f}" for v in weight)
            t = ":".join(f"{v:.4f}" for v in temperature)
            handle.write(f"weights={w} temperatures={t} mIoU={miou:.6f}\n")
            rows.append((miou, w, t))
    print("BEST", max(rows))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache_a", required=True)
    parser.add_argument("--cache_b", required=True)
    parser.add_argument("--cache_c", required=True)
    parser.add_argument("--weight_sets", required=True)
    parser.add_argument("--temperature_sets", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--classes", type=int, default=19)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--candidate_chunk", type=int, default=0)
    parser.add_argument("--raw_weights", action="store_true")
    parser.add_argument("--output", required=True)
    main(parser.parse_args())
