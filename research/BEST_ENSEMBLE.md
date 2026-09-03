# SemanticKITTI best validation ensemble

## Fixed result

- Reference validation mIoU: `69.302089%`
- Latest reproduction mIoU: `69.304446%`
- Baseline mIoU: `67.872429%`
- Absolute improvement: `+1.429660` percentage points
- Blend weights: `0.2625 / 0.2625 / 0.475`
- Test-time augmentation: disabled

The three logits are blended before `argmax` in this order:

1. Adaptive-v3 sparse fine-tuned checkpoint (`0.2625`)
2. Sparse-Batch checkpoint (`0.2625`)
3. Independently trained Adaptive-v3 + Sparse checkpoint (`0.475`)

## Reproduction

Run from the repository root:

```bash
bash eval_semantic_kitti_best_ensemble.sh
```

The script validates all required files before starting and writes a new
timestamped directory under:

```text
logs/research/semantic_kitti/Matter3DToken_B-best-ensemble/
```

Each run preserves the three config snapshots, environment metadata, command,
full log, and `results.txt`. Set `DATASET_ROOT`, `GPU_ID`, `PYTHON_BIN`, or
`LOG_ROOT` in the environment only when the corresponding local path differs.
The three checkpoint SHA-256 hashes are verified before evaluation by default;
set `VERIFY_CHECKPOINTS=0` only when intentionally using replacement weights.

The code-only backup does not duplicate multi-GB checkpoints. Run it against
the original artifact directory with:

```bash
cd /inspire/hdd/global_user/ky26289/malaai/Matter3DToken-693
CHECKPOINT_ROOT=/inspire/hdd/global_user/ky26289/malaai/Matter3DToken \
  bash eval_semantic_kitti_best_ensemble.sh
```

## Verification evidence

Peak search:

```text
0.2750 / 0.2750 / 0.4500 -> 69.300132%
0.2500 / 0.2500 / 0.5000 -> 69.300146%
```

Independent repeat with the midpoint:

```text
0.2750 / 0.2750 / 0.4500 -> 69.298130%
0.2625 / 0.2625 / 0.4750 -> 69.302089%
0.2500 / 0.2500 / 0.5000 -> 69.298470%
```

The fixed reproduction completed successfully on 2026-08-28 with
`69.304446%` mIoU. Its timestamped record is under
`logs/research/semantic_kitti/Matter3DToken_B-best-ensemble/` in the source checkout.

The selected midpoint lies on a broad plateau and avoids relying on a
numerically insignificant endpoint difference.

## Rejected additions

- Y-axis mirror TTA: `68.786624%`
- Fourth low-weight legacy model: stopped after the three-model result was fixed
- Adaptive route-only ensemble: approximately `68.305%`
