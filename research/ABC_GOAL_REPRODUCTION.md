# SemanticKITTI A/B/C Strict Goal Reproduction

The acceptance criteria are strict: A, B, and C must each have a saved
validation `best_miou > 68.5`, and the best saved heterogeneous A/B/C blend
must be `> 69.5`. `research/audit_abc_goal.py` is the authoritative verifier;
it reads the checkpoints and the fusion metric file directly.

## Stable Baseline Artifact

| Candidate | Checkpoint | Validation best mIoU | Status |
| --- | --- | ---: | --- |
| B Sparse-Batch | `logs/retrain/semantic_kitti/B-sparse/run_20260828_204549_3971091/ckpt_best.pth` | 68.521816 | Passes the single-module threshold |

No A, C, or fusion result should be described as passing until its own
`goal_audit.md` says `PASS`.

## Reproduction Command

Run from the repository root after no other `train.py` process is using the
fixed local rendezvous endpoint `tcp://127.0.0.1:4444`:

```bash
DATASET_ROOT=/inspire/hdd/project/agentend2end/ky26289/malaai/Datasets/semantic_kitti \
GPU_LIST=0,1,2,3 SEED=31 bash run_component_ablation_pipeline.sh
```

The command creates a unique `logs/research/semantic_kitti/ABC_pipeline_*`
directory and writes the following immutable evidence:

| Evidence | Location inside the run directory |
| --- | --- |
| A Z-Prototype config, initialization, console output, best checkpoint | `A_z_aware_prototype/` |
| C budgeted-router config, initialization, console output, best checkpoint | `C_budgeted_router/` |
| Blended validation metrics and console output | `fusion/metrics.txt`, `fusion/eval.log` |
| Strict threshold result | `goal_audit.md` |

`run_metadata.txt` records the source configuration, initialization checkpoint,
seed, and GPU list for each train call. Both A and C configs set
`scheduler.max_epoch: 20`; the launcher does not implement early stopping.

The A configuration enables `sparse_batch_reindex: true`. Its Z-prototype tokenizer
packs each scan's variable-length token sequence and supplies offsets to the
standard transformer, so no padding token changes the B-initialized attention
normalization. Run `python research/test_z_aware_packed.py` to verify the
two-scan padded/packed route equivalence and a packed Transformer forward.

## Active Retry Chain

The active timestamp-isolated chain uses:

| Role | Directory |
| --- | --- |
| A retry | `logs/research/semantic_kitti/20260830_000000_Zproto_transfer_retry` |
| Main C retry | `logs/research/semantic_kitti/20260830_020100_Adaptive_nearfull_retry` |
| C fallback retry | `logs/research/semantic_kitti/20260830_040100_Adaptive_ultranearfull_retry` |
| Conservative C retry | `logs/research/semantic_kitti/20260830_060100_Adaptive_conservative_retry` |
| Main fusion | `logs/research/semantic_kitti/20260830_030100_ABC_nearfull_retry_fusion` |
| Fallback fusion | `logs/research/semantic_kitti/20260830_050100_ABC_ultranearfull_retry_fusion` |
| Conservative fusion | `logs/research/semantic_kitti/20260830_070100_ABC_conservative_retry_fusion` |

`research/wait_and_train_z_aware_prototype_retry.sh` serializes A after the currently
active older training run. `research/wait_z_retry_then_abc_audit.sh` then
trains C, evaluates fusion, and runs the strict audit; it invokes the fallback
C only if the preceding audit fails. The final conservative C keeps 99.9% of
the fine-token budget while retaining active budgeted-router partitioning.
