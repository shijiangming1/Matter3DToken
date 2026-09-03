# Matter3DToken

Tokenizing What Matters for Transformer-Based Point Cloud Semantic Segmentation

Matter3DToken treats 3D tokenization as representation-resource allocation under an explicit token budget. LiDAR scenes are highly non-uniform: broad road surfaces are geometrically simple, while boundaries, sparse objects, occlusions, and vertically overlapping structures carry disproportionately dense semantic evidence. Instead of distributing the same token capacity everywhere or replacing the backbone with specialized attention, Matter3DToken concentrates 3D adaptivity at the tokenizer interface and retains a standard, non-hierarchical global Transformer.

The implementation accompanies the paper draft in [`paper/Matter3DToken_Chinese_Draft4.docx`](paper/Matter3DToken_Chinese_Draft4.docx).

## Method

Matter3DToken combines three complementary components before a shared Transformer:

1. **Z-aware Prototype Tokenizer** (`ZAwarePrototypeTokenizer`) detects vertical ambiguity inside a BEV pillar and allocates a small number of additional height-aware prototypes only where needed. Each prototype carries a learned feature and a representative 3D coordinate.
2. **Budgeted Adaptive Router with Detail Relay** (`BudgetedAdaptiveRouter`) chooses, under an explicit token budget, whether a region should retain fine pillar tokens or share coarse context. Detail Relay bypasses global attention and restores signed local residuals during point-level decoding.
3. **Sparse-Batch Re-index** (`sparse_batch_reindex`) concatenates only valid variable-length tokens and uses scene offsets to preserve sample isolation. Runtime therefore follows the number of effective tokens rather than the longest padded sequence.

The Z-aware and budgeted-routing token sets are concatenated within each scene, processed by one shared Transformer, and mapped back to points through exact saved routes. A learnable gate combines their point-level contexts. No KNN upsampling is used.

## Code map

| Paper component | Public code name | Implementation |
|---|---|---|
| Full model | `Matter3DToken` | `matter3dtoken/model.py` |
| Z-aware Prototype Tokenizer | `ZAwarePrototypeTokenizer` | `matter3dtoken/z_aware_prototype_tokenizer.py` |
| Budgeted Adaptive Router + Detail Relay | `BudgetedAdaptiveRouter` | `matter3dtoken/budgeted_adaptive_router.py` |
| Sparse-Batch Re-index | `sparse_batch_reindex` | `matter3dtoken/model.py`, `matter3dtoken/transformer.py`, `datasets/pc_dataset.py` |
| Shared standard Transformer | `Transformer` | `matter3dtoken/transformer.py` |

Evaluation and calibration utilities are collected under `tools/`; executable
checks are under `tests/`. Historical controller logs and internal research
records are intentionally excluded from the release.

The public configuration keys use the same terminology:

```yaml
vit:
  z_aware_prototypes:
    enabled: true
  budgeted_adaptive_router:
    enabled: true
  sparse_batch_reindex: true
```

## Installation

The tested software family is Python 3.11, PyTorch 2.5, CUDA 12.x, and BF16-capable NVIDIA GPUs.

```bash
conda create -n matter3dtoken python=3.11 -y
conda activate matter3dtoken
pip install -r requirements.txt
```

Install the CUDA-enabled PyTorch build appropriate for the local driver before running training.

## SemanticKITTI

Expected layout:

```text
<DATASET_ROOT>/dataset/sequences/
├── 00/
├── ...
└── 21/
```

The default training configuration is:

```text
configs/semantic_kitti/Matter3DToken_joint_abc_h200_20e.yaml
```

It jointly enables all three proposed components and trains for 20 epochs.

## Full model training

```bash
cd /inspire/hdd/global_user/ky26289/malaai/Matter3DToken

CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATASET_ROOT=/path/to/semantic_kitti \
./train_matter3dtoken.sh
```

Every run receives a timestamped directory containing the exact configuration, environment information, package list, source diff, console log, `ckpt_last.pth`, and `ckpt_best.pth`.

To perform a quick infrastructure check without the one-time InstanceCutMix extraction:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATASET_ROOT=/path/to/semantic_kitti \
./train_matter3dtoken_without_instance_cutmix.sh
```

This changes the augmentation setting and is intended for debugging, not for the paper's final comparison.

## Component ablations

The principal single-component configurations are:

```text
A: configs/semantic_kitti/Matter3DToken_B-z-aware-prototype-v3-transfer-20.yaml
B: configs/semantic_kitti/Matter3DToken_B-sparse-batch-reindex-20.yaml
C: configs/semantic_kitti/Matter3DToken_B-budgeted-router-nearfull-semantic-20.yaml
```

Train one configuration with:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
DATASET_ROOT=/path/to/semantic_kitti \
CONFIG=configs/semantic_kitti/Matter3DToken_B-sparse-batch-reindex-20.yaml \
LOG_PATH=logs/ablation/sparse_batch_reindex \
SEED=31 \
./train_semantickitti_component.sh
```

Run the complete three-component ablation sequence with:

```bash
DATASET_ROOT=/path/to/semantic_kitti \
GPU_LIST=0,1,2,3 \
SEED=31 \
./train_component_ablation_4gpu.sh
```

## Recorded validation evidence

The source experiment archive reported the following SemanticKITTI validation results:

| Output | Validation mIoU |
|---|---:|
| Z-aware output anchored to Sparse-Batch | 68.513314% |
| Sparse-Batch Re-index | 68.526227% |
| Budgeted Adaptive Router output anchored to Sparse-Batch | 68.740377% |
| Classwise calibrated three-output fusion | 69.759820% |

The 69.759820% number is a calibrated fusion of three saved outputs, not the score of a single jointly trained checkpoint. It must not be reported as the end-to-end joint model result. Large checkpoints, cached logits, and training logs are intentionally excluded from this lightweight code release.

## Naming and provenance

This directory is a paper-facing refactor of the working research repository. Model, module, package, configuration, and launcher names have been aligned with the Matter3DToken paper. The tensor operations, token construction, routing equations, attention implementation, losses, and optimization flow are unchanged by this refactor.

The code retains the upstream license and notices in `LICENSE` and `NOTICE`.
