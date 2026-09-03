#!/usr/bin/env bash
# Full A/B/C retraining on four GPUs.
# Order: B -> A/C -> heterogeneous fusion evaluation.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${ROOT}"

DATASET_ROOT="${DATASET_ROOT:-/inspire/hdd/project/agentend2end/ky26289/malaai/Datasets/semantic_kitti}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
SEED="${SEED:-31}"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"

B_RUN="${ROOT}/logs/retrain/semantic_kitti/B-sparse-4gpu/run_${RUN_ID}"
ABC_ROOT="${ROOT}/logs/research/semantic_kitti/ABC_full_4gpu_${RUN_ID}"

if [[ ! -d "${DATASET_ROOT}/dataset/sequences" ]]; then
    echo "Invalid SemanticKITTI dataset path: ${DATASET_ROOT}" >&2
    exit 1
fi

mkdir -p "${B_RUN}" "${ABC_ROOT}"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

echo "[$(date -u +%FT%TZ)] Start full A/B/C retraining"
echo "dataset=${DATASET_ROOT}"
echo "gpus=${GPU_LIST}"
echo "seed=${SEED}"
echo "B output=${B_RUN}"
echo "ABC output=${ABC_ROOT}"

# Stage 1: train Sparse-Batch B for the configured 20 epochs.
CUDA_VISIBLE_DEVICES="${GPU_LIST}" \
CONFIG="configs/semantic_kitti/Matter3DToken_B-sparse-batch-reindex-20.yaml" \
LOG_PATH="${B_RUN}" \
DATASET_ROOT="${DATASET_ROOT}" \
SEED="${SEED}" \
MATTER3DTOKEN_DIST_URL="${MATTER3DTOKEN_DIST_URL:-tcp://127.0.0.1:4457}" \
bash "${ROOT}/train_semantickitti_component.sh" \
    2>&1 | tee "${B_RUN}.launcher.log"

BASE_B="${B_RUN}/ckpt_best.pth"
if [[ ! -f "${BASE_B}" ]]; then
    echo "B training did not produce: ${BASE_B}" >&2
    exit 1
fi

# Stage 2: initialize from the new B checkpoint, train A and C, then evaluate.
BASE_B="${BASE_B}" \
GPU_LIST="${GPU_LIST}" \
SEED="${SEED}" \
DATASET_ROOT="${DATASET_ROOT}" \
OUT_ROOT="${ABC_ROOT}" \
MATTER3DTOKEN_DIST_URL="${MATTER3DTOKEN_DIST_URL:-tcp://127.0.0.1:4458}" \
bash "${ROOT}/run_component_ablation_pipeline.sh" \
    2>&1 | tee "${ABC_ROOT}.launcher.log"

echo "[$(date -u +%FT%TZ)] Full A/B/C retraining completed"
echo "Artifacts: ${ABC_ROOT}"
