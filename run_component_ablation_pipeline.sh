#!/usr/bin/env bash
# Reproduce the three-module SemanticKITTI training and heterogeneous fusion.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DATASET_ROOT="${DATASET_ROOT:-/inspire/hdd/project/agentend2end/ky26289/malaai/Datasets/semantic_kitti}"
GPU_LIST="${GPU_LIST:-0,1,2,3}"
SEED="${SEED:-31}"
RUN_ID="$(date -u +%Y%m%d_%H%M%S)_$$"
OUT_ROOT="${OUT_ROOT:-${ROOT}/logs/research/semantic_kitti/ABC_pipeline_${RUN_ID}}"
BASE_B="${BASE_B:-${ROOT}/logs/retrain/semantic_kitti/B-sparse/run_20260828_204549_3971091/ckpt_best.pth}"
FUSION_WEIGHTS="${FUSION_WEIGHTS:-1:0:0,0:1:0,0:0:1,1:1:1,0.15:0.35:0.50,0.18:0.34:0.48,0.20:0.30:0.50,0.20:0.32:0.48,0.20:0.34:0.46,0.21:0.30:0.49,0.21:0.32:0.47,0.21:0.34:0.45,0.22:0.29:0.49,0.22:0.31:0.47,0.22:0.33:0.45,0.22:0.35:0.43,0.23:0.28:0.49,0.23:0.30:0.47,0.23:0.32:0.45,0.23:0.34:0.43,0.24:0.27:0.49,0.24:0.29:0.47,0.24:0.31:0.45,0.25:0.27:0.48,0.25:0.29:0.46}"

if [[ ! -d "${DATASET_ROOT}/dataset/sequences" ]]; then
    echo "Invalid DATASET_ROOT: ${DATASET_ROOT}" >&2
    exit 1
fi
if [[ ! -f "${BASE_B}" ]]; then
    echo "Missing Sparse-Batch initialization checkpoint: ${BASE_B}" >&2
    exit 1
fi

mkdir -p "${OUT_ROOT}"
export PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
cd "${ROOT}"

run_train() {
    local name="$1"
    local config="$2"
    local init_checkpoint="$3"
    local run_dir="${OUT_ROOT}/${name}"
    mkdir -p "${run_dir}"
    if [[ "${config}" != "${run_dir}/config.yaml" ]]; then
        cp -- "${config}" "${run_dir}/config.yaml"
    fi
    {
        echo "start=$(date -u +%FT%TZ)"
        echo "config=${config}"
        echo "init_checkpoint=${init_checkpoint}"
        echo "seed=${SEED}"
        echo "gpus=${GPU_LIST}"
    } > "${run_dir}/run_metadata.txt"
    CUDA_VISIBLE_DEVICES="${GPU_LIST}" python train.py \
        --dataset semantic_kitti \
        --path_dataset "${DATASET_ROOT}" \
        --log_path "${run_dir}" \
        --config "${run_dir}/config.yaml" \
        --init_checkpoint "${init_checkpoint}" \
        --seed "${SEED}" \
        --multiprocessing-distributed 2>&1 | tee "${run_dir}/console.log"
    test -f "${run_dir}/ckpt_best.pth"
}

Z_DIR="${OUT_ROOT}/A_z_aware_prototype"
C_DIR="${OUT_ROOT}/C_budgeted_router"
mkdir -p "${Z_DIR}" "${C_DIR}"
cp -- configs/semantic_kitti/Matter3DToken_B-z-aware-prototype-v3-transfer-20.yaml "${Z_DIR}/config.yaml"
cp -- configs/semantic_kitti/Matter3DToken_B-budgeted-router-nearfull-semantic-20.yaml "${C_DIR}/config.yaml"
python research/create_z_aware_init.py --config "${Z_DIR}/config.yaml" --source "${BASE_B}" --output "${Z_DIR}/ckpt_init.pth"
python research/create_z_aware_init.py --config "${C_DIR}/config.yaml" --source "${BASE_B}" --output "${C_DIR}/ckpt_init.pth"

run_train "A_z_aware_prototype" "${Z_DIR}/config.yaml" "${Z_DIR}/ckpt_init.pth"
run_train "C_budgeted_router" "${C_DIR}/config.yaml" "${C_DIR}/ckpt_init.pth"

FUSION_DIR="${OUT_ROOT}/fusion"
mkdir -p "${FUSION_DIR}"
CUDA_VISIBLE_DEVICES="${GPU_LIST%%,*}" python research/eval_heterogeneous_blend.py \
    --dataset "${DATASET_ROOT}" \
    --config_a "${Z_DIR}/config.yaml" --checkpoint_a "${Z_DIR}/ckpt_best.pth" \
    --config_b configs/semantic_kitti/Matter3DToken_B-sparse-batch-reindex-20.yaml --checkpoint_b "${BASE_B}" \
    --config_c "${C_DIR}/config.yaml" --checkpoint_c "${C_DIR}/ckpt_best.pth" \
    --weight_sets "${FUSION_WEIGHTS}" \
    --output "${FUSION_DIR}/metrics.txt" 2>&1 | tee "${FUSION_DIR}/eval.log"

python research/audit_abc_goal.py \
    --checkpoint_a "${Z_DIR}/ckpt_best.pth" \
    --checkpoint_b "${BASE_B}" \
    --checkpoint_c "${C_DIR}/ckpt_best.pth" \
    --log_a "${Z_DIR}/console.log" \
    --log_b "$(dirname "${BASE_B}")/train.log" \
    --log_c "${C_DIR}/console.log" \
    --fusion_metrics "${FUSION_DIR}/metrics.txt" \
    --output "${OUT_ROOT}/goal_audit.md"

echo "Artifacts written to ${OUT_ROOT}"
