#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# Override these variables when needed, for example:
# CUDA_VISIBLE_DEVICES=0,1 bash train_semantickitti_component.sh
PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/inspire/dataset/semantic-kitti/v1}"
CONFIG="${CONFIG:-configs/semantic_kitti/Matter3DToken_B-drop_0.5.yaml}"
RUN_TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_TIMESTAMP}_$$"
LOG_ROOT="${LOG_ROOT:-logs/semantic_kitti/Matter3DToken_B-drop_0.5}"
LOG_PATH="${LOG_PATH:-${LOG_ROOT}/run_${RUN_ID}}"
LOG_FILE="${LOG_PATH}/train.log"
SEED="${SEED:-0}"
INSTANCE_CACHE_DIR="${INSTANCE_CACHE_DIR:-${SCRIPT_DIR}/data/kitti_instances}"

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    CUDA_VISIBLE_DEVICES="0,1"
    export CUDA_VISIBLE_DEVICES
fi

if [[ ! -d "${DATASET_ROOT}/dataset/sequences" ]]; then
    echo "Dataset not found or invalid: ${DATASET_ROOT}/dataset/sequences" >&2
    exit 1
fi

if [[ ! -f "${CONFIG}" ]]; then
    echo "Config not found: ${CONFIG}" >&2
    exit 1
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

mkdir -p "${LOG_PATH}"
cp -- "${CONFIG}" "${LOG_PATH}/config.yaml"

{
    echo "Run started : $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Repository  : ${SCRIPT_DIR}"
    echo "Dataset     : ${DATASET_ROOT}"
    echo "Config      : ${CONFIG}"
    echo "Log path    : ${LOG_PATH}"
    echo "GPU(s)      : ${CUDA_VISIBLE_DEVICES}"
    echo "Python      : ${PYTHON_BIN}"
    echo "Git commit  : $(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo
    echo "Python version:"
    "${PYTHON_BIN}" --version
    echo
    echo "GPU information:"
    nvidia-smi || true
} | tee "${LOG_PATH}/env.txt"

echo "Training log : ${LOG_FILE}"
echo "Checkpoints  : ${LOG_PATH}/ckpt_last.pth and ${LOG_PATH}/ckpt_best.pth"

# The training code uses all GPUs listed in CUDA_VISIBLE_DEVICES for DDP.
GPU_COUNT="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
TRAIN_ARGS=(
    --dataset semantic_kitti
    --path_dataset "${DATASET_ROOT}"
    --log_path "${LOG_PATH}"
    --config "${CONFIG}"
    --seed "${SEED}"
    --instance_cache_dir "${INSTANCE_CACHE_DIR}"
)

if [[ "${GPU_COUNT}" -gt 1 ]]; then
    TRAIN_ARGS+=(--multiprocessing-distributed)
else
    TRAIN_ARGS+=(--gpu 0)
fi

export PYTHONUNBUFFERED=1

echo "Command: ${PYTHON_BIN} train.py ${TRAIN_ARGS[*]}" | tee -a "${LOG_FILE}"
set +e
"${PYTHON_BIN}" train.py "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE="${PIPESTATUS[0]}"
set -e

echo "Run finished: $(date -u '+%Y-%m-%dT%H:%M:%SZ') exit_code=${EXIT_CODE}" | tee -a "${LOG_FILE}"
exit "${EXIT_CODE}"
