#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/inspire/dataset/semantic-kitti/v1}"
CONFIG="${CONFIG:-configs/semantic_kitti/Matter3DToken_joint_abc_h200_20e.yaml}"
LOG_ROOT="${LOG_ROOT:-logs/offline_h200/Matter3DToken_joint_abc_20e}"
SEED="${SEED:-31}"
INSTANCE_CACHE_DIR="${INSTANCE_CACHE_DIR:-${SCRIPT_DIR}/data/kitti_instances}"

if [[ -z "${CUDA_VISIBLE_DEVICES+x}" ]]; then
    GPU_COUNT_DETECTED="$(nvidia-smi -L | awk 'END {print NR}')"
    if [[ "${GPU_COUNT_DETECTED}" -lt 1 ]]; then
        echo "No NVIDIA GPU was detected." >&2
        exit 1
    fi
    CUDA_VISIBLE_DEVICES="$(seq -s, 0 "$((GPU_COUNT_DETECTED - 1))")"
    export CUDA_VISIBLE_DEVICES
fi

GPU_COUNT="$(awk -F, '{print NF}' <<< "${CUDA_VISIBLE_DEVICES}")"
if [[ "${GPU_COUNT}" -lt 1 ]]; then
    echo "CUDA_VISIBLE_DEVICES does not contain a valid GPU list." >&2
    exit 1
fi
if (( 8 % GPU_COUNT != 0 )); then
    echo "Global batch size 8 must be divisible by GPU count ${GPU_COUNT}." >&2
    echo "Use 1, 2, 4, or 8 visible GPUs." >&2
    exit 1
fi

if [[ ! -x "$(command -v "${PYTHON_BIN}" 2>/dev/null || true)" ]]; then
    echo "Python executable was not found: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${CONFIG}" ]]; then
    echo "Configuration file was not found: ${CONFIG}" >&2
    exit 1
fi
if [[ ! -d "${DATASET_ROOT}/dataset/sequences" ]]; then
    echo "SemanticKITTI dataset was not found: ${DATASET_ROOT}/dataset/sequences" >&2
    exit 1
fi

RUN_TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_ID="${RUN_TIMESTAMP}_$$"
RUN_DIR="${LOG_ROOT}/run_${RUN_ID}"
LOG_FILE="${RUN_DIR}/train.log"
ENV_FILE="${RUN_DIR}/environment.txt"
mkdir -p "${RUN_DIR}"
cp "${CONFIG}" "${RUN_DIR}/config.yaml"
git status --short > "${RUN_DIR}/git_status.txt" 2>/dev/null || true
git diff --binary > "${RUN_DIR}/code_changes.patch" 2>/dev/null || true
"${PYTHON_BIN}" -m pip freeze > "${RUN_DIR}/python_packages.txt" 2>/dev/null || true

MASTER_PORT="${MASTER_PORT:-$((29500 + ($$ % 1000)))}"
export MATTER3DTOKEN_DIST_URL="tcp://127.0.0.1:${MASTER_PORT}"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"

{
    echo "Run started (UTC): $(date -u '+%Y-%m-%dT%H:%M:%SZ')"
    echo "Mode             : fresh training (no checkpoint loaded)"
    echo "Repository       : ${SCRIPT_DIR}"
    echo "Dataset          : ${DATASET_ROOT}"
    echo "Configuration    : ${CONFIG}"
    echo "Run directory    : ${RUN_DIR}"
    echo "Visible GPUs     : ${CUDA_VISIBLE_DEVICES}"
    echo "GPU count        : ${GPU_COUNT}"
    echo "Global batch size: 8"
    echo "Epochs           : 20"
    echo "Precision        : BF16 autocast"
    echo "Random seed      : ${SEED}"
    echo "Distributed URL  : ${MATTER3DTOKEN_DIST_URL}"
    echo "Git commit       : $(git rev-parse HEAD 2>/dev/null || echo unavailable)"
    echo "Config SHA256    : $(sha256sum "${CONFIG}" | awk '{print $1}')"
    echo
    "${PYTHON_BIN}" --version
    "${PYTHON_BIN}" -c 'import torch; print("PyTorch:", torch.__version__); print("CUDA:", torch.version.cuda); print("cuDNN:", torch.backends.cudnn.version())'
    echo
    nvidia-smi
} | tee "${ENV_FILE}"

TRAIN_ARGS=(
    --dataset semantic_kitti
    --path_dataset "${DATASET_ROOT}"
    --log_path "${RUN_DIR}"
    --config "${CONFIG}"
    --seed "${SEED}"
    --instance_cache_dir "${INSTANCE_CACHE_DIR}"
)
if [[ "${GPU_COUNT}" -gt 1 ]]; then
    TRAIN_ARGS+=(--multiprocessing-distributed)
else
    TRAIN_ARGS+=(--gpu 0)
fi

echo "Training log: ${LOG_FILE}"
echo "Checkpoints : ${RUN_DIR}/ckpt_last.pth and ${RUN_DIR}/ckpt_best.pth"
echo "Command     : ${PYTHON_BIN} train.py ${TRAIN_ARGS[*]}" | tee "${LOG_FILE}"

set +e
"${PYTHON_BIN}" train.py "${TRAIN_ARGS[@]}" 2>&1 | tee -a "${LOG_FILE}"
EXIT_CODE="${PIPESTATUS[0]}"
set -e

{
    echo
    echo "Run finished (UTC): $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "Exit code         : ${EXIT_CODE}"
    echo "Run directory     : ${RUN_DIR}"
} | tee -a "${LOG_FILE}"

exit "${EXIT_CODE}"
