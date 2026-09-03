#!/usr/bin/env bash

set -euo pipefail

# Train Matter3DToken A+B+C while disabling the one-time InstanceCutMix
# extraction/augmentation. The original source code and YAML are untouched.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATASET_ROOT="${DATASET_ROOT:-/inspire/dataset/semantic-kitti/v1}"
BASE_CONFIG="${BASE_CONFIG:-configs/semantic_kitti/Matter3DToken_joint_abc_h200_20e.yaml}"
LOG_ROOT="${LOG_ROOT:-logs/offline_h200/Matter3DToken_joint_abc_skip_instance}"
SEED="${SEED:-31}"
INSTANCE_CACHE_DIR="${INSTANCE_CACHE_DIR:-${SCRIPT_DIR}/data/kitti_instances}"

if [[ ! -f "${BASE_CONFIG}" ]]; then
    echo "Configuration file was not found: ${BASE_CONFIG}" >&2
    exit 1
fi

# The existing launcher records the exact CONFIG file in the run directory.
# Generate the modified copy outside the repository and remove it on exit.
TEMP_CONFIG="$(mktemp /tmp/matter3dtoken_no_instance_cutmix.XXXXXX.yaml)"
cleanup() {
    rm -f "${TEMP_CONFIG}"
}
trap cleanup EXIT

sed -E \
    's/^([[:space:]]*instance_cutmix:)[[:space:]]*(true|True|TRUE)[[:space:]]*$/\1 false/' \
    "${BASE_CONFIG}" > "${TEMP_CONFIG}"

if ! rg -q '^[[:space:]]*instance_cutmix:[[:space:]]*false[[:space:]]*$' "${TEMP_CONFIG}"; then
    echo "Could not disable instance_cutmix in ${BASE_CONFIG}" >&2
    exit 1
fi

echo "Starting Matter3DToken A+B+C training with instance extraction skipped."
echo "Dataset : ${DATASET_ROOT}"
echo "Config  : ${BASE_CONFIG} (temporary instance_cutmix=false copy)"
echo "Logs    : ${LOG_ROOT}"

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}" \
DATASET_ROOT="${DATASET_ROOT}" \
CONFIG="${TEMP_CONFIG}" \
LOG_ROOT="${LOG_ROOT}" \
SEED="${SEED}" \
INSTANCE_CACHE_DIR="${INSTANCE_CACHE_DIR}" \
bash "${SCRIPT_DIR}/train_matter3dtoken.sh"
