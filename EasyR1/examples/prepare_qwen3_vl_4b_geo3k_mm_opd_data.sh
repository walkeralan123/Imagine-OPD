#!/bin/bash

set -euo pipefail
set -x

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
PROJECT_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
WORKSPACE_ROOT=$(cd "${PROJECT_ROOT}/../.." && pwd)

DEFAULT_OPD_INPUT="${WORKSPACE_ROOT}/data/opd/opd_6k.json"
if [ "$#" -gt 0 ]; then
    OPD_INPUTS=("$@")
else
    OPD_INPUTS=("${DEFAULT_OPD_INPUT}")
fi

OPD_NORMALIZED_JSON=${OPD_NORMALIZED_JSON:-${WORKSPACE_ROOT}/data/opd_json/opd_train_normalized.json}
OPD_SAVE_DIR=${OPD_SAVE_DIR:-${WORKSPACE_ROOT}/data/opd_json}
OPD_DUPLICATE_TO=${OPD_DUPLICATE_TO:-48}
OPD_LIMIT=${OPD_LIMIT:-}

VSTAR_SOURCE=${VSTAR_SOURCE:-${WORKSPACE_ROOT}/data/vstar}
VSTAR_SAVE_DIR=${VSTAR_SAVE_DIR:-${WORKSPACE_ROOT}/data/vstar_bench}

DEDUPE_BY_SAMPLE_ID=${DEDUPE_BY_SAMPLE_ID:-1}
SKIP_MISSING_IMAGES=${SKIP_MISSING_IMAGES:-0}

cd "${PROJECT_ROOT}"

normalize_cmd=(
    python examples/data_preprocess/opd_xctarget_to_json.py
    --inputs "${OPD_INPUTS[@]}"
    --save_path "${OPD_NORMALIZED_JSON}"
)

if [ "${DEDUPE_BY_SAMPLE_ID}" = "1" ]; then
    normalize_cmd+=(--dedupe_by_sample_id)
fi

if [ "${SKIP_MISSING_IMAGES}" = "1" ]; then
    normalize_cmd+=(--skip_missing_images)
fi

if [ -n "${OPD_LIMIT}" ]; then
    normalize_cmd+=(--limit "${OPD_LIMIT}")
fi

"${normalize_cmd[@]}"

python examples/data_preprocess/opd_json.py \
    --input "${OPD_NORMALIZED_JSON}" \
    --save_dir "${OPD_SAVE_DIR}" \
    --duplicate_to "${OPD_DUPLICATE_TO}"

python examples/data_preprocess/vstar_bench.py \
    --dataset_path "${VSTAR_SOURCE}" \
    --save_dir "${VSTAR_SAVE_DIR}"

echo "Prepared training parquet: ${OPD_SAVE_DIR}/train.parquet"
echo "Prepared validation parquet: ${VSTAR_SAVE_DIR}/test.parquet"
