#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
VENV_DIR="${PROJECT_ROOT}/.venv-vlmevalkit"

python3 -m venv "${VENV_DIR}"
source "${VENV_DIR}/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "${SCRIPT_DIR}/requirements-vlmevalkit.txt"
python -m pip install -e "${SCRIPT_DIR}/VLMEvalKit"

cat <<EOF
VLMEvalKit virtual environment is ready.

Activate it with:
  source "${VENV_DIR}/bin/activate"

Optional environment variables:
  export VLMEVAL_ROOT="${SCRIPT_DIR}/VLMEvalKit"
  export LMUData="\${LMUData:-${PROJECT_ROOT}/.cache/LMUData}"

Smoke test:
  python -c "import vlmeval; print(vlmeval.__version__)"

Run local evaluation:
  python "${SCRIPT_DIR}/run_vlmevalkit_local.py" --help
EOF
