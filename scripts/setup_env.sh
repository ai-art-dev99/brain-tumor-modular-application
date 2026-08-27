#!/usr/bin/env bash
# =============================================================================
# setup_env.sh -- one-time bootstrap of the RunPod environment.
#
# Run ONCE after creating the pod for the first time. The virtual environment
# is created on the network volume (/workspace), so it survives pod
# termination and does not need to be rebuilt on every restart.
#
# Usage:  bash setup_env.sh
# =============================================================================
set -euo pipefail

WORKSPACE=/workspace
VENV="${WORKSPACE}/venv"
REPO="${WORKSPACE}/repo"

echo "==> Checking that the network volume is mounted"
if ! mountpoint -q "${WORKSPACE}" 2>/dev/null && [ ! -d "${WORKSPACE}" ]; then
    echo "ERROR: ${WORKSPACE} does not exist."
    echo "The network volume is not attached. Stop and redeploy the pod with"
    echo "the volume mounted at ${WORKSPACE}, otherwise all work will be lost"
    echo "when the pod is terminated."
    exit 1
fi

echo "==> Creating directory layout"
mkdir -p "${WORKSPACE}"/{data/{raw,clean,manifest},repo,outputs,cache/{hf,torch}}

echo "==> Recording the base image identity (needed for the Methods section)"
{
    echo "captured_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname: $(hostname)"
    echo "kernel: $(uname -r)"
    echo "python: $(python3 --version 2>&1)"
    echo "cuda_driver:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv 2>&1 | sed 's/^/  /'
} > "${WORKSPACE}/base_image_info.txt"
cat "${WORKSPACE}/base_image_info.txt"

echo
echo "==> Creating virtualenv at ${VENV}"
echo "    (--system-site-packages so torch/torchvision come from the base image"
echo "     instead of being re-downloaded, ~3 GB saved and no CUDA mismatch)"
if [ ! -d "${VENV}" ]; then
    python3 -m venv --system-site-packages "${VENV}"
else
    echo "    already exists, reusing"
fi

# shellcheck disable=SC1091
source "${VENV}/bin/activate"

echo "==> Upgrading pip tooling"
pip install --upgrade pip setuptools wheel

echo "==> Verifying that torch is visible from the base image"
python - <<'PY'
import sys
try:
    import torch
except Exception as e:
    sys.exit(f"ERROR: torch not importable from the base image: {e}")
print(f"  torch       {torch.__version__}")
print(f"  cuda build  {torch.version.cuda}")
print(f"  cuda avail  {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  device      {torch.cuda.get_device_name(0)}")
else:
    print("  WARNING: no CUDA device visible. Check the pod's GPU allocation.")
PY

echo "==> Installing project dependencies"
pip install -r "${REPO}/requirements.txt"

echo "==> Freezing the resolved environment"
pip freeze > "${REPO}/requirements.lock.txt"
echo "    wrote ${REPO}/requirements.lock.txt  <-- commit this file"

echo "==> Configuring shell defaults"
BASHRC="${HOME}/.bashrc"
add_line() { grep -qxF "$1" "${BASHRC}" 2>/dev/null || echo "$1" >> "${BASHRC}"; }
add_line "source ${VENV}/bin/activate"
add_line "export HF_HOME=${WORKSPACE}/cache/hf"
add_line "export TORCH_HOME=${WORKSPACE}/cache/torch"
add_line "export PYTHONHASHSEED=0"
add_line "export CUBLAS_WORKSPACE_CONFIG=:4096:8"   # required for deterministic cuBLAS
add_line "cd ${REPO}"

echo
echo "============================================================"
echo "Setup complete."
echo
echo "  venv     : ${VENV}"
echo "  repo     : ${REPO}"
echo "  data     : ${WORKSPACE}/data"
echo "  outputs  : ${WORKSPACE}/outputs"
echo
echo "Open a new shell (or 'source ~/.bashrc') to activate the environment."
echo "Next step: fetch the datasets with scripts/fetch_data.sh"
echo "============================================================"
