#!/usr/bin/env bash
# =============================================================================
# capture_run_env.sh -- snapshot the exact environment of a single run.
#
# Call this at the START of every experiment, from inside the run script, so
# that outputs/runs/<run_id>/ is self-describing. This is what turns
# "we used Google Colab" into an answerable Methods section (reviewer point 6).
#
# Usage:  bash scripts/capture_run_env.sh <run_id>
# =============================================================================
set -euo pipefail

RUN_ID="${1:?usage: capture_run_env.sh <run_id>}"
OUT="/workspace/outputs/runs/${RUN_ID}/env"
mkdir -p "${OUT}"

echo "==> Capturing environment for run_id=${RUN_ID}"

# --- hardware ----------------------------------------------------------------
{
    echo "# GPU"
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap \
               --format=csv 2>&1 || echo "no nvidia-smi"
    echo
    echo "# CPU"
    echo "logical_cores: $(nproc)"
    lscpu 2>/dev/null | grep -E 'Model name|Socket|Thread|Core' || true
    echo
    echo "# Memory"
    free -h 2>/dev/null | head -2 || true
} > "${OUT}/hardware.txt"

# --- software ----------------------------------------------------------------
{
    echo "captured_at: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "hostname: $(hostname)"
    echo "kernel: $(uname -r)"
    echo "python: $(python --version 2>&1)"
} > "${OUT}/system.txt"

pip freeze > "${OUT}/pip_freeze.txt"

python - > "${OUT}/torch.txt" <<'PY'
import torch, platform
print("torch:", torch.__version__)
print("torchvision:", __import__("torchvision").__version__)
print("cuda_build:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("cuda_available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
print("platform:", platform.platform())
PY

# --- code provenance ---------------------------------------------------------
{
    echo "commit: $(git rev-parse HEAD 2>/dev/null || echo 'not a git repo')"
    echo "branch: $(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo '-')"
    echo "dirty: $(test -n "$(git status --porcelain 2>/dev/null)" && echo yes || echo no)"
} > "${OUT}/git.txt"

# A dirty working tree means the recorded commit does not describe the code
# that actually ran. Warn loudly rather than silently producing an
# unreproducible result.
if [ -n "$(git status --porcelain 2>/dev/null)" ]; then
    echo "    WARNING: uncommitted changes present. Commit before a final run."
    git diff > "${OUT}/uncommitted.diff" 2>/dev/null || true
fi

# --- data provenance ---------------------------------------------------------
# Hash the manifest rather than the images: cheap, and it pins the exact
# dataset state (post-deduplication) that the run consumed.
MANIFEST=/workspace/data/manifest/manifest.csv
if [ -f "${MANIFEST}" ]; then
    {
        echo "manifest: ${MANIFEST}"
        echo "sha256: $(sha256sum "${MANIFEST}" | cut -d' ' -f1)"
        echo "rows: $(($(wc -l < "${MANIFEST}") - 1))"
    } > "${OUT}/data.txt"
else
    echo "manifest: NOT FOUND (run build_manifest.py first)" > "${OUT}/data.txt"
fi

echo "    wrote ${OUT}/"
