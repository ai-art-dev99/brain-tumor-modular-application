#!/usr/bin/env bash
# =============================================================================
# report_env.sh -- produce the exact strings for the Computational environment
#                  paragraph, and refuse to produce them if they are not
#                  internally consistent.
#
# Why this exists rather than a copied pod specification:
#
#   A provider's instance listing says what was ALLOCATED. The Methods section
#   claims what EXECUTED. Those differ whenever a pod is resized, a base image
#   is replaced, or a wheel is upgraded after launch -- all of which happened at
#   least once in this project. capture_run_env.sh already records the executing
#   environment per run, so the stored record is the source; a live capture is
#   the fallback, and it is labelled as such rather than passed off as the run's
#   own record.
#
#   It also checks one thing a specification sheet cannot: that the GPU's
#   compute capability is actually present in the installed PyTorch build. A
#   Blackwell-class card (sm_120) with a CUDA 12.4 build of PyTorch 2.4 is a
#   combination that cannot execute, so reporting that triple would be a claim
#   a knowledgeable reviewer can falsify from the version numbers alone.
#
# Usage:
#   bash report_env.sh                 # prefer the stored record, else live
#   bash report_env.sh --live          # force a live capture
#   bash report_env.sh --run <run_id>  # a specific stored run
# =============================================================================
set -uo pipefail

RUNS=/workspace/outputs/runs
MODE=auto
RUN_ID=""
while [ $# -gt 0 ]; do
    case "$1" in
        --live) MODE=live ;;
        --run)  RUN_ID="${2:?--run needs a run_id}"; shift ;;
        *) echo "unknown argument: $1" >&2; exit 2 ;;
    esac
    shift
done

hr() { printf '%s\n' "------------------------------------------------------------------"; }

# --- 1. what did the runs actually record? -----------------------------------
echo "==> Stored per-run environment records"
FOUND=""
if [ -d "$RUNS" ]; then
    while IFS= read -r d; do
        FOUND="yes"
        printf '    %s\n' "${d%/env}"
    done < <(find "$RUNS" -mindepth 2 -maxdepth 2 -type d -name env 2>/dev/null | sort)
fi
if [ -z "$FOUND" ]; then
    echo "    none found under $RUNS"
    echo "    -> capture_run_env.sh did not run, or outputs live elsewhere."
    if [ "$MODE" = auto ]; then
        echo "    -> falling back to a LIVE capture, which describes this machine now,"
        echo "       not necessarily the machine the published numbers came from."
        MODE=live
    fi
fi

# --- 2. pick a source --------------------------------------------------------
SRC=""
if [ "$MODE" != live ]; then
    if [ -n "$RUN_ID" ]; then
        SRC="$RUNS/$RUN_ID/env"
    else
        SRC=$(find "$RUNS" -mindepth 2 -maxdepth 2 -type d -name env 2>/dev/null | sort | tail -1)
    fi
    if [ ! -d "$SRC" ]; then
        echo "    requested record not found; switching to live capture" >&2
        MODE=live
    fi
fi

if [ "$MODE" != live ]; then
    hr; echo "==> Stored record: $SRC"; hr
    for f in hardware.txt system.txt torch.txt data.txt git.txt; do
        if [ -f "$SRC/$f" ]; then
            echo "--- $f"; cat "$SRC/$f"; echo
        fi
    done
    if [ -f "$SRC/pip_freeze.txt" ]; then
        echo "--- pip_freeze.txt (reported packages only)"
        grep -iE '^(torch|torchvision|timm|scikit-learn|numpy|pandas|scipy|statsmodels|Pillow|opencv|imagehash|h5py|albumentations)([=<>!~]|$)' \
            "$SRC/pip_freeze.txt" || echo "    (no matches)"
        echo
    fi
    echo "NOTE: every value above was written at the start of that run."
    echo "      Consistency of the GPU/PyTorch pair is checked below against"
    echo "      THIS machine; if this is a different pod, re-run with --run"
    echo "      on the pod that produced the numbers, or trust the stored file."
    hr
fi

# --- 3. live capture and the consistency check -------------------------------
echo "==> Live capture on this machine"
hr
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total,compute_cap --format=csv
else
    echo "nvidia-smi not available"
fi
echo
echo "logical_cores: $(nproc 2>/dev/null || echo NA)"
free -h 2>/dev/null | awk 'NR<=2'
echo
python - <<'PY'
import importlib, platform, sys

print("python:", platform.python_version())


def ver(name):
    try:
        return importlib.import_module(name).__version__
    except Exception:
        return None


for mod in ["torch", "torchvision", "timm", "sklearn", "numpy", "scipy",
            "pandas", "statsmodels"]:
    v = ver(mod)
    label = "scikit-learn" if mod == "sklearn" else mod
    print(f"{label}: {v if v else 'NOT INSTALLED'}")

try:
    import torch
except Exception as e:
    print(f"\nFATAL: torch not importable ({e}); nothing can be reported.")
    sys.exit(1)

print("torch_cuda_build:", torch.version.cuda)
print("cudnn:", torch.backends.cudnn.version())
print("cuda_available:", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("\nCHECK: no CUDA device visible here. If the published runs used a")
    print("       GPU, this machine is not the one to capture from.")
    sys.exit(0)

name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
sm = f"sm_{cap[0]}{cap[1]}"
arch = torch.cuda.get_arch_list()
print("device_name:", name)
print("compute_capability:", f"{cap[0]}.{cap[1]}", f"({sm})")
print("torch_arch_list:", ", ".join(arch))

print()
print("=" * 66)
print("CONSISTENCY CHECK: can this PyTorch build execute on this GPU?")
print("=" * 66)

compiled = sm in arch
# PTX for a lower arch can JIT forward onto a newer one only within the same
# major family; across a major-version boundary it cannot.
same_major = any(a.startswith(f"sm_{cap[0]}") for a in arch)

ok = False
try:
    a = torch.randn(256, 256, device="cuda")
    (a @ a).sum().item()
    torch.cuda.synchronize()
    ok = True
except Exception as e:
    err = f"{type(e).__name__}: {e}"

if ok:
    print(f"  PASS  a matmul executed on {name}.")
    if not compiled:
        print(f"        {sm} is not in the compiled arch list, so this build is")
        print( "        running via PTX JIT. It works, but record the exact")
        print( "        torch build string in the Methods, not just the version.")
    print()
    print("  Report these values:")
    print(f"    GPU                 {name}")
    print(f"    compute capability  {cap[0]}.{cap[1]}")
    print(f"    PyTorch             {torch.__version__}  (CUDA {torch.version.cuda})")
else:
    print(f"  FAIL  no kernel executed on {name}.")
    print(f"        {err}")
    print()
    print(f"        {sm} compiled in: {compiled};  same major family present: {same_major}")
    print( "        This GPU and this PyTorch build are incompatible, so the")
    print( "        published numbers were NOT produced by this pairing. Do not")
    print( "        report it. Find the pod that produced them, or state the")
    print( "        environment of the pod that can reproduce them and say so.")
PY
hr
echo "==> Paste the block above into the thread; the Methods sentence will be"
echo "    filled from it verbatim. Do not transcribe the numbers by hand."