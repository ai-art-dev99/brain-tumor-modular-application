# =============================================================================
# Optional: a fully pinned image.
#
# You do NOT need this to start working. Use the stock RunPod PyTorch template
# plus setup_env.sh for day-to-day development.
#
# Build this image later, once requirements.lock.txt has stabilised, so that
# the manuscript can point to a single immutable artefact. Reviewers of
# medical-AI papers increasingly ask for exactly this.
#
# Build and push:
#   docker build -t <dockerhub-user>/brain-tumour-hybrid:1.0 .
#   docker push  <dockerhub-user>/brain-tumour-hybrid:1.0
#
# Then in RunPod: Deploy -> Custom template -> use that image name.
# =============================================================================

# IMPORTANT: replace this tag with the EXACT tag of the RunPod base image you
# developed against (see /workspace/base_image_info.txt written by setup_env.sh).
# Pinning by digest (@sha256:...) is stronger still and is what makes the build
# genuinely reproducible.
FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL org.opencontainers.image.title="brain-tumour-hybrid"
LABEL org.opencontainers.image.description="Leakage-controlled evaluation of EfficientNetB0 hybrid classifiers for brain tumour MRI"

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8 \
    HF_HOME=/workspace/cache/hf \
    TORCH_HOME=/workspace/cache/torch

# libgl1 / libglib2.0-0 are needed by opencv even in the headless build
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        unzip \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace/repo

# Install from the LOCK file, not requirements.txt, so the image is exact.
COPY requirements.lock.txt .
RUN pip install --no-cache-dir -r requirements.lock.txt

COPY . .

CMD ["/bin/bash"]
