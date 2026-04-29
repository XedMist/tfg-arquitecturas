# CUDA 12.6 Python 3.10 (Needed for timm 0.6.11, which is needed for MambaOut).
FROM nvcr.io/nvidia/pytorch:24.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    CUDA_HOME=/usr/local/cuda \
    FORCE_CUDA=1 \
    # Build CUDA extensions for A30 (Ampere)
    TORCH_CUDA_ARCH_LIST="8.0" \
    # Slightly nicer builds for big extensions
    MAX_JOBS=8

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    ninja-build \
    cmake \
    pkg-config \
    ffmpeg \
    libglib2.0-0 \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Pin numpy to 1.x (This Torch build breaks with numpy 2.x)
# Also pin setuptools < 70 to keep pkg_resources.packaging available for torch cpp_extension flows.
RUN python -m pip install --upgrade pip && \
    printf "numpy<2\nsetuptools<70\n" > /tmp/constraints.txt && \
    python -m pip install -c /tmp/constraints.txt --upgrade --force-reinstall \
      "numpy<2" "setuptools<70" "wheel" "packaging"

# Core Python deps (MambaOut README asks for timm==0.6.11)
RUN pip install -c /tmp/constraints.txt \
      timm==0.6.11 \
      transformers==4.44.2 \
      accelerate \
      datasets \
      einops \
      safetensors \
      pillow \
      scipy \
      pandas \
      matplotlib \
      opencv-python-headless \
      vit-pytorch

# Install hustvl/Vim inside the image (includes CUDA extensions)
RUN git clone --depth 1 https://github.com/hustvl/Vim.git /opt/Vim && \
    pip install -c /tmp/constraints.txt --no-build-isolation /opt/Vim/causal-conv1d && \
    pip install -c /tmp/constraints.txt --no-build-isolation /opt/Vim/mamba-1p1p1

# Clone MambaOut inside the image
RUN git clone --depth 1 https://github.com/yuweihao/MambaOut.git /opt/MambaOut

RUN pip install -c /tmp/constraints.txt \
    "albumentations>=2.0.8" \
    "hydra-core>=1.3.2" \
    "omegaconf>=2.3.0" \
    "rich>=15.0.0" \
    "torchmetrics>=1.9.0"

ENV PYTHONPATH="/opt/Vim:/opt/Vim/vim:/opt/MambaOut:/workspace/src"

WORKDIR /workspace

COPY pyproject.toml README.md ./
COPY src/ ./src/

RUN mkdir -p data outputs checkpoints

CMD ["python", "src/main.py"]
