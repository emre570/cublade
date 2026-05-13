#!/bin/bash
# Detect CUDA version and install cublade with matching PyTorch.
# CUDA kernels are NOT compiled here - they JIT-compile on first use
# (cached at ~/.cache/torch_extensions/, so subsequent runs are instant).

set -e

# Detect CUDA version from nvcc
if command -v nvcc &> /dev/null; then
    CUDA_VERSION=$(nvcc --version | grep "release" | sed -n 's/.*release \([0-9]*\.[0-9]*\).*/\1/p')
    echo "Detected CUDA $CUDA_VERSION"
else
    echo "Error: nvcc not found. Install CUDA toolkit first."
    exit 1
fi

# Map CUDA version to PyTorch wheel tag
case "$CUDA_VERSION" in
    11.8*) CUDA_TAG="cu118" ;;
    12.1*) CUDA_TAG="cu121" ;;
    12.4*) CUDA_TAG="cu124" ;;
    12.6*|12.8*) CUDA_TAG="cu128" ;;
    13.0*) CUDA_TAG="cu130" ;;
    *)
        echo "Warning: CUDA $CUDA_VERSION not directly supported, trying cu130"
        CUDA_TAG="cu130"
        ;;
esac

TORCH_INDEX="https://download.pytorch.org/whl/$CUDA_TAG"
echo "Using PyTorch index: $TORCH_INDEX"

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
fi

source .venv/bin/activate

# Install torch first (from correct index)
echo "Installing PyTorch for CUDA $CUDA_TAG..."
uv pip install torch --index-url "$TORCH_INDEX"

# Install cublade (metadata-only, no kernel compilation here)
echo "Installing cublade (metadata only - kernels JIT-compile on first use)..."
uv pip install -e .

echo ""
echo "Done. Activate with: source .venv/bin/activate"
echo "First call to a CUDA kernel takes ~5-30s (compile); subsequent calls are instant."
