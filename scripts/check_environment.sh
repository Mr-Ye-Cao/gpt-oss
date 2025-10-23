#!/bin/bash
# Check if the environment is set up correctly

echo "="
echo "Environment Check for gpt-oss-20b vLLM Setup"
echo "="
echo ""

# Check conda
if command -v conda &> /dev/null; then
    echo "✓ Conda is installed: $(conda --version)"
else
    echo "✗ Conda is not installed"
fi

# Check if conda environment exists
if conda env list | grep -q "gpt-oss-vllm"; then
    echo "✓ Conda environment 'gpt-oss-vllm' exists"

    # Activate and check packages
    eval "$(conda shell.bash hook)"
    conda activate gpt-oss-vllm

    # Check vLLM
    if python -c "import vllm" 2>/dev/null; then
        VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null)
        echo "✓ vLLM is installed: version $VLLM_VERSION"
    else
        echo "✗ vLLM is not installed"
    fi

    # Check openai-harmony
    if python -c "import openai_harmony" 2>/dev/null; then
        echo "✓ openai-harmony is installed"
    else
        echo "✗ openai-harmony is not installed"
    fi

    # Check gpt-oss package
    if python -c "import gpt_oss" 2>/dev/null; then
        echo "✓ gpt-oss package is installed"
    else
        echo "✗ gpt-oss package is not installed"
    fi
else
    echo "✗ Conda environment 'gpt-oss-vllm' not found"
fi

echo ""

# Check GPU
if command -v nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected:"
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1
else
    echo "✗ NVIDIA GPU not detected or nvidia-smi not found"
fi

echo ""

# Check model
MODEL_PATH="/home/ye/ml-experiments/gpt-oss/gpt-oss-20b"
if [ -d "$MODEL_PATH" ]; then
    echo "✓ Model directory exists at $MODEL_PATH"
    if [ -f "$MODEL_PATH/config.json" ]; then
        echo "✓ Model config found"
    else
        echo "✗ Model config not found (model may be incomplete)"
    fi
else
    echo "✗ Model directory not found at $MODEL_PATH"
fi

echo ""

# Check Docker
if command -v docker &> /dev/null; then
    if docker ps &> /dev/null 2>&1; then
        echo "✓ Docker is installed and running"
        if docker images python:3.11 --format "{{.Repository}}" | grep -q "python"; then
            echo "✓ Python 3.11 Docker image is available"
        else
            echo "⚠ Python 3.11 Docker image not found (run ./scripts/setup_docker.sh)"
        fi
    else
        echo "⚠ Docker is installed but not accessible (may need sudo or user in docker group)"
    fi
else
    echo "⚠ Docker is not installed (optional, needed for Python tool)"
fi

echo ""
echo "="
echo "Check complete!"
echo "="
