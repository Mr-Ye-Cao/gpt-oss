#!/bin/bash
# Start vLLM server for gpt-oss-20b with built-in tools enabled
# NOTE: Built-in tools (--tool-server demo) have known bugs in vLLM 0.11.0
# Use the fixed tool_calling_demo_fixed.py for more reliable tool calling

set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate gpt-oss-vllm

# Check if model exists
MODEL_PATH="/home/ye/ml-experiments/gpt-oss/gpt-oss-20b"
if [ ! -d "$MODEL_PATH" ]; then
    echo "Error: Model directory not found at $MODEL_PATH"
    echo "Please download the model first using:"
    echo "  conda activate gpt-oss-vllm"
    echo "  hf download openai/gpt-oss-20b --local-dir $MODEL_PATH"
    exit 1
fi

# Set environment variables to avoid CUDA OOM
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0

# Check if Docker is available for Python tool
if command -v docker &> /dev/null; then
    echo "✓ Docker detected - Python interpreter tool will be available"
else
    echo "⚠ Docker not found - Python interpreter tool will not work"
    echo "  Install Docker with: ./scripts/setup_docker.sh"
fi

# Start the server
echo "Starting vLLM server with built-in tools..."
echo "Server will be available at http://127.0.0.1:8000"
echo "API docs at http://127.0.0.1:8000/docs"
echo ""
echo "Built-in tools enabled:"
echo "  - Browser (web search via Exa)"
echo "  - Python interpreter (via Docker)"
echo ""
echo "⚠️  NOTE: Built-in tools have known bugs in current vLLM version"
echo "    For more reliable tool calling, use: python scripts/tool_calling_demo_fixed.py"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run vLLM with built-in tools enabled
vllm serve "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 8192 \
    --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 2048 \
    --max-num-seqs 4 \
    --disable-log-requests \
    --trust-remote-code \
    --tool-server demo
