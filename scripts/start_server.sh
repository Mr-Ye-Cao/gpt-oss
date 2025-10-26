#!/bin/bash
# Start vLLM server for gpt-oss-20b with conda environment

set -e

# Get the directory of this script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Load environment variables from .env file
if [ -f "$PROJECT_ROOT/.env" ]; then
    export $(grep -v '^#' "$PROJECT_ROOT/.env" | grep -v '^$' | xargs)
fi

# Check if API key is set
if [ -z "$VLLM_API_KEY" ]; then
    echo "Error: VLLM_API_KEY not set in .env file"
    echo "Please add VLLM_API_KEY to $PROJECT_ROOT/.env"
    exit 1
fi

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

# Start the server
echo "Starting vLLM server with API key authentication..."
echo "Server will be available at http://127.0.0.1:8000"
echo "API docs at http://127.0.0.1:8000/docs"
echo ""
echo "IMPORTANT: API key authentication is enabled"
echo "All requests must include: Authorization: Bearer \$VLLM_API_KEY"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run vLLM directly with optimal settings and API key authentication
vllm serve "$MODEL_PATH" \
    --host 0.0.0.0 \
    --port 8000 \
    --api-key "$VLLM_API_KEY" \
    --max-model-len 32768 \
    --served-model-name "gpt-oss-20b" \
    --gpu-memory-utilization 0.85 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 4 \
    --disable-log-requests \
    --trust-remote-code
