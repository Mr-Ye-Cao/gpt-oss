#!/bin/bash
# Start vLLM server with LOWER memory usage (for testing or if you have other processes)

set -e

eval "$(conda shell.bash hook)"
conda activate gpt-oss-vllm

MODEL_PATH="/home/ye/ml-experiments/gpt-oss/gpt-oss-20b"

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_USE_FLASHINFER_SAMPLER=0

echo "Starting vLLM server with LOWER memory footprint..."
echo "Server will be available at http://127.0.0.1:8000"
echo ""
echo "⚠️  Lower memory = smaller KV cache = less concurrent capacity"
echo ""

# Lower memory settings:
# - Smaller context (4096 instead of 8192)
# - Only 1-2 concurrent requests
# - Less GPU memory utilization
vllm serve "$MODEL_PATH" \
    --host 127.0.0.1 \
    --port 8000 \
    --max-model-len 4096 \
    --gpu-memory-utilization 0.75 \
    --max-num-batched-tokens 1024 \
    --max-num-seqs 2 \
    --disable-log-requests \
    --trust-remote-code

# Expected memory usage: ~20-22 GB instead of ~28 GB
