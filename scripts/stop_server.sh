#!/bin/bash
# Stop vLLM server

echo "Stopping vLLM server..."
pkill -f "vllm serve" || echo "No vLLM server running"
echo "Done."
