#!/usr/bin/env python
"""
Optimized vLLM server for gpt-oss-20b
Configured for single-user, low-latency inference on RTX 5090
"""

import argparse
import os
import sys

def main():
    parser = argparse.ArgumentParser(description="Start vLLM server for gpt-oss-20b")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/ye/ml-experiments/gpt-oss/gpt-oss-20b",
        help="Path to the model directory"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (127.0.0.1 for local only, 0.0.0.0 for network access)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to serve on"
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=8192,
        help="Maximum context length (lower = faster prefill)"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization (0.9 = 90%)"
    )

    args = parser.parse_args()

    # Set environment variable for flashinfer sampler (may need to disable for stability)
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

    # Import vLLM components
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.entrypoints.openai.serving_chat import OpenAIServingChat
    from vllm.entrypoints.openai.serving_completion import OpenAIServingCompletion
    import uvicorn
    import asyncio

    print("=" * 80)
    print("Starting vLLM Server for gpt-oss-20b")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Server: http://{args.host}:{args.port}")
    print(f"Max context: {args.max_model_len} tokens")
    print(f"GPU memory utilization: {args.gpu_memory_utilization * 100}%")
    print("=" * 80)
    print("\nOptimizations for single-user, low-latency:")
    print("  - Small batch size for faster response")
    print("  - Moderate context length for faster prefill")
    print("  - CUDA graphs for reduced overhead")
    print("=" * 80)

    # Start the vLLM OpenAI-compatible server
    os.system(f"""vllm serve {args.model_path} \\
        --host {args.host} \\
        --port {args.port} \\
        --max-model-len {args.max_model_len} \\
        --gpu-memory-utilization {args.gpu_memory_utilization} \\
        --max-num-batched-tokens 2048 \\
        --max-num-seqs 8 \\
        --disable-log-requests \\
        --trust-remote-code
    """)

if __name__ == "__main__":
    main()
