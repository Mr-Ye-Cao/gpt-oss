# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is the gpt-oss repository by OpenAI, containing reference implementations for running the gpt-oss-120b and gpt-oss-20b models. The project includes:
- Multiple inference backends (PyTorch, Triton, Metal, vLLM, Transformers)
- Tool implementations (browser, python execution, apply_patch)
- Client applications (terminal chat, Responses API server)
- Evaluation scripts and examples

**Important**: These models use the Harmony response format exclusively and will not work correctly with other formats.

## Development Setup

### Installation

```bash
# Basic tools only
pip install -e .

# For PyTorch implementation (requires 4x H100 GPUs)
pip install -e ".[torch]"

# For Triton implementation (single 80GB GPU)
# First install triton from source:
git clone https://github.com/triton-lang/triton
cd triton/
pip install -r python/requirements.txt
pip install -e . --verbose --no-build-isolation
pip install -e python/triton_kernels
cd ..
pip install -e ".[triton]"

# For Metal implementation (Apple Silicon only)
GPTOSS_BUILD_METAL=1 pip install -e ".[metal]"

# For testing
pip install -e ".[test]"
```

### Model Download

```bash
# Download gpt-oss-120b
hf download openai/gpt-oss-120b --include "original/*" --local-dir gpt-oss-120b/

# Download gpt-oss-20b (full download including all formats)
hf download openai/gpt-oss-20b --local-dir gpt-oss-20b/
```

### vLLM Production Setup (RTX 5090 / Single GPU)

A complete vLLM setup optimized for single-user, low-latency inference is available in the `scripts/` directory:

```bash
# Quick start
./scripts/start_server.sh                # Start OpenAI-compatible API server
conda activate gpt-oss-vllm
python scripts/simple_chat.py            # Interactive chat
python scripts/tool_calling_demo.py      # Test function calling
python scripts/benchmark_latency.py      # Measure performance

# Check setup
./scripts/check_environment.sh

# Full documentation
# See VLLM_SETUP.md for complete guide
```

This setup includes:
- **Conda environment** `gpt-oss-vllm` with vLLM 0.11.0, openai-harmony, and all dependencies
- **Server scripts** with optimizations for single-user, low-latency (max_num_seqs=4, max_model_len=8192)
- **Client scripts** for chat, tool calling, and API integration
- **Performance benchmarking** tools
- **Tool support** for browser (web search) and Python execution capabilities

The vLLM server is OpenAI-compatible and can be used with any OpenAI client library.

## Common Commands

### Running Inference

```bash
# PyTorch backend (requires 4x H100)
torchrun --nproc-per-node=4 -m gpt_oss.generate gpt-oss-120b/original/

# Triton backend (single 80GB GPU)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m gpt_oss.generate --backend triton gpt-oss-120b/original/

# Metal backend (after converting weights)
python gpt_oss/metal/examples/generate.py gpt-oss-20b/metal/model.bin -p "your prompt"
```

### Chat Application

```bash
# Basic chat with triton backend
python -m gpt_oss.chat gpt-oss-120b/original/ --backend triton

# With browser tool enabled
python -m gpt_oss.chat gpt-oss-120b/original/ --backend triton -b

# With python tool enabled
python -m gpt_oss.chat gpt-oss-120b/original/ --backend triton -p

# With custom reasoning effort
python -m gpt_oss.chat gpt-oss-120b/original/ --backend triton -r high

# With vLLM backend
python -m gpt_oss.chat gpt-oss-120b/ --backend vllm
```

### Responses API Server

```bash
# Start with triton backend
python -m gpt_oss.responses_api.serve --checkpoint gpt-oss-120b/original/ --inference-backend triton --port 8000

# Start with Metal backend (macOS)
python -m gpt_oss.responses_api.serve --checkpoint gpt-oss-120b/metal/model.bin --inference-backend metal

# Start with Ollama backend
python -m gpt_oss.responses_api.serve --inference-backend ollama --port 8000

# Start with vLLM backend
python -m gpt_oss.responses_api.serve --checkpoint gpt-oss-120b/ --inference-backend vllm
```

### Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_responses_api.py

# Run with verbose output
pytest -v
```

### Metal Weight Conversion

```bash
# Convert SafeTensor weights to Metal format
python gpt_oss/metal/scripts/create-local-model.py -s <model_dir> -d <output_file>
```

## Code Architecture

### Inference Backends

The repository provides multiple inference implementations with different performance characteristics:

- **PyTorch (`gpt_oss/torch/`)**: Unoptimized reference implementation showing exact model architecture. Requires 4x H100 GPUs due to lack of optimization. Main entry point: `gpt_oss.torch.model.TokenGenerator`

- **Triton (`gpt_oss/triton/`)**: Optimized implementation using custom Triton kernels for MoE with MXFP4 quantization. Runs on single 80GB GPU. Uses `gpt_oss.triton.moe` for the MoE kernel and custom attention in `gpt_oss.triton.attention`

- **Metal (`gpt_oss/metal/`)**: Apple Silicon implementation with custom Metal shaders. Requires weight conversion to Metal binary format. C++ implementation compiled via CMake.

- **vLLM**: Production-ready serving via vLLM library integration in `gpt_oss/vllm/token_generator.py`

### Harmony Format Integration

All model interactions use the `openai-harmony` package for message encoding/decoding:
- System messages contain tool definitions and configuration
- Messages are encoded to token IDs via `encoding.render_conversation_for_completion()`
- Model outputs are parsed back to structured messages via `encoding.parse_messages_from_completion_tokens()`
- Stop tokens are retrieved via `encoding.stop_tokens_for_assistant_actions()`

### Tool System

Tools implement the base `Tool` class from `gpt_oss/tools/tool.py`:

- **SimpleBrowserTool (`gpt_oss/tools/simple_browser/`)**: Implements `search`, `open`, and `find` operations. Uses pluggable backends (YouCom, Exa) via `gpt_oss.tools.simple_browser.backend.Backend`. Manages scrollable text windows and citation extraction.

- **PythonTool (`gpt_oss/tools/python_docker/`)**: Executes Python code in Docker containers (or via uv/Jupyter depending on `PYTHON_EXECUTION_BACKEND` env var). Stateless implementation overriding the harmony default.

- **apply_patch (`gpt_oss/tools/apply_patch.py`)**: File creation/modification/deletion tool.

Tool usage pattern:
1. Add tool config to system message via `SystemContent.with_tools()` or specific methods like `with_browser_tool()`
2. After inference, check if last message has tool recipient
3. Call `tool.process(message)` to get response messages
4. Extend conversation with response and continue inference

### Responses API (`gpt_oss/responses_api/`)

FastAPI-based server implementing OpenAI Responses API:
- `api_server.py`: Core FastAPI app with streaming and non-streaming endpoints
- `inference/`: Backend adapters for triton, metal, ollama, vllm, transformers
- `events.py`: Server-sent event formatting
- `utils.py`: Conversation management and tool execution

The server handles tool execution internally (browser tool is integrated by default).

### Model Architecture Details

Both models use:
- Mixture of Experts (MoE) with 128 experts, 4 active per token
- MXFP4 quantization for MoE weights (stored as `tensor.blocks` + `tensor.scales`)
- Rotary embeddings with YaRN scaling and NTK-aware interpolation
- Sliding window attention (128 tokens)
- RMSNorm for normalization

Key configuration in `gpt_oss.torch.model.ModelConfig`.

### Precision and Quantization

MoE weights use MXFP4 format:
- `tensor.blocks`: FP4 values packed 2 per uint8
- `tensor.scales`: Block-wise scaling factors
- Block scaling along last dimension
- All other tensors in BF16
- Recommended activation precision: BF16

## Important Patterns

### Distributed Execution

PyTorch and Triton backends support distributed execution:
- Use `init_distributed()` from `gpt_oss.torch.utils`
- Broadcast user input across ranks in multi-GPU setups
- MoE supports tensor parallelism for the larger model

### Environment Variables

- `PYTHON_EXECUTION_BACKEND`: Controls python tool backend (`docker`, `dangerously_use_uv`, `dangerously_use_local_jupyter`)
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`: Required for Triton backend to avoid OOM
- `YDC_API_KEY`: Required for YouCom browser backend
- `EXA_API_KEY`: Required for Exa browser backend

### Sampling Parameters

Recommended settings:
- `temperature=1.0`
- `top_p=1.0`

### Reasoning Effort

Models support configurable reasoning effort levels (low, medium, high) via `ReasoningEffort` enum from openai-harmony.

## Contributing

Per README.md, this repository accepts bug fixes but not new features. New implementations should be contributed to `awesome-gpt-oss.md` instead.
