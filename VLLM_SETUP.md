# vLLM Setup for gpt-oss-20b on RTX 5090

Complete guide for running gpt-oss-20b with vLLM on your RTX 5090 GPU, optimized for single-user, low-latency inference.

## Quick Start

```bash
# 1. Start the server
./scripts/start_server.sh

# 2. In a new terminal, test with simple chat
conda activate gpt-oss-vllm
python scripts/simple_chat.py

# 3. Or test with tool calling
python scripts/tool_calling_demo.py
```

## Setup Summary

Your environment is configured with:

- **Conda Environment**: `gpt-oss-vllm` (Python 3.12)
- **Model**: gpt-oss-20b (~16GB VRAM usage of your 32GB)
- **Server**: OpenAI-compatible API at `http://localhost:8000`
- **Optimization**: Single-user, low-latency configuration

## Environment Setup

### Conda Environment

The setup created a conda environment named `gpt-oss-vllm` with:
- Python 3.12
- vLLM 0.11.0 with CUDA 12.x support
- openai-harmony (for harmony format)
- gpt-oss package (for tool implementations)
- All required dependencies

To activate:
```bash
conda activate gpt-oss-vllm
```

### Model Location

Model is downloaded at:
```
/home/ye/ml-experiments/gpt-oss/gpt-oss-20b/
```

Contains:
- `*.safetensors` - Model weights in HuggingFace format (for vLLM)
- `original/` - Original format weights (for Triton implementation)
- `metal/` - Metal format weights (for macOS)
- `config.json`, `tokenizer.json` - Model configuration

## Server Management

### Starting the Server

```bash
./scripts/start_server.sh
```

This script:
- Activates the conda environment automatically
- Checks if the model exists
- Starts vLLM with optimized settings
- Serves at `http://127.0.0.1:8000`

Server configuration (optimized for your single-user RTX 5090 setup):
- **max_model_len**: 8192 tokens (reasonable context, faster prefill)
- **gpu_memory_utilization**: 0.9 (90% of 32GB VRAM)
- **max_num_batched_tokens**: 2048 (small batch for low latency)
- **max_num_seqs**: 4 (limited concurrent requests for faster single-request response)

### Stopping the Server

```bash
./scripts/stop_server.sh
```

Or just press `Ctrl+C` in the server terminal.

### Checking Server Status

```bash
# Check if server is running
curl http://localhost:8000/health

# View API documentation
# Open in browser: http://localhost:8000/docs
```

## Client Scripts

### 1. Simple Chat (`scripts/simple_chat.py`)

Interactive chat interface with conversation history.

```bash
conda activate gpt-oss-vllm
python scripts/simple_chat.py
```

Features:
- Streaming responses for immediate feedback
- Conversation history maintained
- Commands: `clear`, `quit`/`exit`

### 2. Tool Calling Demo (`scripts/tool_calling_demo.py`)

Demonstrates function calling capabilities with example tools.

```bash
python scripts/tool_calling_demo.py
```

Included example tools:
- `get_weather(location, unit)` - Weather queries
- `calculate(expression)` - Math calculations

Try asking:
- "What's the weather in San Francisco?"
- "Calculate the square root of 144"
- "What's 25 * 17 + 100?"

### 3. API Client Library (`scripts/api_client.py`)

Reusable client class for building your own applications.

```python
from scripts.api_client import GPTOSSClient

client = GPTOSSClient()

# Simple query
response = client.simple_query("What is Python?")
print(response)

# Streaming
for chunk in client.simple_query("Count to 5", stream=True):
    print(chunk.choices[0].delta.content, end="", flush=True)

# Full chat with history
messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
]
response = client.chat(messages=messages)
print(response.choices[0].message.content)
```

### 4. Latency Benchmark (`scripts/benchmark_latency.py`)

Measure performance metrics.

```bash
python scripts/benchmark_latency.py
```

Benchmarks:
- Time to first token (TTFT)
- Tokens per second throughput
- End-to-end latency
- Both streaming and non-streaming modes

## Tool Calling Setup

The gpt-oss models support tool calling via the OpenAI function calling API.

### Browser Tool (Web Search)

To enable the browser tool, you need an API key from one of these providers:

**Option 1: You.com (Free tier available)**
1. Sign up at https://api.you.com/
2. Get your API key
3. Set environment variable:
   ```bash
   export YDC_API_KEY="your-api-key-here"
   ```

**Option 2: Exa (Semantic search)**
1. Sign up at https://exa.ai/
2. Get your API key
3. Set environment variable:
   ```bash
   export EXA_API_KEY="your-api-key-here"
   ```

### Python Tool (Code Execution)

Requires Docker for secure code execution in containers.

**Install Docker:**
```bash
# Ubuntu/Debian
sudo apt-get update
sudo apt-get install docker.io
sudo systemctl start docker
sudo usermod -aG docker $USER  # Add yourself to docker group
newgrp docker  # Or log out and back in

# Then pull Python image
./scripts/setup_docker.sh
```

### Environment Variables

Add to your `~/.bashrc` or `~/.zshrc`:

```bash
# Browser tool API key (choose one)
export YDC_API_KEY="your-you.com-api-key"
# or
export EXA_API_KEY="your-exa-api-key"

# Python execution backend (default: docker)
export PYTHON_EXECUTION_BACKEND="docker"

# Optional: Disable flashinfer sampler if you encounter issues
export VLLM_USE_FLASHINFER_SAMPLER="0"
```

## Customizing the Server

Edit `scripts/start_server.sh` to modify server parameters:

```bash
vllm serve "$MODEL_PATH" \
    --host 127.0.0.1 \          # Change to 0.0.0.0 for network access
    --port 8000 \                # Change port if needed
    --max-model-len 8192 \       # Increase for longer context (slower)
    --gpu-memory-utilization 0.9 \  # Adjust VRAM usage
    --max-num-batched-tokens 2048 \  # Decrease for lower latency
    --max-num-seqs 4 \           # Concurrent requests
    --disable-log-requests \     # Remove to see request logs
    --trust-remote-code
```

### Latency vs Throughput Tradeoffs

**For even lower latency (single conversation):**
```bash
--max-model-len 4096          # Smaller context
--max-num-batched-tokens 1024 # Smaller batches
--max-num-seqs 1              # One request at a time
```

**For longer conversations:**
```bash
--max-model-len 16384         # Larger context (slower prefill)
--max-num-batched-tokens 4096
```

**For multiple concurrent users:**
```bash
--max-num-seqs 16             # More concurrent requests
--max-num-batched-tokens 8192 # Larger batches
```

## Using with Other Applications

The server is OpenAI-compatible, so you can use it with any OpenAI-compatible client:

### Using with Official OpenAI Python Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Required but not used
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[
        {"role": "user", "content": "Hello!"}
    ]
)

print(response.choices[0].message.content)
```

### Using with LangChain

```python
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed",
    model="openai/gpt-oss-20b"
)

response = llm.invoke("What is the capital of France?")
print(response.content)
```

### Using with Curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "temperature": 1.0
  }'
```

## Troubleshooting

### Check Environment

```bash
./scripts/check_environment.sh
```

This verifies:
- Conda environment setup
- Package installations
- GPU availability
- Model download status
- Docker status

### Common Issues

**Server won't start - CUDA Out of Memory:**
```bash
# Reduce memory usage in start_server.sh
--gpu-memory-utilization 0.8  # Use less VRAM
--max-model-len 4096         # Smaller context
```

**Slow responses:**
- Check GPU usage: `nvidia-smi`
- Reduce `max-num-seqs` for single-user
- Reduce `max-model-len` for faster prefill

**Tool calling not working:**
- Check API keys are set: `echo $YDC_API_KEY`
- For Python tool, verify Docker: `docker ps`
- See tool setup section above

**Connection errors:**
```bash
# Make sure server is running
curl http://localhost:8000/health

# Check if port is in use
lsof -i :8000
```

### Logs and Debugging

Server logs show:
- Request processing
- Token generation speed
- Error messages

To enable request logging, remove `--disable-log-requests` from `start_server.sh`.

## Performance Expectations (RTX 5090)

Based on the 20B model with MXFP4 quantization:

- **VRAM Usage**: ~16GB (plenty of headroom on 32GB)
- **Throughput**: 40-80 tokens/second (varies by prompt length)
- **Time to First Token**: 50-200ms (depends on context length)
- **Max Context**: Up to 128K with RoPE scaling (use 8K-16K for best latency)

Your RTX 5090 should provide excellent performance for this model!

## Next Steps

1. **Test the setup**: Run `python scripts/simple_chat.py`
2. **Benchmark**: Run `python scripts/benchmark_latency.py` to see your performance
3. **Set up tools**: Configure API keys for browser tool
4. **Install Docker**: Run `./scripts/setup_docker.sh` for Python tool
5. **Build your app**: Use `scripts/api_client.py` as a starting point

## Additional Resources

- **vLLM Documentation**: https://docs.vllm.ai/
- **gpt-oss Model Card**: https://arxiv.org/abs/2508.10925
- **OpenAI Harmony Format**: https://github.com/openai/harmony
- **gpt-oss Repository**: https://github.com/openai/gpt-oss
- **OpenAI Cookbook**: https://cookbook.openai.com/topic/gpt-oss

## Support

For issues specific to this setup, check:
1. Run environment check: `./scripts/check_environment.sh`
2. Check vLLM logs in the server terminal
3. Refer to vLLM documentation for server configuration
4. Check gpt-oss repository issues for model-specific questions

Enjoy your personal AI inference setup! 🚀
