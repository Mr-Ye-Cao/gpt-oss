# ✅ vLLM Setup Complete for gpt-oss-20b!

Your RTX 5090 is ready to run gpt-oss-20b with vLLM. Everything has been set up and optimized for single-user, low-latency inference.

## 🎉 What's Been Set Up

### Environment
- ✅ Conda environment: `gpt-oss-vllm` (Python 3.12)
- ✅ vLLM 0.11.0 with CUDA 12.x support
- ✅ openai-harmony for Harmony format
- ✅ gpt-oss package with tool implementations
- ✅ All dependencies installed

### Model
- ✅ gpt-oss-20b downloaded (21B parameters, ~16GB VRAM)
- ✅ Located at: `/home/ye/ml-experiments/gpt-oss/gpt-oss-20b/`
- ✅ All formats available (HuggingFace, original, metal)

### Hardware Verified
- ✅ NVIDIA GeForce RTX 5090 (32GB VRAM)
- ✅ Driver version: 580.95.05
- ✅ Plenty of VRAM headroom (using ~16GB of 32GB)

### Scripts Created
All scripts are in `/home/ye/ml-experiments/gpt-oss/scripts/`:

**Server Management:**
- `start_server.sh` - Start vLLM server
- `stop_server.sh` - Stop vLLM server
- `start_vllm_server.py` - Python server wrapper

**Client Applications:**
- `simple_chat.py` - Interactive chat interface
- `tool_calling_demo.py` - Function calling examples
- `api_client.py` - Reusable API client library
- `benchmark_latency.py` - Performance benchmarking

**Utilities:**
- `check_environment.sh` - Verify setup
- `setup_docker.sh` - Install Docker for Python tool
- `README.md` - Quick reference

### Documentation
- ✅ `VLLM_SETUP.md` - Complete setup and usage guide
- ✅ `CLAUDE.md` - Updated with vLLM setup instructions
- ✅ `scripts/README.md` - Script reference

## 🚀 Quick Start (Next Steps)

### 1. Start the Server

```bash
cd /home/ye/ml-experiments/gpt-oss
./scripts/start_server.sh
```

This will:
- Activate the conda environment
- Check the model exists
- Start vLLM on http://localhost:8000
- Show you the configuration

**Wait for:** "Uvicorn running on http://127.0.0.1:8000"

### 2. Test with Simple Chat (New Terminal)

```bash
cd /home/ye/ml-experiments/gpt-oss
conda activate gpt-oss-vllm
python scripts/simple_chat.py
```

Try asking:
- "What is machine learning?"
- "Explain transformers in simple terms"
- "Write a Python function to calculate fibonacci"

Type `quit` to exit.

### 3. Test Tool Calling

```bash
python scripts/tool_calling_demo.py
```

Try:
- "What's 25 * 17 + 100?"
- "Calculate the square root of 144"
- "What's the weather in San Francisco?" (will simulate - needs API key for real data)

### 4. Benchmark Performance

```bash
python scripts/benchmark_latency.py
```

This will measure your RTX 5090's performance with gpt-oss-20b.

## 📊 Expected Performance

On your RTX 5090, you should see approximately:

- **VRAM Usage**: ~16GB (50% of available)
- **Throughput**: 50-100 tokens/second
- **Time to First Token**: 50-200ms
- **Latency**: Very low (optimized for single-user)

Your actual performance may be better given the RTX 5090's capabilities!

## 🔧 Optional: Enable Advanced Tools

### Browser Tool (Web Search)

Get a free API key from You.com:
1. Sign up at https://api.you.com/
2. Get your API key
3. Add to your shell config:
   ```bash
   echo 'export YDC_API_KEY="your-key-here"' >> ~/.bashrc
   source ~/.bashrc
   ```

### Python Execution Tool

Install Docker for secure code execution:
```bash
./scripts/setup_docker.sh
```

## 📖 Full Documentation

For complete details, see:
- **VLLM_SETUP.md** - Complete guide with all options
- **scripts/README.md** - Script reference
- **CLAUDE.md** - Repository guide

## 🛠️ Customization

### Change Server Settings

Edit `scripts/start_server.sh` to customize:
- Port (default: 8000)
- Context length (default: 8192)
- Memory usage (default: 90%)
- Concurrent requests (default: 4)

### Network Access

To allow access from other machines on your network:
```bash
# In start_server.sh, change:
--host 127.0.0.1  # to:
--host 0.0.0.0
```

## 🐛 Troubleshooting

If something doesn't work:

```bash
# 1. Check everything is set up
./scripts/check_environment.sh

# 2. Check GPU
nvidia-smi

# 3. Test server health
curl http://localhost:8000/health

# 4. Check if port is in use
lsof -i :8000
```

See VLLM_SETUP.md "Troubleshooting" section for more help.

## 📝 Usage Examples

### Python API

```python
from scripts.api_client import GPTOSSClient

client = GPTOSSClient()

# Simple query
response = client.simple_query("Explain neural networks")
print(response)

# Streaming
for chunk in client.simple_query("Count to 10", stream=True):
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
```

### OpenAI Client

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Hello!"}]
)

print(response.choices[0].message.content)
```

### Curl

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello!"}],
    "stream": true
  }'
```

## 🎯 Optimization Tips

For your single-user setup on RTX 5090:

**Current configuration** (already set):
- Small batches for low latency ✅
- Moderate context (8K) for fast prefill ✅
- 90% GPU memory utilization ✅

**To go even faster:**
- Reduce `--max-model-len` to 4096
- Set `--max-num-seqs` to 1
- Reduce `--max-num-batched-tokens` to 1024

**For longer conversations:**
- Increase `--max-model-len` to 16384 or 32768
- Accept slightly higher latency

## 🎊 You're All Set!

Your personal AI inference server is ready to go. The setup is optimized for:
- ✅ Low latency responses
- ✅ Single user (you)
- ✅ Maximum throughput on RTX 5090
- ✅ Flexible API access
- ✅ Tool calling support

Enjoy your local AI! 🚀

---

**Need help?** Check VLLM_SETUP.md or run `./scripts/check_environment.sh`
