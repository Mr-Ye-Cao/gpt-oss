# Scripts for gpt-oss-20b vLLM Setup

Quick reference for all scripts in this directory.

## Server Management

### Start Server
```bash
./start_server.sh
```
Starts the vLLM server with optimized settings for RTX 5090.
- Activates conda environment automatically
- Server at http://localhost:8000
- Optimized for single-user, low-latency inference

### Stop Server
```bash
./stop_server.sh
```
Stops the running vLLM server.

## Client Scripts

All client scripts require the server to be running and the conda environment to be activated:
```bash
conda activate gpt-oss-vllm
```

### Simple Chat
```bash
python simple_chat.py
```
Interactive chat with conversation history and streaming responses.

### Tool Calling Demo
```bash
python tool_calling_demo.py
```
Demonstrates function calling with example tools (weather, calculator).

### API Client Library
```bash
python api_client.py
```
Reusable client class - use this to build your own applications.
See `VLLM_SETUP.md` for usage examples.

### Benchmark
```bash
python benchmark_latency.py
```
Measures latency and throughput metrics.

## Setup & Maintenance

### Check Environment
```bash
./check_environment.sh
```
Verifies that everything is installed and configured correctly.

### Setup Docker (Optional)
```bash
./setup_docker.sh
```
Installs Docker support for Python tool execution.
Only needed if you want the model to execute Python code.

## Quick Start

```bash
# 1. Check everything is set up
./check_environment.sh

# 2. Start the server
./start_server.sh

# 3. In a new terminal, test it out
conda activate gpt-oss-vllm
python simple_chat.py
```

For detailed documentation, see `../VLLM_SETUP.md`.
