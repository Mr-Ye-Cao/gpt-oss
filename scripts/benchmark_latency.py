#!/usr/bin/env python
"""
Benchmark latency and throughput of gpt-oss-20b server
"""

import time
import sys
from api_client import GPTOSSClient
from typing import List
import statistics


def benchmark_single_request(client: GPTOSSClient, prompt: str, max_tokens: int = 100):
    """Benchmark a single request"""
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    response = client.chat(messages=messages, max_tokens=max_tokens)
    end_time = time.time()

    total_time = end_time - start_time
    generated_tokens = response.usage.completion_tokens
    tokens_per_second = generated_tokens / total_time if total_time > 0 else 0

    return {
        "total_time": total_time,
        "generated_tokens": generated_tokens,
        "tokens_per_second": tokens_per_second,
        "prompt_tokens": response.usage.prompt_tokens
    }


def benchmark_streaming(client: GPTOSSClient, prompt: str, max_tokens: int = 100):
    """Benchmark streaming response"""
    messages = [{"role": "user", "content": prompt}]

    start_time = time.time()
    first_token_time = None
    full_response = ""
    completion_tokens = 0
    chunk_count = 0
    content_chunks = 0

    # Request with stream_options to get usage data
    for chunk in client.chat(
        messages=messages,
        max_tokens=max_tokens,
        stream=True,
        stream_options={"include_usage": True}
    ):
        chunk_count += 1

        # Check for content - gpt-oss uses reasoning_content field
        if chunk.choices and len(chunk.choices) > 0:
            delta = chunk.choices[0].delta
            if delta:
                # gpt-oss models stream tokens in reasoning_content, not content
                token_text = delta.reasoning_content if hasattr(delta, 'reasoning_content') else delta.content
                if token_text:  # Check if we have actual text (not None or empty string)
                    if first_token_time is None:
                        first_token_time = time.time()
                    full_response += token_text
                    content_chunks += 1

        # Get usage from final chunk
        if hasattr(chunk, 'usage') and chunk.usage is not None:
            completion_tokens = chunk.usage.completion_tokens

    end_time = time.time()

    # Validate we received content
    if first_token_time is None:
        # No content received - this shouldn't happen
        print(f"\n    ⚠️  WARNING: No content received! Got {chunk_count} chunks but no reasoning_content")
        ttft = 0  # Mark as invalid
    else:
        ttft = first_token_time - start_time

    total_time = end_time - start_time
    tokens_per_second = completion_tokens / total_time if total_time > 0 and completion_tokens > 0 else 0

    return {
        "ttft": ttft,  # Time to first token
        "total_time": total_time,
        "generated_tokens": completion_tokens,
        "tokens_per_second": tokens_per_second,
        "response_length_chars": len(full_response),
        "chunk_count": chunk_count
    }


def main():
    print("=" * 80)
    print("GPT-OSS-20B Latency Benchmark")
    print("=" * 80)
    print()

    # Connect to server
    try:
        client = GPTOSSClient()
        print("✓ Connected to vLLM server at http://localhost:8000\n")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nMake sure the server is running: ./scripts/start_server.sh")
        sys.exit(1)

    # Test prompts
    test_prompts = [
        ("Short prompt", "What is 2+2?", 50),
        ("Medium prompt", "Explain the concept of neural networks in simple terms.", 150),
        ("Long generation", "Write a detailed explanation of how transformers work in AI.", 300),
    ]

    print("Running benchmarks...\n")

    # Non-streaming benchmarks
    print("=" * 80)
    print("NON-STREAMING BENCHMARKS")
    print("=" * 80)

    for name, prompt, max_tokens in test_prompts:
        print(f"\nTest: {name} ({max_tokens} max tokens)")
        print("-" * 40)

        results = []
        for i in range(3):
            print(f"  Run {i+1}/3...", end=" ", flush=True)
            result = benchmark_single_request(client, prompt, max_tokens)
            results.append(result)
            print(f"{result['tokens_per_second']:.1f} tok/s")

        # Calculate averages
        avg_time = statistics.mean([r["total_time"] for r in results])
        avg_tps = statistics.mean([r["tokens_per_second"] for r in results])
        avg_tokens = statistics.mean([r["generated_tokens"] for r in results])

        print(f"\n  Average:")
        print(f"    Total time: {avg_time:.3f}s")
        print(f"    Generated tokens: {avg_tokens:.1f}")
        print(f"    Throughput: {avg_tps:.1f} tokens/second")

    # Streaming benchmarks
    print("\n" + "=" * 80)
    print("STREAMING BENCHMARKS")
    print("=" * 80)

    for name, prompt, max_tokens in test_prompts:
        print(f"\nTest: {name} ({max_tokens} max tokens)")
        print("-" * 40)

        results = []
        for i in range(3):
            print(f"  Run {i+1}/3...", end=" ", flush=True)
            result = benchmark_streaming(client, prompt, max_tokens)
            results.append(result)

            # Show warning for suspicious TTFT
            if result['ttft'] < 0.001:
                print(f"TTFT: {result['ttft']:.3f}s ⚠️, {result['tokens_per_second']:.1f} tok/s ({result['chunk_count']} chunks)")
            else:
                print(f"TTFT: {result['ttft']:.3f}s, {result['tokens_per_second']:.1f} tok/s")

        # Calculate averages (filter out invalid TTFT measurements)
        valid_ttft = [r["ttft"] for r in results if r["ttft"] > 0.001]
        avg_ttft = statistics.mean(valid_ttft) if valid_ttft else 0
        avg_time = statistics.mean([r["total_time"] for r in results])
        avg_tps = statistics.mean([r["tokens_per_second"] for r in results])
        avg_tokens = statistics.mean([r["generated_tokens"] for r in results])

        print(f"\n  Average:")
        if len(valid_ttft) < len(results):
            print(f"    Time to first token (TTFT): {avg_ttft:.3f}s ({len(valid_ttft)}/{len(results)} valid measurements)")
        else:
            print(f"    Time to first token (TTFT): {avg_ttft:.3f}s")
        print(f"    Total time: {avg_time:.3f}s")
        print(f"    Generated tokens: {avg_tokens:.1f}")
        print(f"    Throughput: {avg_tps:.1f} tokens/second")

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
