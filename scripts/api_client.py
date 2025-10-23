"""
Reusable OpenAI-compatible API client for gpt-oss-20b vLLM server
"""

from openai import OpenAI
from typing import List, Dict, Any, Optional, Iterator
import json


class GPTOSSClient:
    """Client for interacting with gpt-oss-20b via vLLM OpenAI-compatible API"""

    def __init__(
        self,
        base_url: str = "http://localhost:8000/v1",
        api_key: str = "not-needed",
        model: str = "/home/ye/ml-experiments/gpt-oss/gpt-oss-20b"
    ):
        """
        Initialize the client

        Args:
            base_url: Base URL of the vLLM server
            api_key: API key (not needed for local server, but required by OpenAI client)
            model: Model name/path
        """
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 2048,
        stream: bool = False,
        tools: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ):
        """
        Send a chat completion request

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature (1.0 recommended)
            max_tokens: Maximum tokens to generate
            stream: Whether to stream the response
            tools: Optional list of tool definitions for function calling
            **kwargs: Additional arguments to pass to the API

        Returns:
            Chat completion response or iterator if streaming
        """
        params = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }

        if tools:
            params["tools"] = tools

        return self.client.chat.completions.create(**params)

    def chat_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 1.0,
        max_tokens: int = 2048,
        **kwargs
    ) -> Iterator:
        """
        Stream a chat completion

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional arguments

        Yields:
            Response chunks
        """
        return self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
            **kwargs
        )

    def simple_query(
        self,
        prompt: str,
        system_message: Optional[str] = None,
        temperature: float = 1.0,
        max_tokens: int = 2048,
        stream: bool = False
    ):
        """
        Simple query interface

        Args:
            prompt: User prompt
            system_message: Optional system message
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stream: Whether to stream

        Returns:
            Response text or iterator if streaming
        """
        messages = []
        if system_message:
            messages.append({"role": "system", "content": system_message})
        messages.append({"role": "user", "content": prompt})

        response = self.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=stream
        )

        if stream:
            return response
        else:
            return response.choices[0].message.content


if __name__ == "__main__":
    # Example usage
    client = GPTOSSClient()

    # Simple query
    print("Testing simple query...")
    response = client.simple_query("What is the capital of France?")
    print(f"Response: {response}\n")

    # Streaming query
    print("Testing streaming query...")
    print("Response: ", end="", flush=True)
    for chunk in client.simple_query("Count from 1 to 5", stream=True):
        delta = chunk.choices[0].delta
        # gpt-oss models stream tokens in reasoning_content, not content
        content = delta.reasoning_content if hasattr(delta, 'reasoning_content') else delta.content
        if content:
            print(content, end="", flush=True)
    print("\n")
