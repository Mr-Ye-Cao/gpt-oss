#!/usr/bin/env python
"""
Tool calling demonstration for gpt-oss-20b (FIXED VERSION)
Works around the issue where gpt-oss outputs tool calls in reasoning_content
instead of using structured tool_calls
"""

import json
import sys
import re
from api_client import GPTOSSClient


# Example tools that the model can call
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Get the current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA"
                    },
                    "unit": {
                        "type": "string",
                        "enum": ["celsius", "fahrenheit"],
                        "description": "The temperature unit"
                    }
                },
                "required": ["location"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "calculate",
            "description": "Perform a mathematical calculation",
            "parameters": {
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "The mathematical expression to evaluate, e.g. '2 + 2' or 'sqrt(16)'"
                    }
                },
                "required": ["expression"]
            }
        }
    }
]


# Tool implementations
def get_weather(location: str, unit: str = "fahrenheit") -> str:
    """Simulated weather API"""
    return f"The weather in {location} is 72°{unit[0].upper()}, partly cloudy"


def calculate(expression: str) -> str:
    """Safe calculator"""
    try:
        import math
        allowed_names = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
        allowed_names.update({"abs": abs, "round": round})
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return str(result)
    except Exception as e:
        return f"Error calculating: {e}"


# Map function names to implementations
FUNCTION_MAP = {
    "get_weather": get_weather,
    "calculate": calculate
}


def parse_tool_calls_from_reasoning(reasoning_content: str):
    """
    Parse tool calls from reasoning_content where gpt-oss outputs them

    Looks for patterns like:
    - {"function_name": {...}}
    - function_name({...})
    - JSON objects matching our tool schemas
    """
    tool_calls = []

    # Try to find JSON objects in the reasoning content
    # Pattern: Look for JSON-like structures
    json_pattern = r'\{[^}]+\}'
    matches = re.finditer(json_pattern, reasoning_content)

    for match in matches:
        try:
            json_obj = json.loads(match.group())

            # Check if this JSON matches any of our tool schemas
            for tool_name, tool_func in FUNCTION_MAP.items():
                # Simple heuristic: if the JSON has keys that match the tool's parameters
                if tool_name == "calculate" and "expression" in json_obj:
                    tool_calls.append({
                        "function": tool_name,
                        "arguments": json_obj
                    })
                elif tool_name == "get_weather" and "location" in json_obj:
                    tool_calls.append({
                        "function": tool_name,
                        "arguments": json_obj
                    })
        except json.JSONDecodeError:
            continue

    return tool_calls


def execute_tool_call(tool_call_dict):
    """Execute a tool call from parsed reasoning content"""
    function_name = tool_call_dict["function"]
    function_args = tool_call_dict["arguments"]

    print(f"\n  🔧 Calling {function_name}({function_args})")

    if function_name in FUNCTION_MAP:
        result = FUNCTION_MAP[function_name](**function_args)
        return result
    else:
        return f"Error: Unknown function {function_name}"


def main():
    print("=" * 80)
    print("GPT-OSS-20B Tool Calling Demo (FIXED VERSION)")
    print("=" * 80)
    print("\nAvailable tools:")
    for tool in TOOLS:
        func = tool["function"]
        print(f"  - {func['name']}: {func['description']}")
    print("\nNOTE: This version parses tool calls from reasoning_content")
    print("Type 'quit' to exit")
    print("=" * 80)
    print()

    # Initialize client
    try:
        client = GPTOSSClient()
        print("✓ Connected to vLLM server")
    except Exception as e:
        print(f"✗ Failed to connect: {e}")
        print("\nMake sure the server is running: ./scripts/start_server.sh")
        sys.exit(1)

    # Conversation with tools - use system prompt to encourage tool use
    messages = [
        {
            "role": "system",
            "content": """You are a helpful assistant with access to these tools:
- calculate: Perform mathematical calculations by returning JSON like {"expression": "2+2"}
- get_weather: Get weather by returning JSON like {"location": "San Francisco, CA"}

When you want to use a tool, output ONLY the JSON for that tool. Do not include any other text."""
        }
    ]

    print()

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input or user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break

            messages.append({"role": "user", "content": user_input})

            # Send request with tools
            response = client.chat(
                messages=messages,
                tools=TOOLS,
                temperature=1.0,
                stream=True,
                stream_options={"include_usage": True}
            )

            # Collect full response from streaming
            full_reasoning = ""
            full_content = ""

            print("\nAssistant: ", end="", flush=True)

            for chunk in response:
                if chunk.choices and len(chunk.choices) > 0:
                    delta = chunk.choices[0].delta

                    # Collect reasoning_content
                    if hasattr(delta, 'reasoning_content') and delta.reasoning_content:
                        full_reasoning += delta.reasoning_content

                    # Collect regular content
                    if delta.content:
                        full_content += delta.content
                        print(delta.content, end="", flush=True)

            print()  # newline after response

            # Check if there are tool calls in the reasoning content
            tool_calls = parse_tool_calls_from_reasoning(full_reasoning)

            if tool_calls:
                print(f"\n[Detected {len(tool_calls)} tool call(s) in reasoning]")

                # Execute each tool call
                tool_results = []
                for tool_call in tool_calls:
                    result = execute_tool_call(tool_call)
                    tool_results.append(result)
                    print(f"  ✓ Result: {result}")

                # Add the assistant message with reasoning
                messages.append({
                    "role": "assistant",
                    "content": full_content if full_content else full_reasoning
                })

                # Add tool results to the conversation
                tool_result_msg = "Tool results:\n" + "\n".join(
                    f"- {tc['function']}: {res}"
                    for tc, res in zip(tool_calls, tool_results)
                )
                messages.append({
                    "role": "user",
                    "content": f"{tool_result_msg}\n\nPlease provide the final answer based on these results."
                })

                # Get final response
                print("\nAssistant (final): ", end="", flush=True)
                final_response = client.chat(messages=messages, stream=True)

                final_content = ""
                for chunk in final_response:
                    if chunk.choices and len(chunk.choices) > 0:
                        delta = chunk.choices[0].delta
                        content = delta.reasoning_content if hasattr(delta, 'reasoning_content') else delta.content
                        if content:
                            print(content, end="", flush=True)
                            final_content += content

                print()
                messages.append({"role": "assistant", "content": final_content})
            else:
                # No tool calls detected, just add the response
                messages.append({
                    "role": "assistant",
                    "content": full_content if full_content else full_reasoning
                })

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
