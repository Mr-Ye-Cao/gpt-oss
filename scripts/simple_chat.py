#!/usr/bin/env python
"""
Simple interactive chat interface for gpt-oss-20b
"""

import sys
from api_client import GPTOSSClient


def main():
    print("=" * 80)
    print("GPT-OSS-20B Interactive Chat")
    print("=" * 80)
    print("Type 'quit' or 'exit' to end the conversation")
    print("Type 'clear' to start a new conversation")
    print("=" * 80)
    print()

    # Initialize client
    try:
        client = GPTOSSClient()
        print("✓ Connected to vLLM server at http://localhost:8000")
    except Exception as e:
        print(f"✗ Failed to connect to server: {e}")
        print("\nMake sure the vLLM server is running:")
        print("  ./scripts/start_server.sh")
        sys.exit(1)

    # Conversation history
    messages = []

    # System message
    system_message = {
        "role": "system",
        "content": "You are a helpful AI assistant powered by gpt-oss-20b. Be concise and helpful."
    }
    messages.append(system_message)

    print()

    while True:
        try:
            # Get user input
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ['quit', 'exit']:
                print("\nGoodbye!")
                break

            if user_input.lower() == 'clear':
                messages = [system_message]
                print("\n✓ Conversation cleared")
                continue

            # Add user message
            messages.append({"role": "user", "content": user_input})

            # Get response with streaming
            print("\nAssistant: ", end="", flush=True)

            full_response = ""
            for chunk in client.chat_stream(messages=messages):
                delta = chunk.choices[0].delta
                # gpt-oss models stream tokens in reasoning_content, not content
                content = delta.reasoning_content if hasattr(delta, 'reasoning_content') else delta.content
                if content:
                    print(content, end="", flush=True)
                    full_response += content

            print()  # New line after response

            # Add assistant response to history
            messages.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"\n\nError: {e}")
            if "Connection" in str(e):
                print("Make sure the vLLM server is still running.")
                break


if __name__ == "__main__":
    main()
