# Security Configuration

## API Key Authentication

This installation has been configured with API key authentication to prevent unauthorized access to the vLLM server.

### Configuration

API authentication is configured via the `.env` file:

```bash
# Private API key for server authentication
VLLM_API_KEY=sk-gptoss-....

# Client configuration - use the same key to authenticate
OPENAI_API_KEY=sk-gptoss-....
OPENAI_BASE_URL=http://127.0.0.1:8000/v1
OPENAI_MODEL=gpt-oss-20b
```

### Important Security Notes

1. **Keep the API key secret**: Never share your API key or commit it to public repositories
2. **The .env file is in .gitignore**: Ensure it stays there to prevent accidental commits
3. **Use environment-specific keys**: If deploying to different environments, use different API keys
4. **Rotate keys periodically**: Generate new API keys regularly for enhanced security

### Generating a New API Key

To generate a new API key, you can use:

```bash
# Generate a random API key
python3 -c "import secrets; print(f'sk-gptoss-{secrets.token_hex(32)}')"
```

Then update both `VLLM_API_KEY` and `OPENAI_API_KEY` in your `.env` file with the new key.

### Using the API

All API requests must include the API key in the `Authorization` header:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-gptoss-...." \
  -d '{
    "model": "gpt-oss-20b",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

The Python client scripts (`simple_chat.py`, `tool_calling_demo.py`, etc.) automatically read the API key from the `.env` file.

### Troubleshooting

**Error: "API key is required"**
- Make sure your `.env` file exists and contains `OPENAI_API_KEY`
- Verify the client script is running from the project root directory
- Check that python-dotenv is installed: `pip install python-dotenv`

**Error: "Unauthorized" (401)**
- The API key in your client doesn't match the server's `VLLM_API_KEY`
- Ensure both `VLLM_API_KEY` and `OPENAI_API_KEY` in `.env` have the same value

### Network Security

The server is configured to listen on `0.0.0.0:8000`, making it accessible from any network interface.

**Additional security recommendations:**

1. **Firewall rules**: Use firewall rules to restrict access to trusted IP addresses only
2. **VPN**: Consider running the server behind a VPN for remote access
3. **Reverse proxy**: Use nginx or similar with additional security headers and rate limiting
4. **HTTPS**: For production use, add TLS/SSL encryption (e.g., via nginx reverse proxy)

### Example Firewall Configuration (ufw)

```bash
# Allow access only from specific IP addresses
sudo ufw allow from 192.168.1.0/24 to any port 8000

# Or allow only from localhost (most secure for local-only use)
sudo ufw allow from 127.0.0.1 to any port 8000
```

## Monitoring

Monitor your server logs for suspicious activity:
- Multiple failed authentication attempts
- Unusual traffic patterns
- Requests from unexpected IP addresses

The vLLM server logs all requests (unless `--disable-log-requests` is removed from `start_server.sh`).
