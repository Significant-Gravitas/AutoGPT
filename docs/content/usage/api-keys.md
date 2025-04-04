# API Key Management

AutoGPT provides a secure system for managing API keys used for various integrations.

## Adding API Keys

### Via Command Line
```bash
python -m autogpt add-api-key --service OPENAI --key YOUR_API_KEY --name "prod-key" --expires 2023-12-31
```

### Programmatically
```python
from autogpt.integrations import ApiKeyManager

key_manager = ApiKeyManager()
key_manager.add_key(
    service="OPENAI",
    key="sk-...",
    name="dev-key",
    expires="2023-12-31"  # Optional
)
```

## Key Attributes

- **Service**: The integration service (OPENAI, GITHUB, etc.)
- **Name**: Human-readable identifier (optional but recommended)
- **Expiration**: ISO-format date (YYYY-MM-DD) when key should rotate

## Storage Location

API keys are stored encrypted in:
```
~/.config/autogpt/integrations/
```

## Viewing Existing Keys

```bash
python -m autogpt list-api-keys
```

## Security Notes

- Keys are encrypted at rest using AES-256
- Never commit keys to version control
- Use environment variables for CI/CD systems