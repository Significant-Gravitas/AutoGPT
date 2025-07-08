# Example Blocks Deployment Guide

## Overview

Example blocks are disabled by default in production environments to keep the production block list clean and focused on real functionality. This guide explains how to control the visibility of example blocks.

## Configuration

Example blocks are controlled by the `ENABLE_EXAMPLE_BLOCKS` setting:

- **Default**: `false` (example blocks are hidden)
- **Development**: Set to `true` to show example blocks

## How to Enable/Disable

### Method 1: Environment Variable (Recommended)

Add to your `.env` file:

```bash
# Enable example blocks in development
ENABLE_EXAMPLE_BLOCKS=true

# Disable example blocks in production (default)
ENABLE_EXAMPLE_BLOCKS=false
```

### Method 2: Configuration File

If you're using a `config.json` file:

```json
{
  "enable_example_blocks": true
}
```

## Implementation Details

The setting is checked in `backend/blocks/__init__.py` during the block loading process:

1. The `load_all_blocks()` function reads the `enable_example_blocks` setting from `Config`
2. If disabled (default), any Python files in the `examples/` directory are skipped
3. If enabled, example blocks are loaded normally

## Production Deployment

For production deployments:

1. **Do not set** `ENABLE_EXAMPLE_BLOCKS` in your production `.env` file (it defaults to `false`)
2. Or explicitly set `ENABLE_EXAMPLE_BLOCKS=false` for clarity
3. Example blocks will not appear in the block list or be available for use

## Development Environment

For local development:

1. Set `ENABLE_EXAMPLE_BLOCKS=true` in your `.env` file
2. Restart your backend server
3. Example blocks will be available for testing and demonstration

## Verification

To verify the setting is working:

```python
# Check current setting
from backend.util.settings import Config
config = Config()
print(f"Example blocks enabled: {config.enable_example_blocks}")

# Check loaded blocks
from backend.blocks import load_all_blocks
blocks = load_all_blocks()
example_blocks = [b for b in blocks.values() if 'examples' in b.__module__]
print(f"Example blocks loaded: {len(example_blocks)}")
```

## Security Note

Example blocks are for demonstration purposes only and may not follow production security standards. Always keep them disabled in production environments.