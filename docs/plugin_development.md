# Plugin Development Guide

This guide explains how to create, test, and distribute plugins for Auto-GPT.

## Overview

Auto-GPT supports two types of plugins:
1. Custom Python plugins (ZIP-based)
2. OpenAI plugins (API-based)

## Creating a Custom Plugin

### Basic Structure

Your plugin should be structured as follows:

```
my_plugin/
├── __init__.py           # Plugin entry point
├── manifest.json         # Plugin metadata
└── ...                  # Additional plugin files
```

### Plugin Template

All plugins must inherit from `AutoGPTPluginTemplate`. Here's a basic example:

```python
from typing import Any, Dict, List, Optional, Tuple
from auto_gpt_plugin_template import AutoGPTPluginTemplate

class MyPlugin(AutoGPTPluginTemplate):
    def __init__(self):
        super().__init__()
        self._name = "My Plugin"
        self._version = "0.1.0"
        self._description = "Description of what my plugin does"

    def can_handle_post_prompt(self) -> bool:
        """Return True if the plugin can handle post-prompt events"""
        return True

    def post_prompt(self, prompt: str) -> str:
        """Handle post-prompt events"""
        return prompt

    def can_handle_on_response(self) -> bool:
        """Return True if the plugin can handle on-response events"""
        return True

    def on_response(self, response: str) -> str:
        """Handle on-response events"""
        return response
```

### Available Plugin Hooks

Plugins can implement these lifecycle hooks:

1. Prompt Handling:
   - `can_handle_post_prompt()` / `post_prompt()`
   - `can_handle_on_response()` / `on_response()`

2. Planning Phase:
   - `can_handle_on_planning()` / `on_planning()`
   - `can_handle_post_planning()` / `post_planning()`

3. Instruction Phase:
   - `can_handle_pre_instruction()` / `pre_instruction()`
   - `can_handle_on_instruction()` / `on_instruction()`
   - `can_handle_post_instruction()` / `post_instruction()`

4. Command Phase:
   - `can_handle_pre_command()` / `pre_command()`
   - `can_handle_post_command()` / `post_command()`

5. Chat Completion:
   - `can_handle_chat_completion()` / `handle_chat_completion()`

### Packaging Your Plugin

1. Create a ZIP file containing your plugin directory:
   ```bash
   zip -r my_plugin.zip my_plugin/
   ```

2. Place the ZIP file in Auto-GPT's `plugins` directory.

## Using OpenAI Plugins

Auto-GPT can also use OpenAI-compatible plugins:

1. Add the plugin URL to your `.env` file:
   ```
   PLUGINS_OPENAI=["https://your-plugin-url.com"]
   ```

2. The plugin must provide:
   - A manifest at `/.well-known/ai-plugin.json`
   - An OpenAPI specification
   - Implemented endpoints

## Plugin Security

### Allowlist/Denylist

Control plugin loading with these `.env` settings:

```
PLUGINS_ALLOWLIST=["plugin1","plugin2"]  # Only these plugins can load
PLUGINS_DENYLIST=["plugin3"]            # These plugins cannot load
```

If a plugin isn't in either list, Auto-GPT will prompt for approval.

### Best Practices

1. Input Validation
   - Validate all inputs
   - Sanitize file paths
   - Check permissions

2. Resource Management
   - Clean up resources
   - Handle errors gracefully
   - Implement timeouts

3. Documentation
   - Document requirements
   - Explain configuration
   - Provide examples

## Testing Your Plugin

1. Create test files in your plugin directory:
   ```python
   def test_my_plugin():
       plugin = MyPlugin()
       result = plugin.post_prompt("test")
       assert result == "expected"
   ```

2. Run tests with pytest:
   ```bash
   pytest my_plugin/tests/
   ```

## Distribution

1. Create a GitHub repository for your plugin
2. Include:
   - README with installation instructions
   - Requirements.txt
   - License
   - Example usage

3. Submit to the Auto-GPT plugin registry (optional)

## Debugging

Enable debug logging in Auto-GPT:
```bash
python -m autogpt --debug
```

Plugin logs will appear in `./output/logs`. 