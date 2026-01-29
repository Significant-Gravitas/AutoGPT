"""Input validation functions for settings."""

from __future__ import annotations

import re
from typing import Any, Callable


def validate_api_key_format(key_name: str, value: str) -> tuple[bool, str]:
    """Validate API key format based on known patterns.

    Args:
        key_name: The name of the key (e.g., "OPENAI_API_KEY")
        value: The key value to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not value:
        return True, ""  # Empty is allowed (will be skipped)

    patterns = {
        "OPENAI_API_KEY": (
            r"^sk-[a-zA-Z0-9-_]{20,}$",
            "OpenAI keys should start with 'sk-' followed by alphanumeric characters",
        ),
        "ANTHROPIC_API_KEY": (
            r"^sk-ant-api03-[a-zA-Z0-9-_]{80,}$",
            "Anthropic keys should start with 'sk-ant-api03-'",
        ),
        "GROQ_API_KEY": (
            r"^gsk_[a-zA-Z0-9]{48,}$",
            "Groq keys should start with 'gsk_'",
        ),
        "TAVILY_API_KEY": (
            r"^tvly-[a-zA-Z0-9-_]{20,}$",
            "Tavily keys should start with 'tvly-'",
        ),
        "GITHUB_API_KEY": (
            r"^(ghp_[a-zA-Z0-9]{36}|github_pat_[a-zA-Z0-9_]{80,})$",
            "GitHub tokens should start with 'ghp_' or 'github_pat_'",
        ),
    }

    if key_name not in patterns:
        return True, ""  # No pattern to validate against

    pattern, error_msg = patterns[key_name]
    if re.match(pattern, value):
        return True, ""

    return False, error_msg


def validate_model_name(model_name: str) -> tuple[bool, str]:
    """Validate that a model name is in a known format.

    Args:
        model_name: The model name to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not model_name:
        return True, ""

    # Known model prefixes
    valid_prefixes = [
        "gpt-3.5",
        "gpt-4",
        "gpt-5",
        "o1",
        "o3",
        "o4",
        "claude-",
        "mixtral",
        "gemma",
        "llama",
    ]

    model_lower = model_name.lower()
    for prefix in valid_prefixes:
        if model_lower.startswith(prefix):
            return True, ""

    # Also allow full model names from enums
    # Just warn, don't block
    return True, f"Note: '{model_name}' is not a recognized model name"


def validate_port(port: int | str) -> tuple[bool, str]:
    """Validate a port number.

    Args:
        port: The port number to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        port_num = int(port)
    except (ValueError, TypeError):
        return False, "Port must be a number"

    if port_num < 1 or port_num > 65535:
        return False, "Port must be between 1 and 65535"

    if port_num < 1024:
        return True, "Note: Ports below 1024 typically require root privileges"

    return True, ""


def validate_url(url: str) -> tuple[bool, str]:
    """Validate a URL format.

    Args:
        url: The URL to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if not url:
        return True, ""

    # Basic URL pattern
    pattern = r"^https?://[a-zA-Z0-9.-]+(:[0-9]+)?(/.*)?$"
    if re.match(pattern, url):
        return True, ""

    return False, "Invalid URL format (should start with http:// or https://)"


def validate_log_level(level: str) -> tuple[bool, str]:
    """Validate a log level.

    Args:
        level: The log level to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    if level.upper() in valid_levels:
        return True, ""

    return False, f"Log level must be one of: {', '.join(valid_levels)}"


def validate_storage_backend(backend: str) -> tuple[bool, str]:
    """Validate a storage backend.

    Args:
        backend: The storage backend to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    valid_backends = ["local", "gcs", "s3"]

    if backend.lower() in valid_backends:
        return True, ""

    return False, f"Storage backend must be one of: {', '.join(valid_backends)}"


def validate_temperature(temp: float | str) -> tuple[bool, str]:
    """Validate a temperature value.

    Args:
        temp: The temperature to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        temp_num = float(temp)
    except (ValueError, TypeError):
        return False, "Temperature must be a number"

    if temp_num < 0 or temp_num > 2:
        return False, "Temperature should be between 0 and 2"

    return True, ""


# Mapping of env var names to validator functions
VALIDATORS: dict[str, Callable[[Any], tuple[bool, str]]] = {
    "OPENAI_API_KEY": lambda v: validate_api_key_format("OPENAI_API_KEY", v),
    "ANTHROPIC_API_KEY": lambda v: validate_api_key_format("ANTHROPIC_API_KEY", v),
    "GROQ_API_KEY": lambda v: validate_api_key_format("GROQ_API_KEY", v),
    "TAVILY_API_KEY": lambda v: validate_api_key_format("TAVILY_API_KEY", v),
    "GITHUB_API_KEY": lambda v: validate_api_key_format("GITHUB_API_KEY", v),
    "SMART_LLM": validate_model_name,
    "FAST_LLM": validate_model_name,
    "AP_SERVER_PORT": validate_port,
    "OPENAI_API_BASE_URL": validate_url,
    "ANTHROPIC_API_BASE_URL": validate_url,
    "GROQ_API_BASE_URL": validate_url,
    "S3_ENDPOINT_URL": validate_url,
    "LOG_LEVEL": validate_log_level,
    "FILE_STORAGE_BACKEND": validate_storage_backend,
    "TEMPERATURE": validate_temperature,
}


def validate_setting(env_var: str, value: str) -> tuple[bool, str]:
    """Validate a setting value.

    Args:
        env_var: The environment variable name
        value: The value to validate

    Returns:
        Tuple of (is_valid, error_message)
    """
    if env_var in VALIDATORS:
        return VALIDATORS[env_var](value)
    return True, ""
