"""Tests for chat configuration and system prompt loading."""

import os
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from backend.server.v2.chat.config import (
    ChatConfig,
    get_config,
    reset_config,
    set_config,
)


class TestChatConfig:
    """Test cases for ChatConfig class."""

    def test_default_config(self) -> None:
        """Test default configuration values."""
        config = ChatConfig()

        assert config.model == "gpt-4o"
        assert config.system_prompt_path == "prompts/chat_system.md"
        assert config.max_context_messages == 50
        assert config.stream_timeout == 300
        assert config.cache_client is True

    def test_config_with_env_vars(self) -> None:
        """Test configuration with environment variables."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "test-api-key",
                "OPENAI_BASE_URL": "https://test.api/v1",
            },
        ):
            config = ChatConfig()

            assert config.api_key == "test-api-key"
            assert config.base_url == "https://test.api/v1"

    def test_config_with_openrouter(self) -> None:
        """Test configuration for OpenRouter."""
        with patch.dict(
            os.environ,
            {
                "OPENROUTER_API_KEY": "or-test-key",
                "USE_OPENROUTER": "true",
            },
        ):
            config = ChatConfig()

            assert config.api_key == "or-test-key"
            assert config.base_url == "https://openrouter.ai/api/v1"

    def test_explicit_config_values(self) -> None:
        """Test explicit configuration values override environment."""
        with patch.dict(
            os.environ,
            {
                "OPENAI_API_KEY": "env-key",
            },
        ):
            config = ChatConfig(
                api_key="explicit-key",
                model="gpt-3.5-turbo",
                max_context_messages=100,
            )

            assert config.api_key == "explicit-key"
            assert config.model == "gpt-3.5-turbo"
            assert config.max_context_messages == 100

    def test_get_system_prompt_from_file(self) -> None:
        """Test loading system prompt from markdown file."""
        # Create a temporary directory and file
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_dir = Path(tmpdir) / "prompts"
            prompt_dir.mkdir()
            prompt_file = prompt_dir / "chat_system.md"

            # Write test content
            test_prompt = "# Test System Prompt\n\nYou are a test assistant."
            prompt_file.write_text(test_prompt)

            # Create config with custom path
            config = ChatConfig()

            # Monkey-patch the path resolution
            with patch.object(
                Path,
                "__new__",
                lambda cls, path: (
                    prompt_file if "chat_system.md" in str(path) else Path(path)
                ),
            ):
                prompt = config.get_system_prompt()

            assert prompt == test_prompt

    def test_get_system_prompt_with_variables(self) -> None:
        """Test system prompt with variable substitution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_dir = Path(tmpdir) / "prompts"
            prompt_dir.mkdir()
            prompt_file = prompt_dir / "chat_system.md"

            # Write template with variables
            template = "Hello {user_name}, you are in {location}."
            prompt_file.write_text(template)

            config = ChatConfig()

            # Monkey-patch the path resolution
            with patch.object(
                Path,
                "__new__",
                lambda cls, path: (
                    prompt_file if "chat_system.md" in str(path) else Path(path)
                ),
            ):
                prompt = config.get_system_prompt(
                    user_name="Alice", location="Wonderland"
                )

            assert prompt == "Hello Alice, you are in Wonderland."

    def test_get_system_prompt_jinja2_template(self) -> None:
        """Test loading system prompt from Jinja2 template."""
        with tempfile.TemporaryDirectory() as tmpdir:
            prompt_dir = Path(tmpdir) / "prompts"
            prompt_dir.mkdir()
            prompt_file = prompt_dir / "chat_system.md.j2"

            # Write Jinja2 template
            template = """# {{ title }}
{% if show_greeting %}
Hello {{ user }}!
{% endif %}
Your role: {{ role | upper }}"""
            prompt_file.write_text(template)

            config = ChatConfig()

            # Monkey-patch the path resolution
            with patch.object(
                Path,
                "__new__",
                lambda cls, path: (
                    prompt_file if "chat_system.md.j2" in str(path) else Path(path)
                ),
            ):
                # Test with Jinja2 available
                try:
                    import jinja2  # noqa: F401

                    prompt = config.get_system_prompt(
                        title="Assistant",
                        show_greeting=True,
                        user="User",
                        role="helper",
                    )
                    assert "# Assistant" in prompt
                    assert "Hello User!" in prompt
                    assert "Your role: HELPER" in prompt
                except ImportError:
                    # If Jinja2 not installed, it should return raw template
                    prompt = config.get_system_prompt()
                    assert "{{ title }}" in prompt

    def test_get_system_prompt_fallback(self) -> None:
        """Test fallback to default prompt when file not found."""
        config = ChatConfig(system_prompt_path="nonexistent/path.md")

        # Should return default prompt
        prompt = config.get_system_prompt()

        assert "AutoGPT Agent Setup Assistant" in prompt
        assert "UNDERSTAND THE USER'S PROBLEM" in prompt
        assert "DISCOVER SUITABLE AGENTS" in prompt

    def test_system_prompt_file_exists(self) -> None:
        """Test that the actual system prompt file exists."""
        config = ChatConfig()

        # Check if the file actually exists in the codebase
        module_dir = Path(__file__).parent
        prompt_path = module_dir / config.system_prompt_path

        assert prompt_path.exists(), f"System prompt file not found at {prompt_path}"

        # Load and verify content
        prompt = config.get_system_prompt()
        assert len(prompt) > 0
        assert "AutoGPT" in prompt


class TestConfigManagement:
    """Test cases for config management functions."""

    def test_get_config_singleton(self) -> None:
        """Test that get_config returns singleton."""
        reset_config()  # Start fresh

        config1 = get_config()
        config2 = get_config()

        assert config1 is config2

    def test_set_config(self) -> None:
        """Test setting custom configuration."""
        reset_config()  # Start fresh

        custom_config = ChatConfig(model="custom-model")
        set_config(custom_config)

        retrieved = get_config()
        assert retrieved is custom_config
        assert retrieved.model == "custom-model"

    def test_reset_config(self) -> None:
        """Test resetting configuration."""
        # Set a custom config
        custom_config = ChatConfig(model="custom-model")
        set_config(custom_config)

        # Reset
        reset_config()

        # Should get new default config
        new_config = get_config()
        assert new_config is not custom_config
        assert new_config.model == "gpt-4o"  # Default value


class TestConfigIntegration:
    """Integration tests for config with other components."""

    @pytest.mark.asyncio
    async def test_config_with_chat_module(self) -> None:
        """Test that chat module uses config correctly."""
        from backend.server.v2.chat.chat import get_openai_client

        # Set custom config
        reset_config()
        custom_config = ChatConfig(
            api_key="test-key-123",
            base_url="https://test.example.com/v1",
            cache_client=False,
        )
        set_config(custom_config)

        # Mock AsyncOpenAI
        with patch("backend.server.v2.chat.chat.AsyncOpenAI") as mock_client:
            instance = mock_client.return_value

            # Get client
            client = get_openai_client()

            # Verify client was created with config values
            mock_client.assert_called_once_with(
                api_key="test-key-123",
                base_url="https://test.example.com/v1",
            )

            assert client is instance

    def test_prompt_loading_performance(self) -> None:
        """Test that prompt loading is reasonably fast."""
        import time

        config = ChatConfig()

        # Measure time to load prompt
        start = time.time()
        prompt = config.get_system_prompt()
        duration = time.time() - start

        # Should be fast (less than 100ms)
        assert duration < 0.1
        assert len(prompt) > 0
