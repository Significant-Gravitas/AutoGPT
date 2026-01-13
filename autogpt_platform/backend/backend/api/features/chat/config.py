"""Configuration management for chat system."""

import os

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class ChatConfig(BaseSettings):
    """Configuration for the chat system."""

    # OpenAI API Configuration
    model: str = Field(
        default="anthropic/claude-opus-4.5", description="Default model to use"
    )
    title_model: str = Field(
        default="openai/gpt-4o-mini",
        description="Model to use for generating session titles (should be fast/cheap)",
    )
    api_key: str | None = Field(default=None, description="OpenAI API key")
    base_url: str | None = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for API (e.g., for OpenRouter)",
    )

    # Session TTL Configuration - 12 hours
    session_ttl: int = Field(default=43200, description="Session TTL in seconds")

    # Streaming Configuration
    max_context_messages: int = Field(
        default=50, ge=1, le=200, description="Maximum context messages"
    )

    stream_timeout: int = Field(default=300, description="Stream timeout in seconds")
    max_retries: int = Field(default=3, description="Maximum number of retries")
    max_agent_runs: int = Field(default=3, description="Maximum number of agent runs")
    max_agent_schedules: int = Field(
        default=3, description="Maximum number of agent schedules"
    )

    # Langfuse Prompt Management Configuration
    # Note: Langfuse credentials are in Settings().secrets (settings.py)
    langfuse_prompt_name: str = Field(
        default="CoPilot Prompt",
        description="Name of the prompt in Langfuse to fetch",
    )

    @field_validator("api_key", mode="before")
    @classmethod
    def get_api_key(cls, v):
        """Get API key from environment if not provided."""
        if v is None:
            # Try to get from environment variables
            # First check for CHAT_API_KEY (Pydantic prefix)
            v = os.getenv("CHAT_API_KEY")
            if not v:
                # Fall back to OPEN_ROUTER_API_KEY
                v = os.getenv("OPEN_ROUTER_API_KEY")
            if not v:
                # Fall back to OPENAI_API_KEY
                v = os.getenv("OPENAI_API_KEY")
        return v

    @field_validator("base_url", mode="before")
    @classmethod
    def get_base_url(cls, v):
        """Get base URL from environment if not provided."""
        if v is None:
            # Check for OpenRouter or custom base URL
            v = os.getenv("CHAT_BASE_URL")
            if not v:
                v = os.getenv("OPENROUTER_BASE_URL")
            if not v:
                v = os.getenv("OPENAI_BASE_URL")
            if not v:
                v = "https://openrouter.ai/api/v1"
        return v

    # Prompt paths for different contexts
    PROMPT_PATHS: dict[str, str] = {
        "default": "prompts/chat_system.md",
        "onboarding": "prompts/onboarding_system.md",
    }

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables
