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
    max_agent_runs: int = Field(default=30, description="Maximum number of agent runs")
    max_agent_schedules: int = Field(
        default=30, description="Maximum number of agent schedules"
    )

    # Long-running operation configuration
    long_running_operation_ttl: int = Field(
        default=600,
        description="TTL in seconds for long-running operation tracking in Redis (safety net if pod dies)",
    )

    # Stream registry configuration for SSE reconnection
    stream_ttl: int = Field(
        default=3600,
        description="TTL in seconds for stream data in Redis (1 hour)",
    )
    stream_max_length: int = Field(
        default=10000,
        description="Maximum number of messages to store per stream",
    )

    # Redis Streams configuration for completion consumer
    stream_completion_name: str = Field(
        default="chat:completions",
        description="Redis Stream name for operation completions",
    )
    stream_consumer_group: str = Field(
        default="chat_consumers",
        description="Consumer group name for completion stream",
    )
    stream_claim_min_idle_ms: int = Field(
        default=60000,
        description="Minimum idle time in milliseconds before claiming pending messages from dead consumers",
    )

    # Redis key prefixes for stream registry
    task_meta_prefix: str = Field(
        default="chat:task:meta:",
        description="Prefix for task metadata hash keys",
    )
    task_stream_prefix: str = Field(
        default="chat:stream:",
        description="Prefix for task message stream keys",
    )
    task_op_prefix: str = Field(
        default="chat:task:op:",
        description="Prefix for operation ID to task ID mapping keys",
    )
    internal_api_key: str | None = Field(
        default=None,
        description="API key for internal webhook callbacks (env: CHAT_INTERNAL_API_KEY)",
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

    @field_validator("internal_api_key", mode="before")
    @classmethod
    def get_internal_api_key(cls, v):
        """Get internal API key from environment if not provided."""
        if v is None:
            v = os.getenv("CHAT_INTERNAL_API_KEY")
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
