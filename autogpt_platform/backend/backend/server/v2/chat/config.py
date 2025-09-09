"""Configuration management for chat system."""

import os
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings


class ChatConfig(BaseSettings):
    """Configuration for the chat system."""

    # OpenAI API Configuration
    model: str = Field(
        default="qwen/qwen3-235b-a22b-2507", description="Default model to use"
    )
    api_key: str | None = Field(default=None, description="OpenAI API key")
    base_url: str | None = Field(
        default="https://openrouter.ai/api/v1",
        description="Base URL for API (e.g., for OpenRouter)",
    )

    # System Prompt Configuration
    system_prompt_path: str = Field(
        default="prompts/chat_system.md",
        description="Path to system prompt file relative to chat module",
    )

    # Streaming Configuration
    max_context_messages: int = Field(
        default=50, ge=1, le=200, description="Maximum context messages"
    )
    stream_timeout: int = Field(default=300, description="Stream timeout in seconds")

    # Client Configuration
    cache_client: bool = Field(
        default=True, description="Whether to cache the OpenAI client"
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
            v = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
            if os.getenv("USE_OPENROUTER") == "true" and not v:
                v = "https://openrouter.ai/api/v1"
        return v

    def get_system_prompt(self, **template_vars) -> str:
        """Load and render the system prompt from file.

        Args:
            **template_vars: Variables to substitute in the template

        Returns:
            Rendered system prompt string

        """
        # Get the path relative to this module
        module_dir = Path(__file__).parent
        prompt_path = module_dir / self.system_prompt_path

        # Check for .j2 extension first (Jinja2 template)
        j2_path = Path(str(prompt_path) + ".j2")
        if j2_path.exists():
            try:
                from jinja2 import Template

                template = Template(j2_path.read_text())
                return template.render(**template_vars)
            except ImportError:
                # Jinja2 not installed, fall back to reading as plain text
                return j2_path.read_text()

        # Check for markdown file
        if prompt_path.exists():
            content = prompt_path.read_text()

            # Simple variable substitution if Jinja2 is not available
            for key, value in template_vars.items():
                placeholder = f"{{{key}}}"
                content = content.replace(placeholder, str(value))

            return content

        # Fallback to default system prompt if file not found
        return self._get_default_system_prompt()

    def _get_default_system_prompt(self) -> str:
        """Get the default system prompt if file is not found."""
        return """# AutoGPT Agent Setup Assistant

You help users find and set up AutoGPT agents to solve their business problems. **Bias toward action** - move quickly to get agents running.

## THE FLOW (Always Follow This Order)

1. **find_agent** → Search for agents that solve their problem
2. **get_agent_details** → Get comprehensive info about chosen agent  
3. **get_required_setup_info** → Verify user has required credentials (MANDATORY before next step)
4. **setup_agent** or **run_agent** → Execute the agent

## YOUR APPROACH

### STEP 1: UNDERSTAND THE PROBLEM (Quick)
- One or two targeted questions max
- What business problem are they trying to solve?
- Move quickly to searching for solutions

### STEP 2: FIND AGENTS
- Use `find_agent` immediately with relevant keywords
- Suggest the best option based on what you know
- Explain briefly how it solves their problem
- Ask them if they would like to use it, if they do move to step 3

### STEP 3: GET DETAILS
- Use `get_agent_details` on their chosen agent
- Explain what the agent does and its requirements
- Keep explanations brief and outcome-focused

### STEP 4: VERIFY SETUP (CRITICAL)
- **ALWAYS** use `get_required_setup_info` before proceeding
- This checks if user has all required credentials
- Tell user what credentials they need (if any)
- Explain credentials are added via the frontend interface

### STEP 5: EXECUTE
- Once credentials verified, use `setup_agent` for scheduled runs OR `run_agent` for immediate execution
- Confirm successful setup/run
- Provide clear next steps

## KEY RULES

### What You DON'T Do:
- Don't help with login (frontend handles this)
- Don't help add credentials (frontend handles this)
- Don't skip `get_required_setup_info` (it's mandatory)
- Don't over-explain technical details
- Don't use ** to highlight text

### What You DO:
- Act fast - get to agent discovery quickly
- Use tools proactively without asking permission
- Keep explanations short and business-focused
- Always verify credentials before setup/run
- Focus on outcomes and value

### Error Handling:
- If authentication needed → Tell user to sign in via the interface
- If credentials missing → Tell user what's needed and where to add them in the frontend
- If setup fails → Identify issue, provide clear fix

## SUCCESS LOOKS LIKE:
- User has an agent running within minutes
- User understands what their agent does
- User knows how to use their agent going forward
- Minimal back-and-forth, maximum action

**Remember: Speed to value. Find agent → Get details → Verify credentials → Run. Keep it simple, keep it moving.**"""

    class Config:
        """Pydantic config."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # Ignore extra environment variables


# Global configuration instance
_config: ChatConfig | None = None


def get_config() -> ChatConfig:
    """Get or create the chat configuration."""
    global _config
    if _config is None:
        _config = ChatConfig()
    return _config


def set_config(config: ChatConfig) -> None:
    """Set the chat configuration."""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the configuration to defaults."""
    global _config
    _config = None
