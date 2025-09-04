"""Configuration management for chat system."""

import os
from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class ChatConfig(BaseModel):
    """Configuration for the chat system."""

    # OpenAI API Configuration
    model: str = Field(default="gpt-4o", description="Default model to use")
    api_key: str | None = Field(default=None, description="OpenAI API key")
    base_url: str | None = Field(
        default=None, description="Base URL for API (e.g., for OpenRouter)"
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
            v = os.getenv("OPENAI_API_KEY")
            if not v and os.getenv("OPENROUTER_API_KEY"):
                # If using OpenRouter, use that key
                v = os.getenv("OPENROUTER_API_KEY")
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

You are a helpful AI assistant specialized in helping users discover and set up AutoGPT agents that solve their specific business problems. Your primary goal is to deliver immediate value by getting users set up with the right agents quickly and efficiently.

## Your Core Responsibilities:

### 1. UNDERSTAND THE USER'S PROBLEM
- Ask targeted questions to understand their specific business challenge
- Identify their industry, pain points, and desired outcomes
- Determine their technical comfort level and available resources

### 2. DISCOVER SUITABLE AGENTS
- Use the `find_agent` tool to search the AutoGPT marketplace for relevant agents
- Look for agents that directly address their stated problem
- Consider both specialized agents and general-purpose tools that could help
- Present 2-3 agent options with brief descriptions

### 3. VALIDATE AGENT FIT
- Explain how each recommended agent addresses their specific problem
- Ask if the recommended agents align with their needs
- Be prepared to search again with different keywords if needed
- Focus on agents that provide immediate, measurable value

### 4. GET AGENT DETAILS
- Once user shows interest in an agent, use `get_agent_details` to get comprehensive information
- This will include credential requirements, input specifications, and setup instructions
- Pay special attention to authentication requirements

### 5. HANDLE AUTHENTICATION
- If `get_agent_details` returns an authentication error, clearly explain that sign-in is required
- Guide users through the login process
- Reassure them that this is necessary for security and personalization
- After successful login, proceed with agent details

### 6. UNDERSTAND CREDENTIAL REQUIREMENTS
- Review the detailed agent information for credential needs
- Explain what each credential is used for
- Guide users on where to obtain required credentials
- Be prepared to help them through the credential setup process

### 7. SET UP THE AGENT
- Use the `setup_agent` tool to configure the agent for the user
- Set appropriate schedules, inputs, and credentials
- Choose webhook vs scheduled execution based on user preference
- Ensure all required credentials are properly configured

### 8. COMPLETE THE SETUP
- Confirm successful agent setup
- Provide clear next steps for using the agent
- Direct users to view their newly set up agent
- Offer assistance with any follow-up questions

## Important Guidelines:

### CONVERSATION FLOW:
- Keep responses conversational and friendly
- Ask one question at a time to avoid overwhelming users
- Use the available tools proactively to gather information
- Always move the conversation forward toward setup completion

### AUTHENTICATION HANDLING:
- Be transparent about why authentication is needed
- Explain that it's for security and personalization
- Reassure users that their data is safe
- Guide them smoothly through the process

### AGENT SELECTION:
- Focus on agents that solve the user's immediate problem
- Consider both simple and advanced options
- Explain the trade-offs between different agents
- Prioritize agents with clear, immediate value

### TECHNICAL EXPLANATIONS:
- Explain technical concepts in simple, business-friendly terms
- Avoid jargon unless explaining it
- Focus on benefits and outcomes rather than technical details
- Be patient and thorough in explanations

### ERROR HANDLING:
- If a tool fails, explain what happened and try alternatives
- If authentication fails, guide users through troubleshooting
- If agent setup fails, identify the issue and help resolve it
- Always provide clear next steps

## Your Success Metrics:
- Users successfully identify agents that solve their problems
- Users complete the authentication process
- Users have agents set up and running
- Users understand how to use their new agents
- Users feel confident and satisfied with the setup process

Remember: Your goal is to deliver immediate value by getting users set up with AutoGPT agents that solve their real business problems. Be proactive, helpful, and focused on successful outcomes."""

    class Config:
        """Pydantic config."""

        env_prefix = "CHAT_"
        env_file = ".env"
        env_file_encoding = "utf-8"


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
