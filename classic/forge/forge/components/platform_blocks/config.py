from pydantic import BaseModel, Field


class PlatformBlocksConfig(BaseModel):
    """Configuration for platform blocks integration."""

    enabled: bool = Field(
        default=True, description="Whether platform blocks are enabled"
    )
    platform_url: str = Field(
        default="https://platform.agpt.co",
        description="Platform API URL for execution",
    )
    api_key: str = Field(default="", description="Platform API key for authentication")
    user_id: str = Field(default="", description="User ID for credential lookup")
    timeout: int = Field(default=60, description="Execution timeout in seconds")
