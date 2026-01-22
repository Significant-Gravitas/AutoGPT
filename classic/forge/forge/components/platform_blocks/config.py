from pydantic import BaseModel, Field


class PlatformBlocksConfig(BaseModel):
    """Configuration for platform blocks integration."""

    enabled: bool = Field(
        default=True, description="Whether platform blocks are enabled"
    )
    platform_url: str = Field(
        default="https://platform.agpt.co",
        description="Platform API base URL",
    )
    api_key: str = Field(default="", description="Platform API key for authentication")
    timeout: int = Field(default=60, description="Request timeout in seconds")
