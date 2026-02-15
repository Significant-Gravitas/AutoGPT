from typing import Optional

from pydantic import BaseModel, SecretStr

from forge.models.config import UserConfigurable


class PlatformBlocksConfig(BaseModel):
    """Configuration for platform blocks integration.

    Set PLATFORM_API_KEY environment variable to enable platform blocks.
    """

    enabled: bool = UserConfigurable(default=True, from_env="PLATFORM_BLOCKS_ENABLED")
    platform_url: str = UserConfigurable(
        default="https://platform.agpt.co",
        from_env="PLATFORM_URL",
    )
    api_key: Optional[SecretStr] = UserConfigurable(
        default=None,
        from_env="PLATFORM_API_KEY",
        exclude=True,
    )
    timeout: int = UserConfigurable(default=60, from_env="PLATFORM_TIMEOUT")
