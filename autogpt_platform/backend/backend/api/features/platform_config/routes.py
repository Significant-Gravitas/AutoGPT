from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.util.platform_payment import is_platform_payment_enabled
from backend.util.settings import Settings

settings = Settings()

router = APIRouter(prefix="/platform-config", tags=["platform-config"])


class PlatformConfig(BaseModel):
    payment_enabled: bool = Field(
        description="Whether the platform is paid for, so an account without a "
        "plan is paywalled (`ENABLE_PLATFORM_PAYMENT`)."
    )
    mobile_min_version: str | None = Field(
        default=None,
        description="Oldest mobile app version still allowed in; an older app "
        "asks to be updated (`MOBILE_MIN_APP_VERSION`). Null means no minimum.",
    )


@router.get("", operation_id="get_platform_config")
async def get_platform_config() -> PlatformConfig:
    """Settings the browser and the mobile app need, the same for every caller."""
    return PlatformConfig(
        payment_enabled=is_platform_payment_enabled(),
        mobile_min_version=settings.config.mobile_min_app_version.strip() or None,
    )
