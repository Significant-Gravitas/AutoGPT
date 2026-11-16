from fastapi import APIRouter
from pydantic import BaseModel, Field

from backend.util.platform_payment import is_platform_payment_enabled

router = APIRouter(prefix="/platform-config", tags=["platform-config"])


class PlatformConfig(BaseModel):
    payment_enabled: bool = Field(
        description="Whether the platform is paid for, so an account without a "
        "plan is paywalled (`ENABLE_PLATFORM_PAYMENT`)."
    )


@router.get("", operation_id="get_platform_config")
async def get_platform_config() -> PlatformConfig:
    """This deployment's settings the browser needs, the same for every caller."""
    return PlatformConfig(payment_enabled=is_platform_payment_enabled())
