from datetime import datetime
from typing import Literal

from prisma.enums import APIKeyPermission
from pydantic import BaseModel


class APIAuthorizationInfo(BaseModel):
    user_id: str
    scopes: list[APIKeyPermission]
    type: Literal["oauth", "api_key"]
    created_at: datetime
    expires_at: datetime | None = None
    last_used_at: datetime | None = None
    revoked_at: datetime | None = None
    organization_id: str | None = None
    # When set, actions authorized by this principal are pinned to this
    # team within ``organization_id`` (API keys minted with a team scope).
    team_id_restriction: str | None = None
