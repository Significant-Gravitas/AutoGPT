from datetime import datetime
from typing import Literal, Optional

from prisma.enums import APIKeyPermission
from pydantic import BaseModel


class APIAuthorizationInfo(BaseModel):
    user_id: str
    scopes: list[APIKeyPermission]
    type: Literal["oauth", "api_key"]
    created_at: datetime
    expires_at: Optional[datetime] = None
    last_used_at: Optional[datetime] = None
    revoked_at: Optional[datetime] = None
    organization_id: Optional[str] = None
    # When set, actions authorized by this principal are pinned to this
    # team within ``organization_id`` (API keys minted with a team scope).
    team_id_restriction: Optional[str] = None
