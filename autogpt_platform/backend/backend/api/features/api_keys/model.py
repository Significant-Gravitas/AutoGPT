from typing import Annotated, Optional

import pydantic

from backend.data.auth.api_key import (
    OAUTH_ONLY_PERMISSIONS,
    APIKeyInfo,
    APIKeyPermission,
)
from backend.util.text_limits import (
    MAX_DESCRIPTION_LENGTH,
    MAX_ID_LENGTH,
    MAX_NAME_LENGTH,
)


def _no_oauth_only_permissions(
    permissions: list[APIKeyPermission],
) -> list[APIKeyPermission]:
    oauth_only = sorted(set(permissions) & OAUTH_ONLY_PERMISSIONS)
    if oauth_only:
        raise ValueError(
            f"{', '.join(oauth_only)} can only be granted through OAuth, not to an "
            "API key"
        )
    return permissions


# Repeats pass the enum check, so the count is what bounds the list.
APIKeyPermissions = Annotated[
    list[APIKeyPermission],
    pydantic.Field(max_length=len(APIKeyPermission)),
    pydantic.AfterValidator(_no_oauth_only_permissions),
]


class CreateAPIKeyRequest(pydantic.BaseModel):
    name: str = pydantic.Field(max_length=MAX_NAME_LENGTH)
    permissions: APIKeyPermissions
    description: Optional[str] = pydantic.Field(
        default=None, max_length=MAX_DESCRIPTION_LENGTH
    )
    team_id: Optional[str] = pydantic.Field(default=None, max_length=MAX_ID_LENGTH)


class CreateAPIKeyResponse(pydantic.BaseModel):
    api_key: APIKeyInfo
    plain_text_key: str


class UpdatePermissionsRequest(pydantic.BaseModel):
    permissions: APIKeyPermissions
