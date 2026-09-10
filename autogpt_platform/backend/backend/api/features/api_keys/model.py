from typing import Optional

import pydantic

from backend.data.auth.api_key import APIKeyInfo, APIKeyPermission


class CreateAPIKeyRequest(pydantic.BaseModel):
    name: str
    permissions: list[APIKeyPermission]
    description: Optional[str] = None


class CreateAPIKeyResponse(pydantic.BaseModel):
    api_key: APIKeyInfo
    plain_text_key: str


class UpdatePermissionsRequest(pydantic.BaseModel):
    permissions: list[APIKeyPermission]
