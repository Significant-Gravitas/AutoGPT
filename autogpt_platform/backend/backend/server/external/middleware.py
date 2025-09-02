from fastapi import HTTPException, Security
from fastapi.security import APIKeyHeader
from prisma.enums import APIKeyPermission

from backend.data.api_key import APIKeyInfo, has_permission, validate_api_key

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


async def require_api_key(api_key: str | None = Security(api_key_header)) -> APIKeyInfo:
    """Base middleware for API key authentication"""
    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")

    api_key_obj = await validate_api_key(api_key)

    if not api_key_obj:
        raise HTTPException(status_code=401, detail="Invalid API key")

    return api_key_obj


def require_permission(permission: APIKeyPermission):
    """Dependency function for checking specific permissions"""

    async def check_permission(
        api_key: APIKeyInfo = Security(require_api_key),
    ) -> APIKeyInfo:
        if not has_permission(api_key, permission):
            raise HTTPException(
                status_code=403,
                detail=f"API key lacks the required permission '{permission}'",
            )
        return api_key

    return check_permission
