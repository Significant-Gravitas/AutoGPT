from fastapi import Depends, HTTPException, Request
from fastapi.security import APIKeyHeader
from prisma.enums import APIKeyPermission

from backend.data.api_key import has_permission, validate_api_key

api_key_header = APIKeyHeader(name="X-API-Key")


async def require_api_key(request: Request):
    """Base middleware for API key authentication"""
    api_key = await api_key_header(request)

    if api_key is None:
        raise HTTPException(status_code=401, detail="Missing API key")

    api_key_obj = await validate_api_key(api_key)

    if not api_key_obj:
        raise HTTPException(status_code=401, detail="Invalid API key")

    request.state.api_key = api_key_obj
    return api_key_obj


def require_permission(permission: APIKeyPermission):
    """Dependency function for checking specific permissions"""

    async def check_permission(api_key=Depends(require_api_key)):
        if not has_permission(api_key, permission):
            raise HTTPException(
                status_code=403,
                detail=f"API key missing required permission: {permission}",
            )
        return api_key

    return check_permission
