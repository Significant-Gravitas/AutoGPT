import logging

from fastapi import HTTPException, Request
from fastapi.security import HTTPBearer

from .config import settings
from .jwt_utils import parse_jwt_token

security = HTTPBearer()
logger = logging.getLogger(__name__)


async def auth_middleware(request: Request):
    if not settings.ENABLE_AUTH:
        # If authentication is disabled, allow the request to proceed
        logger.warn("Auth disabled")
        return {}

    security = HTTPBearer()
    credentials = await security(request)

    if not credentials:
        raise HTTPException(status_code=401, detail="Authorization header is missing")

    try:
        payload = parse_jwt_token(credentials.credentials)
        request.state.user = payload
        logger.debug("Token decoded successfully")
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e))
    return payload
