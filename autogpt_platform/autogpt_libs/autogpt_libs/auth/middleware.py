import inspect
import logging
from typing import Any, Callable, Optional

from fastapi import HTTPException, Request, Security
from fastapi.security import APIKeyHeader, HTTPBearer
from starlette.status import HTTP_401_UNAUTHORIZED

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


class APIKeyValidator:
    """
    Configurable API key validator that supports custom validation functions
    for FastAPI applications.

    This class provides a flexible way to implement API key authentication with optional
    custom validation logic. It can be used for simple token matching
    or more complex validation scenarios like database lookups.

    Examples:
        Simple token validation:
        ```python
        validator = APIKeyValidator(
            header_name="X-API-Key",
            expected_token="your-secret-token"
        )

        @app.get("/protected", dependencies=[Depends(validator.get_dependency())])
        def protected_endpoint():
            return {"message": "Access granted"}
        ```

        Custom validation with database lookup:
        ```python
        async def validate_with_db(api_key: str):
            api_key_obj = await db.get_api_key(api_key)
            return api_key_obj if api_key_obj and api_key_obj.is_active else None

        validator = APIKeyValidator(
            header_name="X-API-Key",
            validate_fn=validate_with_db
        )
        ```

    Args:
        header_name (str): The name of the header containing the API key
        expected_token (Optional[str]): The expected API key value for simple token matching
        validate_fn (Optional[Callable]): Custom validation function that takes an API key
            string and returns a boolean or object. Can be async.
        error_status (int): HTTP status code to use for validation errors
        error_message (str): Error message to return when validation fails
    """

    def __init__(
        self,
        header_name: str,
        expected_token: Optional[str] = None,
        validate_fn: Optional[Callable[[str], bool]] = None,
        error_status: int = HTTP_401_UNAUTHORIZED,
        error_message: str = "Invalid API key",
    ):
        # Create the APIKeyHeader as a class property
        self.security_scheme = APIKeyHeader(name=header_name)
        self.expected_token = expected_token
        self.custom_validate_fn = validate_fn
        self.error_status = error_status
        self.error_message = error_message

    async def default_validator(self, api_key: str) -> bool:
        return api_key == self.expected_token

    async def __call__(
        self, request: Request, api_key: str = Security(APIKeyHeader)
    ) -> Any:
        if api_key is None:
            raise HTTPException(status_code=self.error_status, detail="Missing API key")

        # Use custom validation if provided, otherwise use default equality check
        validator = self.custom_validate_fn or self.default_validator
        result = (
            await validator(api_key)
            if inspect.iscoroutinefunction(validator)
            else validator(api_key)
        )

        if not result:
            raise HTTPException(
                status_code=self.error_status, detail=self.error_message
            )

        # Store validation result in request state if it's not just a boolean
        if result is not True:
            request.state.api_key = result

        return result

    def get_dependency(self):
        """
        Returns a callable dependency that FastAPI will recognize as a security scheme
        """

        async def validate_api_key(
            request: Request, api_key: str = Security(self.security_scheme)
        ) -> Any:
            return await self(request, api_key)

        # This helps FastAPI recognize it as a security dependency
        validate_api_key.__name__ = f"validate_{self.security_scheme.model.name}"
        return validate_api_key
