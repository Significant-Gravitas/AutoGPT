"""
API Key authentication utilities for FastAPI applications.
"""

import inspect
import logging
import secrets
from typing import Any, Awaitable, Callable, Optional

from fastapi import HTTPException, Request
from fastapi.security import APIKeyHeader
from starlette.status import HTTP_401_UNAUTHORIZED

from backend.util.exceptions import MissingConfigError

logger = logging.getLogger(__name__)


class APIKeyAuthenticator(APIKeyHeader):
    """
    Configurable API key authenticator for FastAPI applications,
    with support for custom validation functions.

    This class provides a flexible way to implement API key authentication with optional
    custom validation logic. It can be used for simple token matching
    or more complex validation scenarios like database lookups.

    Examples:
        Simple token validation:
        ```python
        api_key_auth = APIKeyAuthenticator(
            header_name="X-API-Key",
            expected_token="your-secret-token"
        )

        @app.get("/protected", dependencies=[Security(api_key_auth)])
        def protected_endpoint():
            return {"message": "Access granted"}
        ```

        Custom validation with database lookup:
        ```python
        async def validate_with_db(api_key: str):
            api_key_obj = await db.get_api_key(api_key)
            return api_key_obj if api_key_obj and api_key_obj.is_active else None

        api_key_auth = APIKeyAuthenticator(
            header_name="X-API-Key",
            validator=validate_with_db
        )
        ```

    Args:
        header_name (str): The name of the header containing the API key
        expected_token (Optional[str]): The expected API key value for simple token matching
        validator (Optional[Callable]): Custom validation function that takes an API key
            string and returns a truthy value if and only if the passed string is a
            valid API key. Can be async.
        status_if_missing (int): HTTP status code to use for validation errors
        message_if_invalid (str): Error message to return when validation fails
    """

    def __init__(
        self,
        header_name: str,
        expected_token: Optional[str] = None,
        validator: Optional[
            Callable[[str], Any] | Callable[[str], Awaitable[Any]]
        ] = None,
        status_if_missing: int = HTTP_401_UNAUTHORIZED,
        message_if_invalid: str = "Invalid API key",
    ):
        super().__init__(
            name=header_name,
            scheme_name=f"{__class__.__name__}-{header_name}",
            auto_error=False,
        )
        self.expected_token = expected_token
        self.custom_validator = validator
        self.status_if_missing = status_if_missing
        self.message_if_invalid = message_if_invalid

    async def __call__(self, request: Request) -> Any:
        api_key = await super().__call__(request)
        if api_key is None:
            raise HTTPException(
                status_code=self.status_if_missing, detail="No API key in request"
            )

        # Use custom validation if provided, otherwise use default equality check
        validator = self.custom_validator or self.default_validator
        result = (
            await validator(api_key)
            if inspect.iscoroutinefunction(validator)
            else validator(api_key)
        )

        if not result:
            raise HTTPException(
                status_code=self.status_if_missing, detail=self.message_if_invalid
            )

        # Store validation result in request state if it's not just a boolean
        if result is not True:
            request.state.api_key = result

        return result

    async def default_validator(self, api_key: str) -> bool:
        if not self.expected_token:
            raise MissingConfigError(
                f"{self.__class__.__name__}.expected_token is not set; "
                "either specify it or provide a custom validator"
            )
        try:
            return secrets.compare_digest(api_key, self.expected_token)
        except TypeError as e:
            # If value is not an ASCII string, compare_digest raises a TypeError
            logger.warning(f"{self.model.name} API key check failed: {e}")
            return False
