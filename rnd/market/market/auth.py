import functools
import logging
import typing

import autogpt_libs.auth.middleware
import fastapi
import prisma.models

logger = logging.getLogger("uvicorn.error")


async def get_user(
    payload: dict = fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware),
) -> dict[str, prisma.models.User | str]:
    if not payload:
        # This handles the case when authentication is disabled
        user_id = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"

    user_id = payload.get("sub")
    logger.info(payload["role"])
    if not user_id:
        raise fastapi.HTTPException(
            status_code=401, detail="User ID not found in token"
        )

    return {"user_id": user_id, "role": payload["role"]}


def require_auth(admin_only: bool = False) -> typing.Callable:
    def decorator(func: typing.Callable) -> typing.Callable:
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            user, role = await get_user()

            if not user:
                raise fastapi.HTTPException(
                    status_code=401, detail="User not authenticated"
                )

            if admin_only and role != "admin":
                raise fastapi.HTTPException(
                    status_code=403, detail="Admin access required"
                )

            kwargs["user"] = user
            return await func(*args, **kwargs)

        return wrapper

    return decorator
