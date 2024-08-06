import fastapi
import fastapi.exceptions
import prisma.models

import autogpt_libs.auth.middleware
import market.db


async def get_user(
    payload: dict = fastapi.Depends(autogpt_libs.auth.middleware.auth_middleware),
) -> prisma.models.User | None:
    if not payload:
        # This handles the case when authentication is disabled
        user_id = "3e53486c-cf57-477e-ba2a-cb02dc828e1a"

    user_id = payload.get("sub")
    if not user_id:
        raise fastapi.exceptions.HTTPException(
            status_code=401, detail="User ID not found in token"
        )
    
    user = await market.db.get_or_create_user(user_id)
    
    return user
