import uuid
from datetime import UTC, datetime, timedelta

from autogpt_libs.auth import auth_middleware
from fastapi import Depends, HTTPException
from prisma.models import StateToken

from autogpt_server.data.user import DEFAULT_USER_ID


def get_user_id(payload: dict = Depends(auth_middleware)) -> str:
    if not payload:
        # This handles the case when authentication is disabled
        return DEFAULT_USER_ID

    user_id = payload.get("sub")
    if not user_id:
        raise HTTPException(status_code=401, detail="User ID not found in token")
    return user_id


def generate_state_token():
    return str(uuid.uuid4())


async def store_state_token(token: str, user_id: str):
    expires_at = datetime.now(UTC) + timedelta(minutes=10)
    await StateToken.create(
        data={"token": token, "userId": user_id, "expiresAt": expires_at}
    )


async def verify_state_token(token: str):
    stored_token = await StateToken.find_unique(where={"token": token})
    if not stored_token:
        raise HTTPException(status_code=400, detail="Invalid state token")
    if stored_token.expiresAt < datetime.now(UTC):
        await StateToken.delete(where={"id": stored_token.id})
        raise HTTPException(status_code=400, detail="Expired state token")

    return True
