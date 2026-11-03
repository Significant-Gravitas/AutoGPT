from datetime import datetime, timezone

from prisma.models import AuthSession


async def is_auth_session_live(session_id: str, user_id: str) -> bool:
    """Whether the session exists, belongs to the user and hasn't expired."""
    session = await AuthSession.prisma().find_unique(where={"id": session_id})
    return (
        session is not None
        and session.userId == user_id
        and session.expiresAt > datetime.now(timezone.utc)
    )
