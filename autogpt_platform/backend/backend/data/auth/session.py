from datetime import UTC, datetime

from prisma.models import AuthSession


async def is_auth_session_active(session_id: str, user_id: str) -> bool:
    now = datetime.now(UTC)
    session = await AuthSession.prisma().find_first(
        where={"id": session_id, "userId": user_id, "expiresAt": {"gt": now}},
        include={"user": True},
    )
    if session is None or session.user is None:
        return False
    user = session.user
    return not user.banned or (user.banExpires is not None and user.banExpires <= now)
