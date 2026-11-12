from datetime import UTC, datetime

from prisma.models import AuthSession

from backend.data.auth.ban import is_ban_active


async def is_auth_session_active(session_id: str, user_id: str) -> bool:
    now = datetime.now(UTC)
    session = await AuthSession.prisma().find_first(
        where={"id": session_id, "userId": user_id, "expiresAt": {"gt": now}},
        include={"user": True},
    )
    if session is None or session.user is None:
        return False
    user = session.user
    return not is_ban_active(user.banned, user.banExpires, now)
