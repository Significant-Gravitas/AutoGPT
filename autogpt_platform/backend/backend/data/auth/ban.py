from datetime import datetime


def is_ban_active(
    banned: bool | None, ban_expires: datetime | None, now: datetime
) -> bool:
    return bool(banned) and (ban_expires is None or ban_expires > now)
