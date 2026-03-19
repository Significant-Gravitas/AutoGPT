from backend.data.invited_user import get_or_activate_user
from backend.data.user import get_user_by_id

_SKIP_EXACT_PATHS = frozenset(
    {
        "/api/auth/check-invite",
        "/api/auth/user",
    }
)
_SKIP_PREFIXES = ("/api/public/",)


def should_enforce_platform_user(path: str) -> bool:
    if not path.startswith("/api"):
        return False

    if path in _SKIP_EXACT_PATHS:
        return False

    return not any(path.startswith(prefix) for prefix in _SKIP_PREFIXES)


async def ensure_platform_user(jwt_payload: dict) -> None:
    user_id = jwt_payload.get("sub")
    if not user_id:
        return

    try:
        await get_user_by_id(user_id)
    except ValueError:
        await get_or_activate_user(jwt_payload)
