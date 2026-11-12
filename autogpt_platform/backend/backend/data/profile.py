"""The marketplace Profile every user has, created with a default handle."""

import logging
import random
import uuid
from typing import Optional

from prisma.errors import UniqueViolationError
from prisma.types import ProfileCreateInput

from backend.data.db import prisma

logger = logging.getLogger(__name__)

# Word lists mirror the legacy generate_username() SQL function so that app-
# and DB-generated handles look consistent.
_PROFILE_USERNAME_ADJECTIVES = (
    "happy",
    "clever",
    "swift",
    "bright",
    "wise",
    "funny",
    "cool",
    "awesome",
    "amazing",
    "fantastic",
    "wonderful",
)
_PROFILE_USERNAME_ANIMALS = (
    "fox",
    "wolf",
    "bear",
    "eagle",
    "owl",
    "tiger",
    "lion",
    "elephant",
    "giraffe",
    "zebra",
)


async def ensure_user_profile(user_id: str) -> None:
    """Give *user_id* their default marketplace Profile if they have none yet.

    Account setup (``get_or_create_user``, via ``POST /auth/user``) creates it,
    but Better Auth issues the session as soon as the identity exists, so a
    first page can ask for the Profile before that call has run, or after its
    best-effort attempt failed. The default name comes from the user's own
    row, never from a token's claims, so it is also right for an admin acting
    as the user. No-op for an id with no platform ``User`` row.
    """
    user = await prisma.user.find_unique(where={"id": user_id})
    if user is None:
        return
    await ensure_profile(user.id, user.email)


async def ensure_profile(user_id: str, email: Optional[str]) -> None:
    """Create a default marketplace Profile for *user_id* if none exists.

    Idempotent and race-safe. A UniqueViolationError has two possible sources:
    a concurrent request that already created this user's Profile (done), or a
    collision on the unique username with *another* user (retry with a fresh
    handle — otherwise the user would be left without a Profile).
    """
    if await prisma.profile.find_unique(where={"userId": user_id}):
        return

    name = (email or "").split("@", 1)[0] or "user"
    for _ in range(3):
        try:
            await prisma.profile.create(
                data=ProfileCreateInput(
                    userId=user_id,
                    name=name,
                    username=await _generate_profile_username(),
                    description="I'm new here",
                    links=[],
                    avatarUrl="",
                )
            )
            return
        except UniqueViolationError:
            if await prisma.profile.find_unique(where={"userId": user_id}):
                # Another in-flight request (or the legacy auth.users trigger)
                # created this user's Profile — nothing to do.
                logger.debug(
                    "Profile for user %s already created concurrently", user_id
                )
                return
            # The generated username collided with another user — loop and
            # retry with a fresh handle.
    logger.warning(
        "Failed to create a unique profile handle for user %s after retries",
        user_id,
    )


async def _generate_profile_username() -> str:
    """Generate a human-friendly profile handle, avoiding obvious collisions.

    The unique constraint on Profile.username is the real guarantee; this
    pre-check just avoids retrying a create in the common case. Falls back to a
    UUID-based handle if we can't find a free friendly name.
    """
    for _ in range(10):
        candidate = (
            f"{random.choice(_PROFILE_USERNAME_ADJECTIVES)}-"
            f"{random.choice(_PROFILE_USERNAME_ANIMALS)}-"
            f"{random.randint(10000, 99999)}"
        )
        if not await prisma.profile.find_unique(where={"username": candidate}):
            return candidate
    return f"user-{uuid.uuid4().hex[:12]}"
