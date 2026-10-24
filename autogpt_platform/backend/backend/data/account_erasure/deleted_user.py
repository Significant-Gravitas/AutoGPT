"""The "deleted user" account: the owner of agents and marketplace listings
whose creator deleted their account while other accounts still use them.

It is a platform `User` row with a marketplace `Profile` and nothing else: no
auth identity, so nobody can sign in as it, and every email and marketing
setting off. It is created on first use, with a fixed ID, the way
`create_default_user` provisions the self-hosted default user.
"""

from datetime import UTC, datetime

from prisma import Prisma
from prisma.enums import BriefingFrequency
from prisma.models import Profile, User

DELETED_USER_ID = "d83236d9-19c5-455f-90c8-4cc6d76d4dac"
# `.invalid` is reserved (RFC 2606): no mailbox can exist, so nobody can
# sign up with it and nothing can be delivered to it.
DELETED_USER_EMAIL = "deleted-user@agpt.invalid"
DELETED_USER_NAME = "Deleted user"
DELETED_USER_USERNAME = "deleted-user"
_DESCRIPTION = "The creator of these workflows deleted their account."
_OPT_OUT_SOURCE = "system_account"


async def ensure_deleted_user(client: Prisma) -> str:
    """Create the account and its Profile if missing; return the Profile ID.
    Never changes an existing one, so it is safe on every run."""
    await User.prisma(client).upsert(
        where={"id": DELETED_USER_ID},
        data={
            "create": {
                "id": DELETED_USER_ID,
                "email": DELETED_USER_EMAIL,
                "emailVerified": False,
                "name": DELETED_USER_NAME,
                "maxEmailsPerDay": 0,
                "briefingFrequency": BriefingFrequency.OFF,
                "alertsEnabled": False,
                "notifyOnStoreVerdict": False,
                "marketingOptOutAt": datetime.now(UTC),
                "marketingOptOutSource": _OPT_OUT_SOURCE,
            },
            "update": {},
        },
    )
    profile = await Profile.prisma(client).find_unique(
        where={"userId": DELETED_USER_ID}
    )
    if profile:
        return profile.id
    profile = await Profile.prisma(client).create(
        data={
            "userId": DELETED_USER_ID,
            "name": DELETED_USER_NAME,
            "username": await _free_username(client),
            "description": _DESCRIPTION,
            "links": [],
        }
    )
    return profile.id


async def _free_username(client: Prisma) -> str:
    """`deleted-user`, unless an account already took that handle. Checked
    first rather than caught: a unique violation would abort the transaction
    this runs in."""
    taken = await Profile.prisma(client).find_unique(
        where={"username": DELETED_USER_USERNAME}
    )
    if taken is None:
        return DELETED_USER_USERNAME
    return f"{DELETED_USER_USERNAME}-{DELETED_USER_ID[:8]}"
