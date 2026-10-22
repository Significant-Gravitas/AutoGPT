"""The notification sender's verified-email gate against the real database.

``User.emailVerified`` defaults to true, so a gate that read it alone would
let every unverified email/password sign-up through. These pin that the auth
identity decides, that a Google sign-in counts, and that a bounce still wins.
"""

from uuid import uuid4

import pytest
from prisma.models import AuthAccount, AuthUser, Organization, User

from backend.data.user import (
    forget_user_email_verification,
    get_or_create_user_with_status,
    get_user_email_verification,
    set_user_email_verification,
)
from backend.util.test import SpinTestServer


async def _account(
    *,
    identity_verified: bool | None,
    provider: str = "credential",
    bounced: bool = False,
) -> str:
    """A platform user, plus an auth identity and sign-in account unless
    *identity_verified* is None."""
    user_id = str(uuid4())
    email = f"{user_id}@example.com"
    if identity_verified is not None:
        await AuthUser.prisma().create(
            data={
                "id": user_id,
                "name": "Gate test",
                "email": email,
                "emailVerified": identity_verified,
            }
        )
        await AuthAccount.prisma().create(
            data={
                "id": str(uuid4()),
                "accountId": user_id,
                "providerId": provider,
                "userId": user_id,
            }
        )
    await User.prisma().create(
        data={"id": user_id, "email": email, "emailVerified": not bounced}
    )
    return user_id


async def _cleanup(ids: list[str]) -> None:
    for user_id in ids:
        forget_user_email_verification(user_id)
    await AuthUser.prisma().delete_many(where={"id": {"in": ids}})
    await User.prisma().delete_many(where={"id": {"in": ids}})


@pytest.mark.asyncio(loop_scope="session")
async def test_only_a_proven_address_that_has_not_bounced_gets_mail(
    server: SpinTestServer,
):
    ids: dict[str, str] = {}
    try:
        ids["unverified_password"] = await _account(identity_verified=False)
        ids["verified_password"] = await _account(identity_verified=True)
        ids["google"] = await _account(identity_verified=True, provider="google")
        # Google proves the address even if the identity flag was never set,
        # e.g. a password sign-up that later signed in with Google.
        ids["google_flag_unset"] = await _account(
            identity_verified=False, provider="google"
        )
        ids["bounced"] = await _account(identity_verified=True, bounced=True)
        ids["bounced_google"] = await _account(
            identity_verified=True, provider="google", bounced=True
        )
        ids["no_identity"] = await _account(identity_verified=None)

        got = {name: await get_user_email_verification(i) for name, i in ids.items()}

        assert got == {
            "unverified_password": False,
            "verified_password": True,
            "google": True,
            "google_flag_unset": True,
            "bounced": False,
            "bounced_google": False,
            "no_identity": False,
        }
    finally:
        await _cleanup(list(ids.values()))


@pytest.mark.asyncio(loop_scope="session")
async def test_a_bounce_is_never_hidden_by_the_cache(server: SpinTestServer):
    """A lookup in flight when the bounce lands can't cache its way past it:
    the bounce flag is read on every call."""
    user_id = await _account(identity_verified=True)
    try:
        # The notification manager asks over RPC, which passes a keyword.
        assert await get_user_email_verification(user_id=user_id) is True
        assert await get_user_email_verification(user_id) is True

        # Written behind the cache's back, as if after a lookup had started.
        await User.prisma().update(where={"id": user_id}, data={"emailVerified": False})

        assert await get_user_email_verification(user_id=user_id) is False
        assert await get_user_email_verification(user_id) is False

        await set_user_email_verification(user_id, True)
        assert await get_user_email_verification(user_id=user_id) is True
    finally:
        await _cleanup([user_id])


@pytest.mark.asyncio(loop_scope="session")
async def test_signing_in_after_verifying_drops_the_cached_refusal(
    server: SpinTestServer,
):
    """Better Auth marks the identity verified from the frontend, which can't
    touch this cache; the ``POST /auth/user`` every sign-in makes drops it."""
    user_id = await _account(identity_verified=False)
    try:
        assert await get_user_email_verification(user_id=user_id) is False

        await AuthUser.prisma().update(
            where={"id": user_id}, data={"emailVerified": True}
        )
        # Still cached: nothing in the backend saw the link being opened.
        assert await get_user_email_verification(user_id=user_id) is False

        await get_or_create_user_with_status(
            {"sub": user_id, "email": f"{user_id}@example.com"}
        )

        assert await get_user_email_verification(user_id=user_id) is True
    finally:
        await Organization.prisma().delete_many(where={"bootstrapUserId": user_id})
        await _cleanup([user_id])
