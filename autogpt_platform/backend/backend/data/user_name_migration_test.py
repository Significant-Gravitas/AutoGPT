"""The migration that clears the names email sign-up auto-filled from the
address keeps every name a user or their Google profile gave, and moves one
given only to AutoPilot's understanding onto the account."""

import re
import uuid
from datetime import UTC, datetime, timedelta
from pathlib import Path

import prisma
import pytest

from backend.util.json import SafeJson
from backend.util.test import SpinTestServer

_MIGRATION = (
    Path(__file__).resolve().parents[2]
    / "migrations/20261026090000_clear_auto_filled_user_names/migration.sql"
)
_COMMENTS = re.compile(r"--[^\n]*")
# The migration's UPDATEs in order, each with the alias it gives its table.
_TARGETS = [('"UserAuthIdentity"', "i"), ('"User"', "u"), ('"User"', "u")]
_SIGN_UP = datetime(2026, 1, 5, 9, 0, tzinfo=UTC)
_DAY = 24 * 60 * 60


@pytest.mark.asyncio(loop_scope="session")
async def test_the_migration_clears_only_the_auto_filled_names(
    server: SpinTestServer,
):
    suffix = uuid.uuid4().hex[:8]
    # (address, stored name, sign-in methods with their seconds after sign-up, the
    # understanding's name): sign-up lowercased the address but auto-filled the
    # local part as typed, and onboarding copied that name into the understanding.
    accounts = {
        "auto_filled": (
            f"jane.doe{suffix}@example.com",
            f"Jane.Doe{suffix}",
            [("credential", 0)],
            f"jane.doe{suffix}",
        ),
        "copied_without_account": (
            f"lee{suffix}@example.com",
            f"lee{suffix}",
            [],
            None,
        ),
        "given": (f"jo{suffix}@example.com", "Jo", [("credential", 0)], "Bob"),
        "google_only": (
            f"john{suffix}@example.com",
            f"John{suffix}",
            [("google", 0)],
            None,
        ),
        # Google first, password added later: the name may be Google's.
        "google_then_password": (
            f"max{suffix}@example.com",
            f"Max{suffix}",
            [("google", 0), ("credential", _DAY)],
            None,
        ),
        # The Supabase copy dates a password to the user's creation, just before Google's.
        "google_copied_from_supabase": (
            f"ann{suffix}@example.com",
            f"ann{suffix}",
            [("credential", 0), ("google", 1)],
            None,
        ),
        # Email first, Google linked later: linking leaves the auto-filled name.
        "password_then_google": (
            f"kim{suffix}@example.com",
            f"kim{suffix}",
            [("credential", 0), ("google", _DAY)],
            None,
        ),
        # A name told to Otto beside an auto-filled one.
        "given_to_otto": (
            f"sam.lee{suffix}@example.com",
            f"sam.lee{suffix}",
            [("credential", 0)],
            " Sam ",
        ),
    }
    ids = {key: str(uuid.uuid4()) for key in accounts}
    try:
        for key, (email, name, providers, understood) in accounts.items():
            await _seed(ids[key], email, name, providers, understood)

        await _run_migration(list(ids.values()))

        identities = {
            row.id: row.name
            for row in await prisma.models.AuthUser.prisma().find_many(
                where={"id": {"in": list(ids.values())}}
            )
        }
        users = {
            row.id: row.name
            for row in await prisma.models.User.prisma().find_many(
                where={"id": {"in": list(ids.values())}}
            )
        }
        for key in ("auto_filled", "copied_without_account", "password_then_google"):
            assert identities[ids[key]] == ""
            assert users[ids[key]] is None
        for key in (
            "given",
            "google_only",
            "google_then_password",
            "google_copied_from_supabase",
        ):
            assert identities[ids[key]] == accounts[key][1]
            assert users[ids[key]] == accounts[key][1]
        assert identities[ids["given_to_otto"]] == ""
        assert users[ids["given_to_otto"]] == "Sam"
    finally:
        await prisma.models.AuthUser.prisma().delete_many(
            where={"id": {"in": list(ids.values())}}
        )
        await prisma.models.User.prisma().delete_many(
            where={"id": {"in": list(ids.values())}}
        )


async def _seed(
    user_id: str,
    email: str,
    name: str,
    providers: list[tuple[str, int]],
    understood: str | None,
) -> None:
    await prisma.models.AuthUser.prisma().create(
        data={"id": user_id, "email": email, "name": name}
    )
    for provider, seconds in providers:
        await prisma.models.AuthAccount.prisma().create(
            data={
                "id": str(uuid.uuid4()),
                "accountId": user_id if provider == "credential" else uuid.uuid4().hex,
                "providerId": provider,
                "userId": user_id,
                "createdAt": _SIGN_UP + timedelta(seconds=seconds),
            }
        )
    await prisma.models.User.prisma().create(
        data={"id": user_id, "email": email, "name": name}
    )
    if understood is not None:
        await prisma.models.CoPilotUnderstanding.prisma().create(
            data={"userId": user_id, "data": SafeJson({"name": understood})}
        )


async def _run_migration(user_ids: list[str]) -> None:
    """The migration's own statements, confined to this test's accounts."""
    sql = _COMMENTS.sub("", _MIGRATION.read_text())
    statements = [part.strip() for part in sql.split(";") if part.strip()]
    assert len(statements) == len(_TARGETS)
    client = prisma.get_client()
    for statement, (table, alias) in zip(statements, _TARGETS):
        assert statement.startswith(f"UPDATE {table}")
        for user_id in user_ids:
            await client.execute_raw(f'{statement} AND {alias}."id" = $1', user_id)
