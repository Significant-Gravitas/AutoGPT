"""The rows go in one transaction, everything that would block or survive the
account's delete before it, and the auth identity with it so the
orphan-healing job never sees an identity without its account. The credit
ledger stays, scrubbed in the same transaction. A dry run only counts."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from backend.copilot.tools.skills import (
    ParsedSkill,
    _claim_skills_cache,
    _skills_cache_key,
    _write_skills_cache,
)
from backend.data import redis_client
from backend.data.account_erasure import database, skills_cache
from backend.data.account_erasure.models import StepStatus, Subject
from backend.data.redis_client import get_redis_async
from backend.util.testing import is_tcp_port_reachable

USER_ID = "3f0b8c1e-4c2a-4f7e-9a51-0c6f2d8e1a77"
SUBJECT = Subject(
    user_id=USER_ID,
    email="sam@example.com",
    has_user_row=True,
    has_auth_identity=True,
    sole_org_ids=["org-personal"],
)


def _labels(subject: Subject) -> list[str]:
    return [deletion.label for deletion in database.planned_deletions(subject)]


def test_the_account_and_its_identity_go_last_identity_first():
    labels = _labels(SUBJECT)
    assert labels[-2:] == ["auth_identity", "user"]


def test_everything_that_blocks_the_account_delete_goes_before_it():
    labels = _labels(SUBJECT)
    for blocker in (
        "webhooks",
        "library_agents",
        "presets",
        "organizations",
    ):
        assert labels.index(blocker) < labels.index("user")
    # An org's ledger and transfers would block the org's own delete.
    assert labels.index("org_credit_transactions") < labels.index("organizations")
    assert labels.index("org_transfer_requests") < labels.index("organizations")


def test_rows_without_a_foreign_key_are_deleted_explicitly():
    labels = _labels(SUBJECT)
    for orphan in (
        "kv_data",
        "refund_requests",
        "private_embeddings",
        "chat_sessions",
        "personal_credentials",
        "org_invitations",
        "team_invitations",
    ):
        assert orphan in labels


def test_the_credit_ledger_is_never_deleted():
    tables = {deletion.table for deletion in database.planned_deletions(SUBJECT)}
    assert "CreditTransaction" not in tables


def test_without_sole_orgs_no_org_is_touched():
    labels = _labels(SUBJECT.model_copy(update={"sole_org_ids": []}))
    assert "organizations" not in labels
    assert "org_credit_transactions" not in labels


def test_invitations_match_the_email_only_when_it_is_known():
    with_email = {d.label: d for d in database.planned_deletions(SUBJECT)}
    without = {
        d.label: d
        for d in database.planned_deletions(SUBJECT.model_copy(update={"email": None}))
    }
    assert with_email["org_invitations"].args == [USER_ID, "sam@example.com"]
    assert without["org_invitations"].args == [USER_ID]
    assert "$2" not in without["org_invitations"].condition


def test_auth_verifications_match_the_email_only_when_it_is_known():
    """Reset links store the user ID as the value; the sign-up and
    verification email cooldowns name the address in the identifier."""
    with_email = {d.label: d for d in database.planned_deletions(SUBJECT)}
    without = {
        d.label: d
        for d in database.planned_deletions(SUBJECT.model_copy(update={"email": None}))
    }
    assert with_email["auth_verifications"].args == [USER_ID, "sam@example.com"]
    assert '"identifier"' in with_email["auth_verifications"].condition
    assert without["auth_verifications"].args == [USER_ID]
    assert "$2" not in without["auth_verifications"].condition


def test_every_deletion_is_scoped_by_a_parameter():
    for deletion in database.planned_deletions(SUBJECT):
        assert "$1" in deletion.condition
        assert deletion.args


@pytest.fixture
def db(monkeypatch):
    tx = MagicMock(name="tx")

    @asynccontextmanager
    async def transaction(timeout):
        yield tx

    execute = AsyncMock(return_value=1)
    query = AsyncMock(return_value=[{"n": 7}])
    invalidate = AsyncMock()
    clear_skills = AsyncMock(return_value=3)
    monkeypatch.setattr(database, "transaction", transaction)
    monkeypatch.setattr(database, "execute_raw_with_schema", execute)
    monkeypatch.setattr(database, "query_raw_with_schema", query)
    monkeypatch.setattr(database, "_invalidate_user_caches", invalidate)
    monkeypatch.setattr(database, "delete_skills_cache", clear_skills)
    scrub = AsyncMock(return_value=5)
    monkeypatch.setattr(database, "scrub_ledger", scrub)
    monkeypatch.setattr(database, "count_unscrubbed", AsyncMock(return_value=6))
    return MagicMock(
        tx=tx,
        execute=execute,
        query=query,
        invalidate=invalidate,
        clear_skills=clear_skills,
        scrub=scrub,
    )


@pytest.mark.asyncio
async def test_a_dry_run_only_counts(db):
    report = await database.delete_rows(SUBJECT, dry_run=True)

    db.execute.assert_not_awaited()
    db.invalidate.assert_not_awaited()
    db.clear_skills.assert_not_awaited()
    assert report.status == StepStatus.WOULD_DO
    assert report.counts["user"] == 7
    assert report.counts["executions"] == 7
    assert report.counts["credit_transactions_scrubbed"] == 6
    db.scrub.assert_not_awaited()
    for call in db.query.await_args_list:
        assert call.args[0].startswith("SELECT COUNT(*)")


@pytest.mark.skipif(
    not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT),
    reason="account erasure cache test requires Redis",
)
@pytest.mark.asyncio
@pytest.mark.parametrize("dry_run", [True, False])
async def test_erasure_removes_only_its_skill_metadata_and_rebuild_claims(
    db, monkeypatch, dry_run
):
    user_id, other_id = str(uuid4()), str(uuid4())
    subject = SUBJECT.model_copy(update={"user_id": user_id, "expert_ids": []})
    redis = await get_redis_async()
    monkeypatch.setattr(
        database, "delete_skills_cache", skills_cache.delete_skills_cache
    )
    owned = [
        _skills_cache_key(user_id),
        _skills_cache_key(user_id, "already-deleted-expert"),
        _skills_cache_key(user_id, "rebuilding"),
        f"copilot:skills_index:v2:{user_id}",
        f"copilot:skills_heal_backoff:{user_id}:expert:already-deleted-expert",
        f"copilot:skills_update_backoff:{user_id}:autopilot",
        f"copilot:skills_update_backoff:{user_id}:already-deleted-expert",
    ]
    preserved = [
        _skills_cache_key(other_id),
        _skills_cache_key(other_id, "already-deleted-expert"),
        _skills_cache_key(other_id, user_id),
        _skills_cache_key(user_id + "-other"),
        f"copilot:skills_heal_backoff:{other_id}:expert:already-deleted-expert",
        f"copilot:skills_update_backoff:{other_id}:autopilot",
        f"unrelated:{user_id}",
    ]
    try:
        for key in owned + preserved:
            if key != _skills_cache_key(user_id, "rebuilding"):
                await redis.set(key, "private skill metadata", ex=900)
        claim = await _claim_skills_cache(user_id, "rebuilding")
        assert claim is not None

        report = await database.delete_rows(subject, dry_run=dry_run)

        assert report.status == (StepStatus.WOULD_DO if dry_run else StepStatus.DONE)
        for key in owned:
            assert bool(await redis.exists(key)) is dry_run, key
        for key in preserved:
            assert await redis.get(key) == "private skill metadata", key
        if not dry_run:
            assert report.counts["skill_cache_entries"] == len(owned)
            await _write_skills_cache(
                user_id,
                [ParsedSkill(name="stale", description="private", body="")],
                expert_id="rebuilding",
                claim=claim,
            )
            assert await redis.get(_skills_cache_key(user_id, "rebuilding")) is None
    finally:
        for key in owned + preserved:
            await redis.delete(key)


@pytest.mark.asyncio
async def test_the_identity_and_the_account_go_in_the_same_transaction(db):
    report = await database.delete_rows(SUBJECT, dry_run=False)

    in_tx = [
        call.args[0]
        for call in db.execute.await_args_list
        if call.kwargs.get("client") is db.tx
    ]
    identity = next(i for i, sql in enumerate(in_tx) if '"UserAuthIdentity"' in sql)
    user = next(
        i for i, sql in enumerate(in_tx) if 'DELETE FROM {schema_prefix}"User"' in sql
    )
    assert identity < user == len(in_tx) - 1
    assert all(sql.startswith("DELETE FROM") for sql in in_tx)
    assert report.status == StepStatus.DONE
    db.invalidate.assert_awaited_once_with(USER_ID, "sam@example.com")
    db.clear_skills.assert_awaited_once_with(USER_ID)
    assert report.counts["skill_cache_entries"] == 3


@pytest.mark.asyncio
async def test_a_skill_cache_failure_leaves_erasure_unfinished_for_retry(db):
    db.clear_skills.side_effect = ConnectionError("redis unavailable")

    with pytest.raises(ConnectionError, match="redis unavailable"):
        await database.delete_rows(SUBJECT, dry_run=False)


@pytest.mark.asyncio
@pytest.mark.parametrize("user_id", ["", "*", "[a-z]", USER_ID + ":*"])
async def test_invalid_cache_owner_ids_never_reach_redis(monkeypatch, user_id):
    connect = AsyncMock()
    monkeypatch.setattr(skills_cache, "get_redis_async", connect)

    with pytest.raises(ValueError):
        await skills_cache.delete_skills_cache(user_id)

    connect.assert_not_awaited()


@pytest.mark.asyncio
async def test_executions_go_in_batches_outside_the_transaction(db):
    db.execute.side_effect = [200, 200, 3] + [1] * 50

    report = await database.delete_rows(SUBJECT, dry_run=False)

    batches = [
        call
        for call in db.execute.await_args_list
        if '"AgentGraphExecution"' in call.args[0]
    ]
    assert len(batches) == 3
    assert all("client" not in call.kwargs for call in batches)
    assert report.counts["executions"] == 403


@pytest.mark.asyncio
async def test_the_ledger_is_scrubbed_in_the_same_transaction(db):
    report = await database.delete_rows(SUBJECT, dry_run=False)

    db.scrub.assert_awaited_once_with(USER_ID, client=db.tx)
    assert report.counts["credit_transactions_scrubbed"] == 5


@pytest.mark.asyncio
async def test_only_the_accounts_own_runs_are_deleted(db):
    await database.delete_rows(SUBJECT, dry_run=False)

    batch = next(
        call.args[0]
        for call in db.execute.await_args_list
        if '"AgentGraphExecution"' in call.args[0]
    )
    # Not runs of its agents by other accounts: those agents were handed over.
    assert '"AgentGraph"' not in batch
    assert 'e."userId" = $1' in batch


def test_rows_that_would_outlive_their_parent_go_before_it():
    """A preset's saved inputs and a notification batch's events point at
    their parent with ON DELETE SET NULL: deleted explicitly, inputs before
    their presets."""
    deletions = {d.label: d for d in database.planned_deletions(SUBJECT)}
    labels = list(deletions)
    assert labels.index("preset_inputs") < labels.index("presets")
    assert labels.index("notification_events") < labels.index("user")
    assert deletions["preset_inputs"].table == "AgentNodeExecutionInputOutput"
    assert '{schema_prefix}"AgentPreset"' in deletions["preset_inputs"].condition
    assert deletions["notification_events"].table == "NotificationEvent"
    assert (
        '{schema_prefix}"UserNotificationBatch"'
        in deletions["notification_events"].condition
    )
