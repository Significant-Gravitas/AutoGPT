"""Unit tests for helpers in backend.data.user."""

import asyncio
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock, call, patch

import prisma.errors
import pytest

from backend.data import user as user_module
from backend.data.notifications import AudienceAction, NotificationResult
from backend.data.user import (
    get_billing_email_recipient,
    is_marketing_opted_out,
    record_marketing_opt_out_by_email,
    record_signup_consent,
    update_user_timezone,
)
from backend.util.exceptions import DatabaseError, NotFoundError


def _application_user(user_id: str, email: str) -> user_module.User:
    now = datetime.now(timezone.utc)
    return user_module.User(
        id=user_id,
        email=email,
        created_at=now,
        updated_at=now,
    )


class TestUpdateUserTimezone:
    @pytest.mark.asyncio
    async def test_invalidates_all_three_user_caches(self):
        prisma_user = MagicMock(id="user-1", email="user@example.com")
        sentinel_user = MagicMock()

        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module.User, "from_db", return_value=sentinel_user),
            patch.object(user_module.get_user_by_id, "cache_delete") as by_id_del,
            patch.object(user_module.get_user_by_email, "cache_delete") as by_email_del,
            patch.object(user_module.get_or_create_user, "cache_clear") as goc_clear,
        ):
            mock_prisma_user.prisma.return_value.update = AsyncMock(
                return_value=prisma_user
            )
            result = await update_user_timezone("user-1", "Europe/London")

        assert result is sentinel_user
        by_id_del.assert_called_once_with("user-1")
        by_email_del.assert_called_once_with("user@example.com")
        goc_clear.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_skips_email_cache_invalidation_when_email_missing(self):
        prisma_user = MagicMock(id="user-1", email=None)
        sentinel_user = MagicMock()

        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module.User, "from_db", return_value=sentinel_user),
            patch.object(user_module.get_user_by_id, "cache_delete") as by_id_del,
            patch.object(user_module.get_user_by_email, "cache_delete") as by_email_del,
            patch.object(user_module.get_or_create_user, "cache_clear") as goc_clear,
        ):
            mock_prisma_user.prisma.return_value.update = AsyncMock(
                return_value=prisma_user
            )
            await update_user_timezone("user-1", "Europe/London")

        by_id_del.assert_called_once_with("user-1")
        by_email_del.assert_not_called()
        goc_clear.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_wraps_prisma_errors_in_database_error(self):
        with patch.object(user_module, "PrismaUser") as mock_prisma_user:
            mock_prisma_user.prisma.return_value.update = AsyncMock(
                side_effect=RuntimeError("connection lost")
            )
            with pytest.raises(DatabaseError) as exc:
                await update_user_timezone("user-1", "Europe/London")

        assert "user-1" in str(exc.value)
        assert "connection lost" in str(exc.value)

    @pytest.mark.asyncio
    async def test_eagerly_re_registers_dream_schedules_with_force_refresh(self):
        """APScheduler cron triggers bind to the timezone at registration
        time. A profile-page timezone change MUST eagerly re-register
        the dream-system crons so they fire at the right local time
        without waiting for the 7-day Redis dedup-key TTL to expire."""
        from backend.copilot.briefing import scheduling as briefing_scheduling
        from backend.copilot.dream import scheduling as dream_scheduling

        prisma_user = MagicMock(id="user-tz", email="user@example.com")
        captured: list[tuple[str, bool]] = []

        async def fake_ensure(user_id: str, *, force_refresh: bool = False):
            captured.append((user_id, force_refresh))
            return {}

        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module.User, "from_db", return_value=MagicMock()),
            patch.object(user_module.get_user_by_id, "cache_delete"),
            patch.object(user_module.get_user_by_email, "cache_delete"),
            patch.object(user_module.get_or_create_user, "cache_clear"),
            patch.object(
                dream_scheduling, "ensure_dream_system_scheduled", new=fake_ensure
            ),
            # This test isolates the dream-system re-registration contract;
            # the sibling morning-briefing re-registration (also wired here)
            # is covered separately by scheduling_test.py.
            patch.object(
                briefing_scheduling,
                "clear_briefing_registration_marker",
                new=AsyncMock(),
            ),
            patch.object(
                briefing_scheduling,
                "ensure_morning_briefing_scheduled",
                new=AsyncMock(),
            ),
        ):
            mock_prisma_user.prisma.return_value.update = AsyncMock(
                return_value=prisma_user
            )
            await update_user_timezone("user-tz", "Europe/Paris")
            # Yield once so the asyncio.create_task body runs before we
            # assert it was called.
            await asyncio.sleep(0)

        assert captured == [("user-tz", True)]

    @pytest.mark.asyncio
    async def test_re_register_task_is_retained_and_its_failure_logged(self):
        """The event loop holds only weak refs to tasks — an unretained
        fire-and-forget re-register can be GC'd mid-flight and its
        exception never observed. The spawn must keep a strong ref in
        ``_background_tasks`` until done and log failures via the
        done-callback instead of dropping them."""
        from backend.copilot.briefing import scheduling as briefing_scheduling
        from backend.copilot.dream import scheduling as dream_scheduling

        prisma_user = MagicMock(id="user-tz", email="user@example.com")

        async def failing_ensure(user_id: str, *, force_refresh: bool = False):
            raise RuntimeError("scheduler unreachable")

        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module.User, "from_db", return_value=MagicMock()),
            patch.object(user_module.get_user_by_id, "cache_delete"),
            patch.object(user_module.get_user_by_email, "cache_delete"),
            patch.object(user_module.get_or_create_user, "cache_clear"),
            patch.object(
                dream_scheduling, "ensure_dream_system_scheduled", new=failing_ensure
            ),
            # Isolate the dream-system task-retention contract under test
            # from the sibling morning-briefing re-registration (also wired
            # here) — real Redis/flag I/O in that path would otherwise give
            # the event loop extra turns and let the dream task above
            # complete (and get discarded) before the assertion below runs.
            patch.object(
                briefing_scheduling,
                "clear_briefing_registration_marker",
                new=AsyncMock(),
            ),
            patch.object(
                briefing_scheduling,
                "ensure_morning_briefing_scheduled",
                new=AsyncMock(),
            ),
            patch.object(user_module.logger, "warning") as warn_mock,
        ):
            mock_prisma_user.prisma.return_value.update = AsyncMock(
                return_value=prisma_user
            )
            await update_user_timezone("user-tz", "Europe/Paris")

            spawned = [
                t
                for t in user_module._background_tasks
                if t.get_name() == "tz-reregister-user-tz"
            ]
            assert spawned, "task must be strongly referenced until it completes"

            await asyncio.gather(*spawned, return_exceptions=True)
            # One more tick so the done-callback (scheduled via
            # call_soon) runs.
            await asyncio.sleep(0)

        assert not user_module._background_tasks & set(spawned)
        warn_mock.assert_called_once()
        assert isinstance(warn_mock.call_args.kwargs["exc_info"], RuntimeError)

    @pytest.mark.asyncio
    async def test_briefing_re_register_runs_in_background_clear_first(self):
        """The briefing re-register must not block the profile update
        (it's a spawned task, like the dream sibling) and must clear the
        stored marker before re-ensuring, or the drift check would read
        the just-superseded timezone and skip the re-register."""
        from backend.copilot.briefing import scheduling as briefing_scheduling
        from backend.copilot.dream import scheduling as dream_scheduling

        prisma_user = MagicMock(id="user-tz", email="user@example.com")
        calls: list[str] = []

        async def fake_clear(user_id: str):
            calls.append("clear")

        async def fake_ensure(user_id: str):
            calls.append("ensure")

        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module.User, "from_db", return_value=MagicMock()),
            patch.object(user_module.get_user_by_id, "cache_delete"),
            patch.object(user_module.get_user_by_email, "cache_delete"),
            patch.object(user_module.get_or_create_user, "cache_clear"),
            patch.object(
                dream_scheduling, "ensure_dream_system_scheduled", new=AsyncMock()
            ),
            patch.object(
                briefing_scheduling,
                "clear_briefing_registration_marker",
                new=fake_clear,
            ),
            patch.object(
                briefing_scheduling,
                "ensure_morning_briefing_scheduled",
                new=fake_ensure,
            ),
        ):
            mock_prisma_user.prisma.return_value.update = AsyncMock(
                return_value=prisma_user
            )
            await update_user_timezone("user-tz", "Europe/Paris")
            assert calls == []  # nothing ran inline — it's a background task

            spawned = [
                t
                for t in user_module._background_tasks
                if t.get_name() == "briefing-tz-reregister-user-tz"
            ]
            assert spawned, "briefing re-register task must be spawned + retained"
            await asyncio.gather(*spawned)

        assert calls == ["clear", "ensure"]


TERMS_VERSION = "2026-10"
EARLIER_TERMS_VERSION = "2025-01"
CONSENTED_AT = datetime(2026, 1, 5, 9, 30, tzinfo=timezone.utc)
TERMS_FIELDS = {"termsAcceptedAt", "termsVersion"}
OPT_OUT_FIELDS = {"marketingOptOutAt", "marketingOptOutSource"}


def _consent_row(
    terms_accepted_at: datetime | None = None,
    terms_version: str | None = None,
    marketing_opt_out_at: datetime | None = None,
    marketing_opt_out_source: str | None = None,
    email: str | None = "user@example.com",
) -> MagicMock:
    """A Prisma User row carrying only what record_signup_consent reads."""
    return MagicMock(
        id="user-1",
        email=email,
        termsAcceptedAt=terms_accepted_at,
        termsVersion=terms_version,
        marketingOptOutAt=marketing_opt_out_at,
        marketingOptOutSource=marketing_opt_out_source,
    )


def _accepted(opted_out: bool, version: str = TERMS_VERSION) -> MagicMock:
    return _consent_row(
        terms_accepted_at=CONSENTED_AT,
        terms_version=version,
        marketing_opt_out_at=CONSENTED_AT if opted_out else None,
        marketing_opt_out_source="signup" if opted_out else None,
    )


def _terms_written(db: MagicMock) -> dict[str, Any]:
    db.update.assert_awaited_once()
    assert db.update.await_args is not None
    assert db.update.await_args.kwargs["where"] == {"id": "user-1"}
    return db.update.await_args.kwargs["data"]


def _opt_out_written(db: MagicMock) -> dict[str, Any]:
    """Conditional on no refusal being stored, whatever the earlier read saw."""
    db.update_many.assert_awaited_once()
    assert db.update_many.await_args is not None
    assert db.update_many.await_args.kwargs["where"] == {
        "id": "user-1",
        "marketingOptOutAt": None,
    }
    return db.update_many.await_args.kwargs["data"]


class TestRecordSignupConsent:
    @pytest.fixture
    def db(self) -> Iterator[MagicMock]:
        """`PrismaUser.prisma()`, read and written. from_db hands the row back
        unchanged, so a test can tell which of the rows it got."""
        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module.User, "from_db", side_effect=lambda row: row),
        ):
            db = mock_prisma_user.prisma.return_value
            db.find_unique = AsyncMock(return_value=_consent_row())
            db.update = AsyncMock(return_value=_consent_row())
            db.update_many = AsyncMock(return_value=1)
            db.clients = mock_prisma_user.prisma
            yield db

    @pytest.fixture(autouse=True)
    def tx(self) -> Iterator[MagicMock]:
        """`transaction()`. `exits` records how each block was left: None for
        a commit, the exception for a rollback."""
        tx = MagicMock(name="tx")
        tx.exits = []

        @asynccontextmanager
        async def fake_transaction() -> AsyncIterator[MagicMock]:
            try:
                yield tx
            except BaseException as e:
                tx.exits.append(e)
                raise
            tx.exits.append(None)

        with patch.object(user_module, "transaction", fake_transaction):
            yield tx

    @pytest.fixture(autouse=True)
    def caches(self) -> Iterator[MagicMock]:
        """The three user caches as children of one mock, so `mock_calls` shows
        whether any of them was touched."""
        caches = MagicMock()
        with (
            patch.object(
                user_module.get_user_by_id, "cache_delete", caches.by_id_delete
            ),
            patch.object(
                user_module.get_user_by_email, "cache_delete", caches.by_email_delete
            ),
            patch.object(
                user_module.get_or_create_user, "cache_clear", caches.or_create_clear
            ),
        ):
            yield caches

    @pytest.fixture(autouse=True)
    def queued(self) -> Iterator[AsyncMock]:
        """`queue_audience_change`: what reached the MailerLite queue."""
        queued = AsyncMock(return_value=NotificationResult(success=True, message=""))
        with patch.object(user_module, "queue_audience_change", queued):
            yield queued

    @pytest.mark.asyncio
    async def test_first_call_stamps_the_terms_and_the_opt_out(self, db: MagicMock):
        fresh = _accepted(True)
        db.find_unique.side_effect = [_consent_row(), fresh]
        before = datetime.now(timezone.utc)

        result = await record_signup_consent("user-1", TERMS_VERSION, True)

        assert db.find_unique.await_args_list == [
            call(where={"id": "user-1"}),
            call(where={"id": "user-1"}),
        ]
        terms = _terms_written(db)
        opt_out = _opt_out_written(db)
        assert set(terms) == TERMS_FIELDS
        assert set(opt_out) == OPT_OUT_FIELDS
        assert terms["termsVersion"] == TERMS_VERSION
        assert opt_out["marketingOptOutSource"] == "signup"
        assert before <= terms["termsAcceptedAt"] <= datetime.now(timezone.utc)
        assert opt_out["marketingOptOutAt"] == terms["termsAcceptedAt"]
        assert result is fresh

    @pytest.mark.asyncio
    async def test_first_call_without_opt_out_stamps_only_the_terms(
        self, db: MagicMock
    ):
        await record_signup_consent("user-1", TERMS_VERSION, False)

        terms = _terms_written(db)
        assert set(terms) == TERMS_FIELDS
        assert terms["termsVersion"] == TERMS_VERSION
        db.update_many.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "opted_out,marketing_opt_out",
        [
            pytest.param(True, True, id="retried-opt-out"),
            pytest.param(True, False, id="no-opt-out-keeps-the-existing-one"),
            pytest.param(False, False, id="retried-without-opt-out"),
        ],
    )
    async def test_nothing_new_writes_nothing_and_keeps_the_caches(
        self,
        db: MagicMock,
        caches: MagicMock,
        opted_out: bool,
        marketing_opt_out: bool,
    ):
        current = _accepted(opted_out)
        db.find_unique.return_value = current

        result = await record_signup_consent("user-1", TERMS_VERSION, marketing_opt_out)

        assert result is current
        db.find_unique.assert_awaited_once()
        db.update.assert_not_awaited()
        db.update_many.assert_not_awaited()
        assert caches.mock_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("marketing_opt_out", [True, False])
    async def test_new_terms_version_restamps_the_terms_and_keeps_the_opt_out(
        self, db: MagicMock, marketing_opt_out: bool
    ):
        db.find_unique.return_value = _accepted(True, version=EARLIER_TERMS_VERSION)

        await record_signup_consent("user-1", TERMS_VERSION, marketing_opt_out)

        terms = _terms_written(db)
        assert set(terms) == TERMS_FIELDS
        assert terms["termsVersion"] == TERMS_VERSION
        assert terms["termsAcceptedAt"] > CONSENTED_AT
        db.update_many.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_version_without_an_acceptance_date_is_stamped(self, db: MagicMock):
        db.find_unique.return_value = _consent_row(terms_version=TERMS_VERSION)

        await record_signup_consent("user-1", TERMS_VERSION, False)

        assert set(_terms_written(db)) == TERMS_FIELDS

    @pytest.mark.asyncio
    async def test_opt_out_after_accepting_the_same_terms_writes_only_the_opt_out(
        self, db: MagicMock
    ):
        db.find_unique.return_value = _accepted(False)

        await record_signup_consent("user-1", TERMS_VERSION, True)

        db.update.assert_not_awaited()
        opt_out = _opt_out_written(db)
        assert set(opt_out) == OPT_OUT_FIELDS
        assert opt_out["marketingOptOutSource"] == "signup"
        assert opt_out["marketingOptOutAt"] > CONSENTED_AT

    @pytest.mark.asyncio
    async def test_a_refusal_recorded_since_the_read_is_not_overwritten(
        self, db: MagicMock, caches: MagicMock
    ):
        """The read saw no refusal, but one landed before the write (a
        concurrent call, or an unsubscribe). The conditional write matches no
        row, and the caller gets that refusal back."""
        unsubscribed = _consent_row(
            terms_accepted_at=CONSENTED_AT,
            terms_version=TERMS_VERSION,
            marketing_opt_out_at=CONSENTED_AT,
            marketing_opt_out_source="email_unsubscribe",
        )
        db.find_unique.side_effect = [_accepted(False), unsubscribed]
        db.update_many.return_value = 0

        result = await record_signup_consent("user-1", TERMS_VERSION, True)

        _opt_out_written(db)
        db.update.assert_not_awaited()
        assert result is unsubscribed
        assert caches.mock_calls == []

    @pytest.mark.asyncio
    async def test_a_write_invalidates_all_three_user_caches(
        self, db: MagicMock, caches: MagicMock
    ):
        db.find_unique.return_value = _consent_row(email="read@example.com")

        await record_signup_consent("user-1", TERMS_VERSION, True)

        caches.by_id_delete.assert_called_once_with("user-1")
        caches.by_email_delete.assert_called_once_with("read@example.com")
        caches.or_create_clear.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_a_write_skips_the_email_cache_when_the_row_has_no_email(
        self, db: MagicMock, caches: MagicMock
    ):
        db.find_unique.return_value = _consent_row(email=None)

        await record_signup_consent("user-1", TERMS_VERSION, True)

        caches.by_id_delete.assert_called_once_with("user-1")
        caches.by_email_delete.assert_not_called()
        caches.or_create_clear.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_missing_user_raises_not_found(
        self, db: MagicMock, caches: MagicMock
    ):
        db.find_unique.return_value = None

        with pytest.raises(NotFoundError, match="user-1"):
            await record_signup_consent("user-1", TERMS_VERSION, True)

        db.update.assert_not_awaited()
        db.update_many.assert_not_awaited()
        assert caches.mock_calls == []

    @pytest.mark.asyncio
    async def test_user_deleted_before_the_write_raises_not_found(
        self, db: MagicMock, caches: MagicMock
    ):
        """prisma-client-py's `update` catches Prisma's P2025 and returns None
        (prisma/actions.py), so this is what a deleted row looks like."""
        db.find_unique.side_effect = [_consent_row(), None]
        db.update.return_value = None
        db.update_many.return_value = 0

        with pytest.raises(NotFoundError, match="user-1"):
            await record_signup_consent("user-1", TERMS_VERSION, True)

        assert caches.mock_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing_call", ["update", "update_many"])
    async def test_a_record_not_found_error_is_a_not_found(
        self, db: MagicMock, caches: MagicMock, failing_call: str
    ):
        """Prisma's own error for a row that vanished mid-write maps to the
        404, not to a database failure."""
        getattr(db, failing_call).side_effect = prisma.errors.RecordNotFoundError(
            {"user_facing_error": {"message": "Record to update not found."}}
        )

        with pytest.raises(NotFoundError, match="user-1"):
            await record_signup_consent("user-1", TERMS_VERSION, True)

        assert caches.mock_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("marketing_opt_out", [True, False])
    async def test_an_older_version_never_replaces_a_newer_acceptance(
        self, db: MagicMock, marketing_opt_out: bool
    ):
        db.find_unique.return_value = _accepted(False)

        await record_signup_consent("user-1", EARLIER_TERMS_VERSION, marketing_opt_out)

        db.update.assert_not_called()
        if marketing_opt_out:
            assert set(_opt_out_written(db)) == OPT_OUT_FIELDS
        else:
            db.update_many.assert_not_called()

    @pytest.mark.asyncio
    async def test_both_writes_share_one_committed_transaction(
        self, db: MagicMock, tx: MagicMock
    ):
        await record_signup_consent("user-1", TERMS_VERSION, True)

        assert db.clients.call_args_list.count(call(tx)) == 2
        _terms_written(db)
        _opt_out_written(db)
        assert tx.exits == [None]

    @pytest.mark.asyncio
    async def test_a_failed_opt_out_write_rolls_back_the_terms(
        self, db: MagicMock, tx: MagicMock, caches: MagicMock
    ):
        failure = RuntimeError("opt-out write failed")
        db.update_many.side_effect = failure

        with pytest.raises(DatabaseError):
            await record_signup_consent("user-1", TERMS_VERSION, True)

        _terms_written(db)
        assert tx.exits == [failure]
        assert caches.mock_calls == []

    @pytest.mark.asyncio
    @pytest.mark.parametrize("failing_call", ["find_unique", "update", "update_many"])
    async def test_wraps_prisma_errors_in_database_error(
        self, db: MagicMock, caches: MagicMock, failing_call: str
    ):
        calls = {
            "find_unique": db.find_unique,
            "update": db.update,
            "update_many": db.update_many,
        }
        calls[failing_call].side_effect = RuntimeError("connection lost")

        with pytest.raises(DatabaseError) as exc:
            await record_signup_consent("user-1", TERMS_VERSION, True)

        assert "user-1" in str(exc.value)
        assert "connection lost" in str(exc.value)
        assert caches.mock_calls == []


class TestSignupRefusalReachesMailerLite:
    """A refusal this call records unsubscribes an existing MailerLite
    subscriber; nothing else ever queues a MailerLite change from here."""

    @pytest.fixture
    def db(self) -> Iterator[MagicMock]:
        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module.User, "from_db", side_effect=lambda row: row),
            patch.object(user_module.get_user_by_id, "cache_delete"),
            patch.object(user_module.get_user_by_email, "cache_delete"),
            patch.object(user_module.get_or_create_user, "cache_clear"),
        ):
            db = mock_prisma_user.prisma.return_value
            db.find_unique = AsyncMock(return_value=_consent_row())
            db.update = AsyncMock(return_value=_consent_row())
            db.update_many = AsyncMock(return_value=1)
            yield db

    @pytest.fixture(autouse=True)
    def tx(self) -> Iterator[None]:
        @asynccontextmanager
        async def fake_transaction() -> AsyncIterator[MagicMock]:
            yield MagicMock(name="tx")

        with patch.object(user_module, "transaction", fake_transaction):
            yield

    @pytest.fixture
    def queued(self) -> Iterator[AsyncMock]:
        queued = AsyncMock(return_value=NotificationResult(success=True, message=""))
        with patch.object(user_module, "queue_audience_change", queued):
            yield queued

    @pytest.mark.asyncio
    async def test_a_new_refusal_queues_an_unsubscribe(
        self, db: MagicMock, queued: AsyncMock
    ):
        await record_signup_consent("user-1", TERMS_VERSION, True)

        queued.assert_awaited_once()
        event = queued.await_args.args[0]
        assert event.action is AudienceAction.UNSUBSCRIBE
        assert event.email == "user@example.com"
        assert event.user_id == "user-1"
        assert event.fields == {}

    @pytest.mark.asyncio
    async def test_the_unsubscribe_is_queued_after_the_refusal_is_committed(
        self, db: MagicMock, queued: AsyncMock
    ):
        """Queued before the commit, the consumer could find the account still
        opted in, and a rolled-back refusal would still unsubscribe."""
        order: list[str] = []
        db.update_many.side_effect = lambda **_: order.append("write") or 1
        queued.side_effect = lambda _: order.append("queue") or NotificationResult(
            success=True, message=""
        )

        @asynccontextmanager
        async def fake_transaction() -> AsyncIterator[MagicMock]:
            yield MagicMock(name="tx")
            order.append("commit")

        with patch.object(user_module, "transaction", fake_transaction):
            await record_signup_consent("user-1", TERMS_VERSION, True)

        assert order == ["write", "commit", "queue"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "current,marketing_opt_out,written",
        [
            pytest.param(_accepted(True), True, 1, id="already-refused"),
            pytest.param(_consent_row(), False, 1, id="no-refusal"),
            pytest.param(
                _accepted(False, EARLIER_TERMS_VERSION), False, 1, id="new-terms-only"
            ),
            pytest.param(_consent_row(), True, 0, id="refused-concurrently"),
        ],
    )
    async def test_nothing_is_queued_without_a_new_refusal(
        self,
        db: MagicMock,
        queued: AsyncMock,
        current: MagicMock,
        marketing_opt_out: bool,
        written: int,
    ):
        db.find_unique.return_value = current
        db.update_many.return_value = written

        await record_signup_consent("user-1", TERMS_VERSION, marketing_opt_out)

        queued.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_account_without_an_email_queues_nothing(
        self, db: MagicMock, queued: AsyncMock
    ):
        db.find_unique.return_value = _consent_row(email=None)

        await record_signup_consent("user-1", TERMS_VERSION, True)

        queued.assert_not_awaited()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "failure",
        [
            pytest.param(
                {"return_value": NotificationResult(success=False, message="down")},
                id="not-queued",
            ),
            pytest.param({"side_effect": RuntimeError("broker gone")}, id="raised"),
        ],
    )
    async def test_a_queue_failure_still_returns_the_recorded_refusal(
        self, db: MagicMock, failure: dict, caplog
    ):
        """The refusal is stored either way, and the consumer re-reads it
        before every other MailerLite write, so the caller is not failed."""
        fresh = _accepted(True)
        db.find_unique.side_effect = [_consent_row(), fresh]

        with (
            patch.object(user_module, "queue_audience_change", AsyncMock(**failure)),
            caplog.at_level("ERROR"),
        ):
            result = await record_signup_consent("user-1", TERMS_VERSION, True)

        assert result is fresh
        assert "Could not queue the MailerLite unsubscribe for user user-1" in (
            caplog.text
        )
        assert "user@example.com" not in caplog.text


class TestRecordMarketingOptOutByEmail:
    """A refusal made outside the app (an unsubscribe in MailerLite)."""

    @pytest.fixture
    def db(self) -> Iterator[MagicMock]:
        with patch.object(user_module, "PrismaUser") as mock_prisma_user:
            db = mock_prisma_user.prisma.return_value
            db.find_unique = AsyncMock(return_value=_consent_row())
            db.find_many = AsyncMock(return_value=[])
            db.update_many = AsyncMock(return_value=1)
            yield db

    @pytest.fixture(autouse=True)
    def caches(self) -> Iterator[MagicMock]:
        caches = MagicMock()
        with (
            patch.object(
                user_module.get_user_by_id, "cache_delete", caches.by_id_delete
            ),
            patch.object(
                user_module.get_user_by_email, "cache_delete", caches.by_email_delete
            ),
            patch.object(
                user_module.get_or_create_user, "cache_clear", caches.or_create_clear
            ),
        ):
            yield caches

    @pytest.mark.asyncio
    async def test_records_the_refusal_with_its_source(
        self, db: MagicMock, caches: MagicMock
    ):
        before = datetime.now(timezone.utc)

        result = await record_marketing_opt_out_by_email(
            "user@example.com", "email_unsubscribe"
        )

        assert result == "user-1"
        db.find_unique.assert_awaited_once_with(where={"email": "user@example.com"})
        db.find_many.assert_not_awaited()
        data = _opt_out_written(db)
        assert data["marketingOptOutSource"] == "email_unsubscribe"
        assert before <= data["marketingOptOutAt"] <= datetime.now(timezone.utc)
        caches.by_id_delete.assert_called_once_with("user-1")
        caches.by_email_delete.assert_called_once_with("user@example.com")
        caches.or_create_clear.assert_called_once_with()

    @pytest.mark.asyncio
    async def test_falls_back_to_a_case_insensitive_match(self, db: MagicMock):
        db.find_unique.return_value = None
        db.find_many.return_value = [_consent_row(email="User@Example.com")]

        result = await record_marketing_opt_out_by_email(
            "user@example.com", "email_unsubscribe"
        )

        assert result == "user-1"
        db.find_many.assert_awaited_once_with(
            where={"email": {"equals": "user@example.com", "mode": "insensitive"}},
            take=2,
        )
        _opt_out_written(db)

    @pytest.mark.asyncio
    async def test_like_wildcards_in_the_address_match_only_themselves(
        self, db: MagicMock
    ):
        """The fallback is an ILIKE: unescaped, `john_smith@` would also match
        `john.smith@` and opt out someone else."""
        db.find_unique.return_value = None

        await record_marketing_opt_out_by_email(
            "john_smith%1@example.com", "email_unsubscribe"
        )

        db.find_many.assert_awaited_once_with(
            where={
                "email": {
                    "equals": "john\\_smith\\%1@example.com",
                    "mode": "insensitive",
                }
            },
            take=2,
        )

    @pytest.mark.asyncio
    async def test_an_ambiguous_case_insensitive_match_writes_nothing(
        self, db: MagicMock, caches: MagicMock
    ):
        db.find_unique.return_value = None
        db.find_many.return_value = [
            _consent_row(email="User@example.com"),
            _consent_row(email="USER@example.com"),
        ]

        assert (
            await record_marketing_opt_out_by_email(
                "user@example.com", "email_unsubscribe"
            )
            is None
        )
        db.update_many.assert_not_awaited()
        assert caches.mock_calls == []

    @pytest.mark.asyncio
    async def test_an_unknown_address_writes_nothing(
        self, db: MagicMock, caches: MagicMock
    ):
        db.find_unique.return_value = None

        assert (
            await record_marketing_opt_out_by_email(
                "x@example.com", "email_unsubscribe"
            )
            is None
        )
        db.update_many.assert_not_awaited()
        assert caches.mock_calls == []

    @pytest.mark.asyncio
    async def test_the_first_refusal_wins(self, db: MagicMock, caches: MagicMock):
        """Already opted out at signup: a later unsubscribe keeps that date and
        source, so a redelivered webhook changes nothing either."""
        db.find_unique.return_value = _accepted(True)

        assert (
            await record_marketing_opt_out_by_email(
                "user@example.com", "email_unsubscribe"
            )
            is None
        )
        db.update_many.assert_not_awaited()
        assert caches.mock_calls == []

    @pytest.mark.asyncio
    async def test_a_refusal_recorded_since_the_read_is_kept(
        self, db: MagicMock, caches: MagicMock
    ):
        db.update_many.return_value = 0

        assert (
            await record_marketing_opt_out_by_email(
                "user@example.com", "email_unsubscribe"
            )
            is None
        )
        _opt_out_written(db)
        assert caches.mock_calls == []

    @pytest.mark.asyncio
    async def test_a_database_failure_names_no_address(self, db: MagicMock):
        db.find_unique.side_effect = RuntimeError("connection lost")

        with pytest.raises(DatabaseError) as raised:
            await record_marketing_opt_out_by_email(
                "user@example.com", "email_unsubscribe"
            )

        assert "user@example.com" not in str(raised.value)


class TestIsMarketingOptedOut:
    """The audience consumer's last check before a MailerLite write."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "row,expected",
        [
            pytest.param(_accepted(True), True, id="opted-out"),
            pytest.param(_accepted(False), False, id="opted-in"),
            pytest.param(None, True, id="deleted-account"),
        ],
    )
    async def test_reads_the_row_not_the_cache(
        self, row: MagicMock | None, expected: bool
    ):
        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module, "get_user_by_id") as cached,
        ):
            db = mock_prisma_user.prisma.return_value
            db.find_unique = AsyncMock(return_value=row)

            assert await is_marketing_opted_out("user-1") is expected

        db.find_unique.assert_awaited_once_with(where={"id": "user-1"})
        cached.assert_not_called()

    @pytest.mark.asyncio
    async def test_a_failed_read_raises(self):
        with patch.object(user_module, "PrismaUser") as mock_prisma_user:
            mock_prisma_user.prisma.return_value.find_unique = AsyncMock(
                side_effect=RuntimeError("connection lost")
            )

            with pytest.raises(DatabaseError, match="user-1"):
                await is_marketing_opted_out("user-1")


class TestGetBillingEmailRecipient:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("opted_out_at", [None, CONSENTED_AT])
    async def test_maps_the_marketing_opt_out(self, opted_out_at: datetime | None):
        row = MagicMock(
            id="user-1",
            email="user@example.com",
            welcomeEmailSentAt=None,
            marketingOptOutAt=opted_out_at,
        )
        row.name = "User"

        with patch.object(user_module, "prisma") as mock_prisma:
            mock_prisma.user.find_first = AsyncMock(return_value=row)
            recipient = await get_billing_email_recipient("cus_123")

        mock_prisma.user.find_first.assert_awaited_once_with(
            where={"stripeCustomerId": "cus_123"}
        )
        assert recipient == user_module.BillingEmailRecipient(
            id="user-1",
            email="user@example.com",
            name="User",
            welcome_email_sent_at=None,
            marketing_opt_out_at=opted_out_at,
        )


class TestTableBackedCredentials:
    """get/set_user_credentials — the IntegrationCredential-backed seam the
    credential store runs on after the blob→table migration."""

    @pytest.mark.asyncio
    async def test_get_decrypts_active_user_rows(self, mocker):
        from unittest.mock import AsyncMock, MagicMock

        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials
        from backend.data.user import get_user_credentials
        from backend.util.encryption import JSONCryptor

        cred = APIKeyCredentials(
            id="cred-1", provider="github", api_key=SecretStr("sk-1"), title="GH"
        )
        row = MagicMock()
        row.id = "cred-1"
        row.encryptedPayload = JSONCryptor().encrypt(cred.model_dump())

        mock_prisma = MagicMock()
        mock_prisma.integrationcredential.find_many = AsyncMock(return_value=[row])
        mocker.patch("backend.data.user.prisma", mock_prisma)

        result = await get_user_credentials("u1")

        assert len(result) == 1
        assert result[0].id == "cred-1"
        assert result[0].api_key.get_secret_value() == "sk-1"
        where = mock_prisma.integrationcredential.find_many.call_args.kwargs["where"]
        assert where == {"ownerType": "USER", "ownerId": "u1", "status": "active"}

    @pytest.mark.asyncio
    async def test_set_updates_existing_creates_new_revokes_missing(self, mocker):
        from unittest.mock import AsyncMock, MagicMock

        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials
        from backend.data.user import set_user_credentials

        kept = APIKeyCredentials(
            id="cred-kept", provider="github", api_key=SecretStr("sk-2"), title="GH"
        )
        new = APIKeyCredentials(
            id="cred-new", provider="notion", api_key=SecretStr("sk-3"), title="N"
        )

        row_kept = MagicMock()
        row_kept.id = "cred-kept"
        row_kept.status = "active"
        row_gone = MagicMock()
        row_gone.id = "cred-gone"
        row_gone.status = "active"

        mock_prisma = MagicMock()
        mock_prisma.integrationcredential.find_many = AsyncMock(
            return_value=[row_kept, row_gone]
        )
        mock_prisma.integrationcredential.update = AsyncMock()
        mock_prisma.integrationcredential.create = AsyncMock()
        mock_prisma.organization.find_first = AsyncMock(
            return_value=MagicMock(id="org-personal")
        )
        mocker.patch("backend.data.user.prisma", mock_prisma)

        await set_user_credentials("u1", [kept, new])

        create_data = mock_prisma.integrationcredential.create.call_args.kwargs["data"]
        assert create_data["id"] == "cred-new"
        assert create_data["organizationId"] == "org-personal"

        update_calls = {
            c.kwargs["where"]["id"]: c.kwargs["data"]
            for c in mock_prisma.integrationcredential.update.call_args_list
        }
        # kept: payload refresh; gone: revoked
        assert "encryptedPayload" in update_calls["cred-kept"]
        assert update_calls["cred-gone"] == {"status": "revoked"}

    @pytest.mark.asyncio
    async def test_set_raises_without_personal_org_for_new_cred(self, mocker):
        from unittest.mock import AsyncMock, MagicMock

        from pydantic import SecretStr

        from backend.data.model import APIKeyCredentials
        from backend.data.user import set_user_credentials
        from backend.util.exceptions import DatabaseError

        new = APIKeyCredentials(
            id="cred-new", provider="notion", api_key=SecretStr("sk-3"), title="N"
        )
        mock_prisma = MagicMock()
        mock_prisma.integrationcredential.find_many = AsyncMock(return_value=[])
        mock_prisma.organization.find_first = AsyncMock(return_value=None)
        mocker.patch("backend.data.user.prisma", mock_prisma)

        with pytest.raises(DatabaseError):
            await set_user_credentials("u1", [new])


class TestGetOrCreateUserStatus:
    @pytest.fixture(autouse=True)
    def stub_user_provisioning(self):
        with (
            patch.object(user_module, "_ensure_user_profile", new_callable=AsyncMock),
            patch.object(
                user_module,
                "ensure_personal_org",
                new_callable=AsyncMock,
                return_value=False,
            ) as ensure_org,
            patch.object(
                user_module, "schedule_posthog_lifecycle_sync"
            ) as posthog_sync,
            patch.object(user_module, "AuthAccount") as auth_account,
            patch.object(user_module, "track_signup_completed"),
        ):
            auth_account.prisma.return_value.find_first = AsyncMock(return_value=None)
            yield ensure_org, posthog_sync

    @pytest.mark.asyncio
    async def test_reports_newly_created_user(self):
        db_user = MagicMock(id="user-new", email="alice@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(
                user_module.User,
                "from_db",
                return_value=_application_user("user-new", "alice@example.com"),
            ),
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=None)
            mock_prisma.user.create = AsyncMock(return_value=db_user)

            result = await user_module.get_or_create_user_with_status(
                {"sub": "user-new", "email": "alice@example.com"}
            )

        assert result.was_created is True

    @pytest.mark.asyncio
    async def test_reports_existing_user(self):
        db_user = MagicMock(id="user-existing", email="bob@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(
                user_module.User,
                "from_db",
                return_value=_application_user("user-existing", "bob@example.com"),
            ),
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)

            result = await user_module.get_or_create_user_with_status(
                {"sub": "user-existing", "email": "bob@example.com"}
            )

        assert result.was_created is False
        mock_prisma.user.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_reports_created_for_a_row_the_auth_hook_inserted_bare(
        self, stub_user_provisioning
    ):
        """The auth hook writes the User row before the client's
        ``POST /auth/user``, which then finds it. The first call to bootstrap
        that row's personal org is still the account's creation: it drives the
        sign-up conversion header and PostHog, exactly once."""
        ensure_org, posthog_sync = stub_user_provisioning
        ensure_org.return_value = True
        db_user = MagicMock(id="user-hooked", email="hook@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(
                user_module.User,
                "from_db",
                return_value=_application_user("user-hooked", "hook@example.com"),
            ),
            patch.object(user_module, "AuthAccount") as auth_account,
            patch.object(user_module, "track_signup_completed") as track,
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)
            auth_account.prisma.return_value.find_first = AsyncMock(
                return_value=MagicMock(providerId="credential")
            )

            result = await user_module.get_or_create_user_with_status(
                {"sub": "user-hooked", "email": "hook@example.com"}
            )

        assert result.was_created is True
        mock_prisma.user.create.assert_not_called()
        posthog_sync.assert_called_once_with("user-hooked")
        track.assert_called_once_with(user_id="user-hooked", signup_method="email")

    @pytest.mark.asyncio
    async def test_a_row_created_concurrently_is_read_back_not_an_error(self):
        """Two first requests for one account (the verify link opened in two
        browsers at once) both miss the row and both create it. The loser's
        unique violation on the id means the row exists: read it back."""
        db_user = MagicMock(id="user-raced", email="race@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(
                user_module.User,
                "from_db",
                return_value=_application_user("user-raced", "race@example.com"),
            ),
        ):
            mock_prisma.user.find_unique = AsyncMock(side_effect=[None, db_user])
            mock_prisma.user.create = AsyncMock(
                side_effect=prisma.errors.UniqueViolationError({})
            )

            result = await user_module.get_or_create_user_with_status(
                {"sub": "user-raced", "email": "race@example.com"}
            )

        assert result.was_created is False

    @pytest.mark.asyncio
    async def test_an_email_owned_by_another_user_still_fails(self):
        with patch.object(user_module, "prisma") as mock_prisma:
            mock_prisma.user.find_unique = AsyncMock(return_value=None)
            mock_prisma.user.create = AsyncMock(
                side_effect=prisma.errors.UniqueViolationError({})
            )

            with pytest.raises(DatabaseError):
                await user_module.get_or_create_user_with_status(
                    {"sub": "user-new", "email": "taken@example.com"}
                )

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "provider_id,expected", [("google", "google"), ("credential", "email")]
    )
    async def test_new_user_sends_signup_completed_with_the_better_auth_provider(
        self, provider_id: str, expected: str
    ):
        # A Better Auth token has no app_metadata; the account row has it.
        db_user = MagicMock(id="user-new", email="alice@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(user_module, "AuthAccount") as auth_account,
            patch.object(
                user_module.User,
                "from_db",
                return_value=_application_user("user-new", "alice@example.com"),
            ),
            patch.object(user_module, "track_signup_completed") as track,
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=None)
            mock_prisma.user.create = AsyncMock(return_value=db_user)
            find_first = AsyncMock(return_value=MagicMock(providerId=provider_id))
            auth_account.prisma.return_value.find_first = find_first

            await user_module.get_or_create_user_with_status(
                {"sub": "user-new", "email": "alice@example.com"}
            )

        assert find_first.await_args.kwargs["where"] == {"userId": "user-new"}
        # The user id only: email and name never go into event properties.
        track.assert_called_once_with(user_id="user-new", signup_method=expected)

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "lookup",
        [
            AsyncMock(return_value=None),
            AsyncMock(side_effect=RuntimeError("auth tables unreachable")),
        ],
    )
    async def test_new_user_without_an_auth_account_falls_back_to_the_token(
        self, lookup: AsyncMock
    ):
        db_user = MagicMock(id="user-new", email="alice@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(user_module, "AuthAccount") as auth_account,
            patch.object(
                user_module.User,
                "from_db",
                return_value=_application_user("user-new", "alice@example.com"),
            ),
            patch.object(user_module, "track_signup_completed") as track,
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=None)
            mock_prisma.user.create = AsyncMock(return_value=db_user)
            auth_account.prisma.return_value.find_first = lookup

            result = await user_module.get_or_create_user_with_status(
                {
                    "sub": "user-new",
                    "email": "alice@example.com",
                    "app_metadata": {"provider": "google", "providers": ["google"]},
                }
            )

        assert result.was_created is True
        track.assert_called_once_with(user_id="user-new", signup_method="google")

    @pytest.mark.asyncio
    async def test_existing_account_syncs_nothing(self, stub_user_provisioning):
        _, posthog_sync = stub_user_provisioning
        db_user = MagicMock(id="user-existing", email="bob@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(
                user_module.User,
                "from_db",
                return_value=_application_user("user-existing", "bob@example.com"),
            ),
            patch.object(user_module, "track_signup_completed") as track,
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)

            await user_module.get_or_create_user_with_status(
                {"sub": "user-existing", "email": "bob@example.com"}
            )

        posthog_sync.assert_not_called()
        track.assert_not_called()


@pytest.mark.parametrize(
    "user_data,expected",
    [
        ({"app_metadata": {"provider": "email"}}, "email"),
        ({"app_metadata": {"provider": ""}}, None),
        ({"app_metadata": "not-a-dict"}, None),
        ({}, None),
    ],
)
def test_legacy_signup_method_reads_the_supabase_provider(user_data: dict, expected):
    assert user_module._legacy_signup_method(user_data) == expected


class TestGetOrCreateUserProfile:
    """get_or_create_user must guarantee a marketplace Profile exists, since
    the auth.users trigger that used to do this is unreliable."""

    @pytest.fixture(autouse=True)
    def stub_ensure_personal_org(self):
        """Stub the personal-org bootstrap for the Profile-focused tests.

        The real bootstrap hits the DB; these tests only exercise the Profile
        branch. Tests that assert on the bootstrap use the yielded mock.
        """
        with patch.object(
            user_module,
            "ensure_personal_org",
            new_callable=AsyncMock,
            return_value=False,
        ) as m:
            yield m

    @pytest.mark.asyncio
    async def test_creates_profile_when_missing(self):
        user_module.get_or_create_user.cache_clear()
        db_user = MagicMock(id="user-new", email="alice@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(
                user_module.User,
                "from_db",
                return_value=_application_user("user-new", "alice@example.com"),
            ),
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)
            # No existing profile, and the generated username is free.
            mock_prisma.profile.find_unique = AsyncMock(return_value=None)
            mock_prisma.profile.create = AsyncMock()

            await user_module.get_or_create_user(
                {"sub": "user-new", "email": "alice@example.com"}
            )

        mock_prisma.profile.create.assert_awaited_once()
        created = mock_prisma.profile.create.await_args.kwargs["data"]
        assert created["userId"] == "user-new"
        # name defaults to the email local-part
        assert created["name"] == "alice"
        assert created["username"]

    @pytest.mark.asyncio
    async def test_does_not_create_profile_when_one_exists(self):
        user_module.get_or_create_user.cache_clear()
        db_user = MagicMock(id="user-has", email="bob@example.com", name="Bob")

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(
                user_module.User,
                "from_db",
                return_value=_application_user("user-has", "bob@example.com"),
            ),
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)
            mock_prisma.profile.find_unique = AsyncMock(return_value=MagicMock())
            mock_prisma.profile.create = AsyncMock()

            await user_module.get_or_create_user(
                {"sub": "user-has", "email": "bob@example.com"}
            )

        mock_prisma.profile.create.assert_not_called()

    @pytest.mark.asyncio
    async def test_profile_creation_failure_does_not_block_user(self):
        """Profile creation is best-effort: a failure is logged but the user
        is still resolved so login/auth isn't broken."""
        user_module.get_or_create_user.cache_clear()
        db_user = MagicMock(id="user-err", email="carol@example.com", name=None)
        sentinel_user = _application_user("user-err", "carol@example.com")

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(user_module.User, "from_db", return_value=sentinel_user),
            patch.object(user_module.logger, "warning") as warn_mock,
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)
            mock_prisma.profile.find_unique = AsyncMock(return_value=None)
            mock_prisma.profile.create = AsyncMock(
                side_effect=RuntimeError("db hiccup")
            )

            result = await user_module.get_or_create_user(
                {"sub": "user-err", "email": "carol@example.com"}
            )

        assert result is sentinel_user
        warn_mock.assert_called_once()

    @pytest.mark.asyncio
    async def test_retries_profile_create_on_username_collision(self):
        """A UniqueViolationError from a username clash (not a userId race)
        must retry with a fresh handle so the user still gets a Profile."""
        user_module.get_or_create_user.cache_clear()
        db_user = MagicMock(id="user-clash", email="dave@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(
                user_module.User,
                "from_db",
                return_value=_application_user("user-clash", "dave@example.com"),
            ),
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)
            # userId never resolves to a Profile (so the clash is on username),
            # and generated usernames pre-check as free.
            mock_prisma.profile.find_unique = AsyncMock(return_value=None)
            # First create() collides on username, the retry succeeds.
            mock_prisma.profile.create = AsyncMock(
                side_effect=[prisma.errors.UniqueViolationError({}), None]
            )

            await user_module.get_or_create_user(
                {"sub": "user-clash", "email": "dave@example.com"}
            )

        assert mock_prisma.profile.create.await_count == 2


class TestGetOrCreateUserPersonalOrg:
    """get_or_create_user must bootstrap a personal org so new sign-ups don't
    hit "No organization context available" on every org-scoped endpoint.

    Unlike the marketplace Profile, this is NOT best-effort: without an org the
    account is unusable, so a bootstrap failure must fail the request loudly.
    """

    @pytest.mark.asyncio
    async def test_bootstraps_personal_org_for_user(self):
        user_module.get_or_create_user.cache_clear()
        db_user = MagicMock(id="user-org", email="erin@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(
                user_module.User,
                "from_db",
                return_value=_application_user("user-org", "erin@example.com"),
            ),
            patch.object(user_module, "_ensure_user_profile", new_callable=AsyncMock),
            patch.object(
                user_module,
                "ensure_personal_org",
                new_callable=AsyncMock,
                return_value=False,
            ) as ensure_org,
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)

            await user_module.get_or_create_user(
                {"sub": "user-org", "email": "erin@example.com"}
            )

        ensure_org.assert_awaited_once_with("user-org")

    @pytest.mark.asyncio
    async def test_org_bootstrap_failure_fails_loudly(self):
        """A failed org bootstrap must raise (DatabaseError) — never return a
        bricked account. Contrast with the best-effort Profile branch."""
        user_module.get_or_create_user.cache_clear()
        db_user = MagicMock(id="user-brick", email="frank@example.com", name=None)

        with (
            patch.object(user_module, "prisma") as mock_prisma,
            patch.object(user_module.User, "from_db", return_value=MagicMock()),
            patch.object(user_module, "_ensure_user_profile", new_callable=AsyncMock),
            patch.object(
                user_module,
                "ensure_personal_org",
                new_callable=AsyncMock,
                side_effect=RuntimeError("org bootstrap exploded"),
            ),
        ):
            mock_prisma.user.find_unique = AsyncMock(return_value=db_user)

            with pytest.raises(DatabaseError) as exc:
                await user_module.get_or_create_user(
                    {"sub": "user-brick", "email": "frank@example.com"}
                )

        assert "org bootstrap exploded" in str(exc.value)


class TestGetAuthUserFlagFields:
    @pytest.mark.asyncio
    async def test_returns_flag_fields_for_existing_auth_user(self):
        created = datetime(2026, 5, 7, 12, 0, 0, tzinfo=timezone.utc)
        auth_user = MagicMock(role="admin", email="a@b.com", createdAt=created)

        with patch.object(user_module, "AuthUser") as mock_auth_user:
            mock_auth_user.prisma.return_value.find_unique = AsyncMock(
                return_value=auth_user
            )
            fields = await user_module.get_auth_user_flag_fields("user-1")

        assert fields is not None
        assert fields.role == "admin"
        assert fields.email == "a@b.com"
        assert fields.created_at == created
        mock_auth_user.prisma.return_value.find_unique.assert_called_once_with(
            where={"id": "user-1"}
        )

    @pytest.mark.asyncio
    async def test_returns_none_when_auth_user_missing(self):
        with patch.object(user_module, "AuthUser") as mock_auth_user:
            mock_auth_user.prisma.return_value.find_unique = AsyncMock(
                return_value=None
            )
            fields = await user_module.get_auth_user_flag_fields("ghost")

        assert fields is None


class TestGetUserDefaultChatRoute:
    @pytest.mark.asyncio
    async def test_returns_the_saved_route(self):
        user = _application_user("user-1", "user@example.com")
        user.default_chat_auth_provider = "codex"
        user.default_chat_credential_id = "cred-1"

        with patch.object(user_module, "get_user_by_id", AsyncMock(return_value=user)):
            route = await user_module.get_user_default_chat_route("user-1")

        assert route == ("codex", "cred-1")

    @pytest.mark.asyncio
    async def test_a_user_with_no_platform_row_yet_reads_as_nothing_saved(self):
        """Sign-up leaves a window before ``POST /auth/user`` creates the row.

        Transport discovery and session creation both read this on paths that
        never touched the user table before the setting existed, so raising
        here would take them down for a freshly signed-up account.
        """
        with patch.object(
            user_module,
            "get_user_by_id",
            AsyncMock(side_effect=ValueError("User not found with ID: user-1")),
        ):
            route = await user_module.get_user_default_chat_route("user-1")

        assert route == (None, None)

    @pytest.mark.asyncio
    async def test_an_old_cached_user_without_route_fields_reads_as_nothing_saved(self):
        user = _application_user("user-1", "user@example.com")
        user.__dict__.pop("default_chat_auth_provider", None)
        user.__dict__.pop("default_chat_credential_id", None)

        with patch.object(user_module, "get_user_by_id", AsyncMock(return_value=user)):
            route = await user_module.get_user_default_chat_route("user-1")

        assert route == (None, None)


class TestSetUserDefaultChatRoute:
    @pytest.mark.asyncio
    async def test_missing_user_is_reported_as_not_found(self):
        with patch.object(user_module, "PrismaUser") as mock_prisma_user:
            prisma_user = mock_prisma_user.prisma.return_value
            prisma_user.find_unique = AsyncMock(return_value=None)
            prisma_user.update_many = AsyncMock()

            with pytest.raises(NotFoundError, match="User user-1 not found"):
                await user_module.set_user_default_chat_route(
                    "user-1", "platform", None
                )

        prisma_user.update_many.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_clearing_provider_also_clears_stale_credential_id(self):
        prisma_user = MagicMock(id="user-1", email="user@example.com")

        with (
            patch.object(user_module, "PrismaUser") as mock_prisma_user,
            patch.object(user_module.get_user_by_id, "cache_delete") as by_id_del,
            patch.object(user_module.get_user_by_email, "cache_delete") as by_email_del,
            patch.object(user_module.get_or_create_user, "cache_clear") as goc_clear,
        ):
            prisma = mock_prisma_user.prisma.return_value
            prisma.find_unique = AsyncMock(return_value=prisma_user)
            prisma.update_many = AsyncMock(return_value=1)

            await user_module.set_user_default_chat_route(
                "user-1", None, "stale-credential"
            )

        prisma.update_many.assert_awaited_once_with(
            where={"id": "user-1"},
            data={
                "defaultChatAuthProvider": None,
                "defaultChatCredentialId": None,
            },
        )
        by_id_del.assert_called_once_with("user-1")
        by_email_del.assert_called_once_with("user@example.com")
        goc_clear.assert_called_once_with()


class TestHealOrphanedAuthIdentities:
    """Auth identity -> platform User is an invariant; the healer restores it
    and reports what it could not restore."""

    @staticmethod
    def _identity(user_id: str, email: str, email_owner_id: str | None = None):
        return user_module.OrphanedAuthIdentity(
            id=user_id,
            email=email,
            name="Someone",
            createdAt=datetime.now(timezone.utc),
            email_owner_id=email_owner_id,
        )

    @pytest.mark.asyncio
    async def test_provisions_each_orphan_from_its_own_email(self):
        orphans = [
            self._identity("auth-1", "one@example.com"),
            self._identity("auth-2", "two@example.com"),
        ]
        provision = AsyncMock()
        with (
            patch.object(
                user_module,
                "find_orphaned_auth_identities",
                AsyncMock(return_value=orphans),
            ) as find,
            patch.object(user_module, "get_or_create_user_with_status", provision),
        ):
            report = await user_module.heal_orphaned_auth_identities(
                grace_secs=300, limit=50
            )

        assert report.healed == ["auth-1", "auth-2"]
        assert report.collided == []
        assert report.failed == []
        # The same provisioning as POST /auth/user, keyed by the identity id
        # and its (already lowercased) auth email.
        assert provision.await_args_list[0].args[0] == {
            "sub": "auth-1",
            "email": "one@example.com",
            "user_metadata": {"name": "Someone"},
        }
        # Grace window: identities younger than this are still signing up.
        older_than = find.await_args.args[0]
        assert older_than < datetime.now(timezone.utc)
        assert find.await_args.args[1] == 50

    @pytest.mark.asyncio
    async def test_reports_an_email_collision_instead_of_guessing(self):
        # A different platform User already owns this email, so the unique
        # index makes the identity unprovisionable. That needs a human.
        collided = self._identity("auth-new", "taken@example.com", "user-old")
        provision = AsyncMock()
        with (
            patch.object(
                user_module,
                "find_orphaned_auth_identities",
                AsyncMock(return_value=[collided]),
            ),
            patch.object(user_module, "get_or_create_user_with_status", provision),
        ):
            report = await user_module.heal_orphaned_auth_identities()

        assert report.collided == [collided]
        assert report.healed == []
        provision.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_one_failure_does_not_stop_the_sweep(self):
        orphans = [
            self._identity("auth-1", "one@example.com"),
            self._identity("auth-2", "two@example.com"),
        ]
        provision = AsyncMock(side_effect=[DatabaseError("boom"), MagicMock()])
        with (
            patch.object(
                user_module,
                "find_orphaned_auth_identities",
                AsyncMock(return_value=orphans),
            ),
            patch.object(user_module, "get_or_create_user_with_status", provision),
        ):
            report = await user_module.heal_orphaned_auth_identities()

        assert report.failed == ["auth-1"]
        assert report.healed == ["auth-2"]
        assert not report.is_clean

    @pytest.mark.asyncio
    async def test_does_not_page_for_an_identity_provisioned_meanwhile(self):
        # A sign-in between the query and the heal already set the account
        # up, so nothing was broken.
        provision = AsyncMock(return_value=MagicMock(was_created=False))
        with (
            patch.object(
                user_module,
                "find_orphaned_auth_identities",
                AsyncMock(return_value=[self._identity("auth-1", "one@example.com")]),
            ),
            patch.object(user_module, "get_or_create_user_with_status", provision),
        ):
            report = await user_module.heal_orphaned_auth_identities()

        provision.assert_awaited_once()
        assert report.is_clean

    @pytest.mark.asyncio
    async def test_clean_when_nothing_is_orphaned(self):
        with patch.object(
            user_module, "find_orphaned_auth_identities", AsyncMock(return_value=[])
        ):
            report = await user_module.heal_orphaned_auth_identities()

        assert report.is_clean


class TestFindOrphanedAuthIdentities:
    @pytest.mark.asyncio
    async def test_queries_identities_without_a_user_row(self):
        cutoff = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
        rows = [
            {
                "id": "auth-1",
                "email": "one@example.com",
                "name": None,
                "createdAt": cutoff,
                "email_owner_id": None,
            }
        ]
        with patch.object(
            user_module, "query_raw_with_schema", AsyncMock(return_value=rows)
        ) as query:
            found = await user_module.find_orphaned_auth_identities(cutoff, limit=7)

        assert [i.id for i in found] == ["auth-1"]
        assert found[0].has_email_collision is False
        sql, older_than, limit = query.await_args.args
        assert '"UserAuthIdentity" a' in sql
        assert "u.id IS NULL" in sql
        # A migrated identity may differ from its platform row only by case;
        # a case-sensitive owner match would heal a duplicate account.
        assert "LOWER(owner.email) = LOWER(a.email)" in sql
        # ... and the owner must be a scalar subquery, not a join: several
        # case-variant platform rows would otherwise return the identity once
        # per variant and let duplicates consume the batch limit.
        assert "LIMIT 1) AS email_owner_id" in sql
        assert "JOIN" not in sql.split("AS email_owner_id")[0]
        # An unverified identity that never held a session is waiting on its
        # verification link, not orphaned.
        assert 'a."emailVerified" OR EXISTS' in sql
        assert '"UserAuthSession" s WHERE s."userId" = a.id' in sql
        # Collisions are never healed, so they sort behind healable rows.
        order_by = sql.split("ORDER BY EXISTS")[1]
        assert "LOWER(o.email) = LOWER(a.email)" in order_by
        assert older_than == cutoff.isoformat()
        assert limit == 7

    @pytest.mark.asyncio
    async def test_maps_a_foreign_email_owner_to_a_collision(self):
        # The SQL aliases the owning platform User's id as email_owner_id; the
        # model must read that back as a collision when it is someone else.
        cutoff = datetime(2026, 9, 14, 12, 0, tzinfo=timezone.utc)
        rows = [
            {
                "id": "auth-new",
                "email": "Taken@Example.com",
                "name": "Someone",
                "createdAt": cutoff,
                "email_owner_id": "user-old",
            }
        ]
        with patch.object(
            user_module, "query_raw_with_schema", AsyncMock(return_value=rows)
        ):
            found = await user_module.find_orphaned_auth_identities(cutoff)

        assert found[0].email_owner_id == "user-old"
        assert found[0].has_email_collision is True
