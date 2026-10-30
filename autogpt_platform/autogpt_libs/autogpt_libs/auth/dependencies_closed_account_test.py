"""A closed account must be rejected by the backend itself.

Tokens are verified statelessly and live for an hour, so ending the account's
sessions does not stop a token already minted. ``get_user_id`` and
``get_request_context`` reject a token whose subject asked to delete their
account, and nothing may self-heal its User row back into existence.
"""

import asyncio
import sys
import types
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi import FastAPI, HTTPException, Request, Security
from fastapi.security import HTTPAuthorizationCredentials
from fastapi.testclient import TestClient
from pytest_mock import MockerFixture

from autogpt_libs.auth import (
    closed_accounts,
    dependencies,
    forget_account_state,
    remember_account_closed,
)
from autogpt_libs.auth.dependencies import (
    get_optional_user_id,
    get_request_context,
    get_user_id,
    get_user_id_allowing_closed,
    requires_admin_user,
    requires_user,
)
from autogpt_libs.auth.jwt_utils import get_jwt_payload

CLOSED = "closed-user"
OPEN = "open-user"
BANNED = "banned-user"


def _request(headers: dict | None = None):
    request = Mock(spec=Request)
    request.headers = headers or {}
    request.method = "GET"
    request.url = "http://test/api/library/agents"
    return request


def _payload(user_id: str, role: str = "user") -> dict:
    return {"sub": user_id, "role": role, "email": f"{user_id}@example.com"}


@pytest.fixture(autouse=True)
def _fresh_caches():
    def clear():
        closed_accounts._CLOSED_ACCOUNT_CACHE.clear()
        closed_accounts._LOOKUPS_IN_FLIGHT.clear()
        dependencies._PROVISIONED_USER_IDS.clear()
        dependencies._HEALS_IN_FLIGHT.clear()

    clear()
    yield
    clear()


@pytest.fixture
def backend(mocker: MockerFixture):
    """Stub the deferred ``backend.*`` imports: a connected DB in which only
    ``CLOSED`` has an AccountDeletionRequest row."""
    org_member = Mock(
        orgId="org-1",
        status="ACTIVE",
        isOwner=True,
        isAdmin=False,
        isBillingManager=False,
    )
    org_member.Org = Mock(deletedAt=None)

    db_mod = types.ModuleType("backend.data.db")
    mocker.patch.object(db_mod, "prisma", Mock(), create=True)
    db_mod.prisma.is_connected = Mock(return_value=True)
    db_mod.prisma.orgmember.find_first = AsyncMock(return_value=org_member)
    db_mod.prisma.orgmember.find_unique = AsyncMock(return_value=org_member)

    deletion_mod = types.ModuleType("backend.data.account_deletion")
    is_closed = AsyncMock(side_effect=lambda user_id: user_id == CLOSED)
    mocker.patch.object(deletion_mod, "is_account_closed", is_closed, create=True)

    mocker.patch.dict(
        sys.modules,
        {
            "backend": types.ModuleType("backend"),
            "backend.data": types.ModuleType("backend.data"),
            "backend.data.db": db_mod,
            "backend.data.account_deletion": deletion_mod,
        },
    )
    heal = mocker.patch(
        "autogpt_libs.auth.dependencies._ensure_platform_user",
        new_callable=AsyncMock,
        return_value=True,
    )
    return types.SimpleNamespace(prisma=db_mod.prisma, is_closed=is_closed, heal=heal)


class TestClosedAccountIsRejected:
    @pytest.mark.asyncio
    async def test_get_user_id_rejects_a_closed_subject(self, backend):
        with pytest.raises(HTTPException) as exc_info:
            await get_user_id(_request(), _payload(CLOSED))

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "This account is closed."

    @pytest.mark.asyncio
    async def test_a_closed_account_is_never_healed(self, backend):
        """Healing would re-create the User row the erasure removes."""
        with pytest.raises(HTTPException):
            await get_user_id(_request(), _payload(CLOSED))

        backend.heal.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_request_context_rejects_a_closed_subject(self, backend):
        with pytest.raises(HTTPException) as exc_info:
            await get_request_context(_request(), _payload(CLOSED))

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "This account is closed."
        backend.prisma.orgmember.find_first.assert_not_awaited()
        backend.heal.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_open_subject_is_unchanged(self, backend):
        payload = _payload(OPEN)

        assert await get_user_id(_request(), payload) == OPEN
        ctx = await get_request_context(_request(), payload)

        assert ctx.user_id == OPEN
        assert ctx.org_id == "org-1"
        backend.heal.assert_awaited_once_with(OPEN, payload)

    def test_the_route_answers_401(self, backend):
        app = FastAPI()

        @app.get("/me")
        def me(user_id: str = Security(get_user_id)):
            return {"user_id": user_id}

        app.dependency_overrides[get_jwt_payload] = lambda: _payload(CLOSED)
        response = TestClient(app).get("/me")

        assert response.status_code == 401
        assert response.json() == {"detail": "This account is closed."}


class TestRouteLevelDependencies:
    """``requires_user`` and ``requires_admin_user`` guard routes that never
    reach ``get_user_id``: POST /auth/user provisions a User row straight from
    the token, and the admin routes act on other users' data."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("dependency", [requires_user, requires_admin_user])
    async def test_reject_a_closed_subject(self, backend, dependency):
        with pytest.raises(HTTPException) as exc_info:
            await dependency(_payload(CLOSED, "admin"))

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "This account is closed."

    @pytest.mark.asyncio
    @pytest.mark.parametrize("dependency", [requires_user, requires_admin_user])
    async def test_an_open_subject_passes_without_a_heal(self, backend, dependency):
        user = await dependency(_payload(OPEN, "admin"))

        assert user.user_id == OPEN
        backend.heal.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_failed_check_lets_the_request_through(
        self, backend, mocker: MockerFixture
    ):
        mocker.patch("autogpt_libs.auth.closed_accounts.logger")
        backend.is_closed.side_effect = RuntimeError("db down")

        assert (await requires_user(_payload(CLOSED))).user_id == CLOSED

    def test_a_route_that_provisions_from_the_token_answers_401(self, backend):
        app = FastAPI()
        provision = Mock()

        @app.post("/auth/user", dependencies=[Security(requires_user)])
        def get_or_create_user(payload: dict = Security(get_jwt_payload)):
            provision(payload)
            return {}

        app.dependency_overrides[get_jwt_payload] = lambda: _payload(CLOSED)
        response = TestClient(app).post("/auth/user")

        assert response.status_code == 401
        assert response.json() == {"detail": "This account is closed."}
        provision.assert_not_called()


class TestImpersonation:
    @pytest.mark.asyncio
    async def test_checks_the_token_subject_not_the_target(self, backend):
        request = _request({"X-Act-As-User-Id": CLOSED})

        assert await get_user_id(request, _payload("admin-1", "admin")) == CLOSED
        ctx = await get_request_context(request, _payload("admin-1", "admin"))

        assert ctx.user_id == CLOSED
        assert [c.args[0] for c in backend.is_closed.await_args_list] == ["admin-1"]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("dependency", [get_user_id, get_request_context])
    async def test_a_closed_admin_cannot_impersonate(self, backend, dependency):
        request = _request({"X-Act-As-User-Id": OPEN})

        with pytest.raises(HTTPException) as exc_info:
            await dependency(request, _payload(CLOSED, "admin"))

        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "This account is closed."


class TestClosedAccountCache:
    @pytest.mark.asyncio
    async def test_a_cached_answer_avoids_a_second_query(self, backend):
        await get_user_id(_request(), _payload(OPEN))
        await get_user_id(_request(), _payload(OPEN))
        for _ in range(2):
            with pytest.raises(HTTPException):
                await get_user_id(_request(), _payload(CLOSED))

        assert [c.args[0] for c in backend.is_closed.await_args_list] == [
            OPEN,
            CLOSED,
        ]

    @pytest.mark.asyncio
    async def test_an_open_answer_is_kept_only_briefly(
        self, backend, mocker: MockerFixture
    ):
        """Another worker may close the account at any moment."""
        clock = mocker.patch("autogpt_libs.auth.closed_accounts.time.monotonic")
        clock.return_value = 1_000.0
        await get_user_id(_request(), _payload(OPEN))
        clock.return_value = 1_000.0 + closed_accounts._ACCOUNT_STATE_TTL_SECS - 1
        await get_user_id(_request(), _payload(OPEN))
        assert backend.is_closed.await_count == 1

        clock.return_value = 1_000.0 + closed_accounts._ACCOUNT_STATE_TTL_SECS
        await get_user_id(_request(), _payload(OPEN))
        assert backend.is_closed.await_count == 2
        assert closed_accounts._ACCOUNT_STATE_TTL_SECS == 30

    @pytest.mark.asyncio
    async def test_a_closed_answer_is_kept_only_briefly_too(
        self, backend, mocker: MockerFixture
    ):
        """Its owner can reopen the account within 14 days, from any worker."""
        clock = mocker.patch("autogpt_libs.auth.closed_accounts.time.monotonic")
        clock.return_value = 1_000.0
        for offset in (0, closed_accounts._ACCOUNT_STATE_TTL_SECS - 1):
            clock.return_value = 1_000.0 + offset
            with pytest.raises(HTTPException):
                await get_user_id(_request(), _payload(CLOSED))
        assert backend.is_closed.await_count == 1

        clock.return_value = 1_000.0 + closed_accounts._ACCOUNT_STATE_TTL_SECS
        with pytest.raises(HTTPException):
            await get_user_id(_request(), _payload(CLOSED))
        assert backend.is_closed.await_count == 2

    @pytest.mark.asyncio
    async def test_the_cache_is_bounded(self, backend, mocker: MockerFixture):
        mocker.patch.object(closed_accounts, "_CLOSED_ACCOUNT_CACHE_MAX", 2)

        for user_id in ("user-1", "user-2", "user-3"):
            await get_user_id(_request(), _payload(user_id))

        assert list(closed_accounts._CLOSED_ACCOUNT_CACHE) == ["user-2", "user-3"]

    @pytest.mark.asyncio
    async def test_remember_account_closed_rejects_without_a_query(self, backend):
        """The worker that closed the account must not wait out the cached
        "open" answer it gave a moment ago."""
        assert await get_user_id(_request(), _payload(OPEN)) == OPEN

        remember_account_closed(OPEN)

        with pytest.raises(HTTPException) as exc_info:
            await get_user_id(_request(), _payload(OPEN))
        assert exc_info.value.status_code == 401
        assert backend.is_closed.await_count == 1

    @pytest.mark.asyncio
    async def test_a_close_remembered_mid_query_wins(self, backend):
        async def closed_while_querying(user_id: str) -> bool:
            remember_account_closed(user_id)
            return False

        backend.is_closed.side_effect = closed_while_querying

        for _ in range(2):
            with pytest.raises(HTTPException):
                await get_user_id(_request(), _payload(OPEN))

    @pytest.mark.asyncio
    async def test_forget_account_state_drops_a_remembered_close(self, backend):
        remember_account_closed(OPEN)

        forget_account_state(OPEN)

        assert await get_user_id(_request(), _payload(OPEN)) == OPEN
        assert backend.is_closed.await_count == 1


class TestConcurrentLookups:
    """A first page load fans out ~20 requests at once; they must share one
    query instead of each paying for its own."""

    @pytest.fixture
    def blocked_query(self, mocker: MockerFixture):
        release = asyncio.Event()
        calls: list[str] = []

        async def query(user_id: str) -> bool:
            calls.append(user_id)
            await release.wait()
            return False

        mocker.patch.object(
            closed_accounts, "_closed_account_query", return_value=query
        )
        return types.SimpleNamespace(release=release, calls=calls)

    @pytest.mark.asyncio
    async def test_a_burst_runs_one_query(self, blocked_query):
        lookups = [
            asyncio.ensure_future(closed_accounts.get_account_state(OPEN))
            for _ in range(20)
        ]
        await asyncio.sleep(0)
        blocked_query.release.set()

        states = await asyncio.gather(*lookups)

        assert blocked_query.calls == [OPEN]
        assert states == [closed_accounts.AccountState.OPEN] * 20
        assert closed_accounts._LOOKUPS_IN_FLIGHT == {}

    @pytest.mark.asyncio
    async def test_a_close_remembered_while_waiting_wins_for_every_waiter(
        self, blocked_query
    ):
        lookups = [
            asyncio.ensure_future(closed_accounts.get_account_state(OPEN))
            for _ in range(5)
        ]
        await asyncio.sleep(0)
        remember_account_closed(OPEN)
        blocked_query.release.set()

        states = await asyncio.gather(*lookups)

        assert blocked_query.calls == [OPEN]
        assert states == [closed_accounts.AccountState.CLOSED] * 5

    @pytest.mark.asyncio
    async def test_a_cancelled_request_does_not_cancel_the_shared_lookup(
        self, blocked_query
    ):
        first = asyncio.ensure_future(closed_accounts.get_account_state(OPEN))
        second = asyncio.ensure_future(closed_accounts.get_account_state(OPEN))
        await asyncio.sleep(0)
        first.cancel()
        blocked_query.release.set()

        assert await second is closed_accounts.AccountState.OPEN
        assert blocked_query.calls == [OPEN]


class TestGetUserIdAllowingClosed:
    @pytest.mark.asyncio
    async def test_returns_the_id_of_a_closed_account_without_healing(self, backend):
        assert await get_user_id_allowing_closed(_request(), _payload(CLOSED)) == (
            CLOSED
        )

        backend.heal.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_an_open_account_resolves_like_get_user_id(self, backend):
        payload = _payload(OPEN)

        assert await get_user_id_allowing_closed(_request(), payload) == OPEN

        backend.heal.assert_awaited_once_with(OPEN, payload)

    @pytest.mark.asyncio
    async def test_supports_impersonation(self, backend):
        request = _request({"X-Act-As-User-Id": CLOSED})

        assert (
            await get_user_id_allowing_closed(request, _payload("admin-1", "admin"))
            == CLOSED
        )

    @pytest.mark.asyncio
    async def test_a_non_admin_still_cannot_impersonate(self, backend):
        request = _request({"X-Act-As-User-Id": CLOSED})

        with pytest.raises(HTTPException) as exc_info:
            await get_user_id_allowing_closed(request, _payload(OPEN))

        assert exc_info.value.status_code == 403


@pytest.fixture
def bans(backend, mocker: MockerFixture):
    """Add the ban lookup: ``BANNED`` is banned by an admin. Deleting an
    account does not ban it, so ``CLOSED`` is not banned."""
    db_mod = sys.modules["backend.data.db"]
    mocker.patch.object(db_mod, "is_connected", Mock(return_value=True), create=True)
    mocker.patch.object(sys.modules["backend.data"], "db", db_mod, create=True)
    user_mod = types.ModuleType("backend.data.user")
    is_banned = AsyncMock(side_effect=lambda user_id: user_id == BANNED)
    mocker.patch.object(user_mod, "is_user_banned", is_banned, create=True)
    mocker.patch.dict(sys.modules, {"backend.data.user": user_mod})
    return is_banned


class TestBannedAndClosed:
    """A closed account is answered as closed. A banned one is refused
    everywhere except the account deletion routes: deleting the account is
    the person's right whatever the ban."""

    @staticmethod
    def _client(user_id: str) -> TestClient:
        app = FastAPI()

        @app.get("/me")
        def me(uid: str = Security(get_user_id)):
            return {"user_id": uid}

        @app.post("/account/deletion")
        def delete_account(uid: str = Security(get_user_id_allowing_closed)):
            return {"user_id": uid}

        app.dependency_overrides[get_jwt_payload] = lambda: _payload(user_id)
        return TestClient(app)

    def test_a_deleted_account_is_told_it_is_closed(self, bans):
        response = self._client(CLOSED).get("/me")

        assert response.status_code == 401
        assert response.json() == {"detail": "This account is closed."}

    def test_the_deletion_route_still_answers_a_deleted_account(self, bans):
        response = self._client(CLOSED).post("/account/deletion")

        assert response.status_code == 200
        assert response.json() == {"user_id": CLOSED}

    def test_an_admin_ban_is_refused_elsewhere(self, bans):
        response = self._client(BANNED).get("/me")

        assert response.status_code == 403
        assert response.json()["detail"]["code"] == "account_banned"

    def test_a_banned_account_may_still_ask_to_delete_itself(self, bans):
        response = self._client(BANNED).post("/account/deletion")

        assert response.status_code == 200
        assert response.json() == {"user_id": BANNED}
        bans.assert_not_awaited()


class TestOptionalAuth:
    """An optional-auth route answers a closed account as someone signed out,
    as it answers a banned one."""

    @staticmethod
    async def _resolve(user_id: str, mocker: MockerFixture) -> str | None:
        mocker.patch(
            "autogpt_libs.auth.dependencies.parse_jwt_token_async",
            new=AsyncMock(return_value=_payload(user_id)),
        )
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials="t")
        return await get_optional_user_id(credentials)

    @pytest.mark.asyncio
    async def test_a_closed_account_reads_as_signed_out(
        self, bans, backend, mocker: MockerFixture
    ):
        assert await self._resolve(CLOSED, mocker) is None
        backend.is_closed.assert_awaited_once_with(CLOSED)

    @pytest.mark.asyncio
    async def test_a_banned_account_reads_as_signed_out(self, bans, mocker):
        assert await self._resolve(BANNED, mocker) is None

    @pytest.mark.asyncio
    async def test_an_open_account_reads_as_itself(self, bans, mocker):
        assert await self._resolve(OPEN, mocker) == OPEN


class TestFailsOpen:
    """Without a way to ask, the account is treated as open: the check must
    never turn a request that would have worked into an error."""

    @pytest.mark.asyncio
    async def test_without_a_database_connection(self, backend):
        backend.prisma.is_connected.return_value = False

        assert await get_user_id(_request(), _payload(CLOSED)) == CLOSED

        backend.is_closed.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_without_the_backend_package(self, mocker: MockerFixture):
        mocker.patch.dict(
            sys.modules,
            {
                "backend": None,
                "backend.data": None,
                "backend.data.db": None,
                "backend.data.account_deletion": None,
            },
        )

        assert await get_user_id(_request(), _payload(CLOSED)) == CLOSED

    @pytest.mark.asyncio
    async def test_on_a_database_error_with_a_warning(
        self, backend, mocker: MockerFixture
    ):
        logger = mocker.patch("autogpt_libs.auth.closed_accounts.logger")
        backend.is_closed.side_effect = RuntimeError("db down")

        assert await get_user_id(_request(), _payload(CLOSED)) == CLOSED
        assert await get_user_id_allowing_closed(_request(), _payload(CLOSED)) == (
            CLOSED
        )

        logger.warning.assert_called_once()

    @pytest.mark.asyncio
    async def test_a_failed_check_is_backed_off_like_an_open_answer(
        self, backend, mocker: MockerFixture
    ):
        """A DB outage costs one query per user per TTL, not one per request."""
        mocker.patch("autogpt_libs.auth.closed_accounts.logger")
        clock = mocker.patch("autogpt_libs.auth.closed_accounts.time.monotonic")
        backend.is_closed.side_effect = RuntimeError("db down")

        clock.return_value = 1_000.0
        await get_user_id(_request(), _payload(CLOSED))
        clock.return_value = 1_000.0 + closed_accounts._ACCOUNT_STATE_TTL_SECS - 1
        await get_user_id(_request(), _payload(CLOSED))
        assert backend.is_closed.await_count == 1

        clock.return_value = 1_000.0 + closed_accounts._ACCOUNT_STATE_TTL_SECS
        await get_user_id(_request(), _payload(CLOSED))
        assert backend.is_closed.await_count == 2


class TestNoHealWhenTheCheckFailed:
    """An account the check could not answer for may be closed, and healing
    a closed account would re-create the User row its erasure removes."""

    @pytest.fixture
    def failing_check(self, backend, mocker: MockerFixture):
        mocker.patch("autogpt_libs.auth.closed_accounts.logger")
        backend.is_closed.side_effect = RuntimeError("db down")
        return backend

    @pytest.mark.asyncio
    async def test_get_user_id_does_not_heal(self, failing_check):
        await get_user_id(_request(), _payload(CLOSED))
        await get_user_id_allowing_closed(_request(), _payload(CLOSED))

        failing_check.heal.assert_not_awaited()

    @pytest.fixture
    def no_personal_org(self, backend, mocker: MockerFixture):
        """Send ``get_request_context`` down its org bootstrap branch."""
        backend.prisma.orgmember.find_first = AsyncMock(return_value=None)
        backend.heal.return_value = False
        orgs_mod = types.ModuleType("backend.api.features.orgs.db")
        mocker.patch.object(
            orgs_mod,
            "get_user_default_team",
            AsyncMock(return_value=("org-1", "team-1")),
            create=True,
        )
        mocker.patch.dict(
            sys.modules,
            {
                "backend.api": types.ModuleType("backend.api"),
                "backend.api.features": types.ModuleType("backend.api.features"),
                "backend.api.features.orgs": types.ModuleType(
                    "backend.api.features.orgs"
                ),
                "backend.api.features.orgs.db": orgs_mod,
            },
        )
        return backend

    @pytest.mark.asyncio
    async def test_get_request_context_does_not_bootstrap_the_user(
        self, failing_check, no_personal_org
    ):
        ctx = await get_request_context(_request(), _payload(CLOSED))

        assert ctx.org_id == "org-1"
        no_personal_org.heal.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_get_request_context_still_bootstraps_an_open_account(
        self, no_personal_org
    ):
        payload = _payload(OPEN)

        await get_request_context(_request(), payload)

        no_personal_org.heal.assert_awaited_once_with(OPEN, payload)
