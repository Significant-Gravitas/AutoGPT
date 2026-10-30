"""The account deletion routes (delete, feedback, status and keep), through
the real service and the real database.

Each test gets a throwaway account of its own, removed afterwards with its
deletion request and this process's closed-account answer for it. Only the
edges are mocked, each where it is used: the flag, Stripe (in credit.py), the
providers behind the connected apps, the scheduler, the AutoPilot turn and
chat-link lookups, the MailerLite queue, the email client and the PostHog
client. The app runs on the session event loop through httpx's ASGI transport;
TestClient's own loop would strand the Prisma connection the next test needs.
"""

import logging
from collections.abc import AsyncIterator, Callable
from datetime import datetime, timedelta, timezone
from unittest.mock import ANY, AsyncMock, MagicMock
from uuid import uuid4

import fastapi
import httpx
import pytest
import pytest_asyncio
import stripe
from autogpt_libs.auth import forget_account_state
from autogpt_libs.auth.jwt_utils import get_jwt_payload
from prisma.enums import (
    AccountDeletionSource,
    APIKeyPermission,
    APIKeyStatus,
    BriefingFrequency,
    SubscriptionTier,
)
from prisma.models import AccountDeletionRequest, APIKey, AuthSession, AuthUser, User
from prisma.types import UserUpdateInput
from pydantic import BaseModel, ConfigDict, SecretStr
from pytest_mock import MockerFixture

from backend.api.features.account_deletion.rate_limit import FEEDBACK_MAX_REQUESTS
from backend.api.features.account_deletion.routes import (
    IMPERSONATION_DETAIL,
    TOO_LATE_DETAIL,
    router,
    status_router,
)
from backend.data.auth.api_key import create_api_key
from backend.data.credit import ACCOUNT_DELETION_CANCELLATION_COMMENT
from backend.data.model import APIKeyCredentials, OAuth2Credentials
from backend.data.notifications import NotificationResult
from backend.data.user import get_user_by_id, is_user_banned
from backend.executor.scheduler import GraphExecutionJobInfo
from backend.integrations.credentials_store import ollama_credentials
from backend.util.json import SafeJson
from backend.util.posthog_events import PostHogEvent
from backend.util.test import SpinTestServer

PACKAGE = "backend.api.features.account_deletion"
DELETE = "/api/account/deletion"
STATUS = "/api/account/deletion"
FEEDBACK = "/api/account/deletion/feedback"
RESTORE = "/api/account/deletion/restore"
COMPLETED = PostHogEvent.ACCOUNT_DELETION_COMPLETED.value
STARTED = PostHogEvent.ACCOUNT_DELETION_STARTED.value
ANSWERED = PostHogEvent.ACCOUNT_DELETION_FEEDBACK.value
CLICKED = PostHogEvent.ACCOUNT_DELETION_ALTERNATIVE_CLICKED.value
ANSWER = {"kind": "answer", "reason": "cost"}
CLOSED = {"detail": "This account is closed."}

app = fastapi.FastAPI()
app.include_router(router, prefix="/api")
app.include_router(status_router, prefix="/api")


# ---------------------------------------------------------------------------
# The account, a client signed in as it, and the mocked edges
# ---------------------------------------------------------------------------


class Account(BaseModel):
    user_id: str
    email: str
    stripe_customer_id: str
    api_key_id: str

    def confirm(self, source: str = "settings") -> dict[str, str]:
        """The body the confirmation card sends, typed carelessly."""
        return {"confirm_email": f"  {self.email.upper()} ", "source": source}


class Edges(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    stripe_list: AsyncMock
    stripe_cancel: AsyncMock
    checkout_list: AsyncMock
    revoke: AsyncMock
    cleanup: AsyncMock
    scheduler: MagicMock
    mailerlite: AsyncMock
    mailer: MagicMock


@pytest_asyncio.fixture(loop_scope="session")
async def account(server: SpinTestServer) -> AsyncIterator[Account]:
    """No plan, a Stripe customer, two signed-in sessions and an API key."""
    user_id = str(uuid4())
    email = f"deletion-route-{user_id}@example.com"
    customer_id = f"cus_{user_id.replace('-', '')[:14]}"
    await _settle_the_pool()
    try:
        await User.prisma().create(
            data={"id": user_id, "email": email, "stripeCustomerId": customer_id}
        )
        await AuthUser.prisma().create(
            data={"id": user_id, "name": "Closing account", "email": email}
        )
        for _ in range(2):
            await _add_session(user_id)
        key, _ = await create_api_key(
            name="integration", user_id=user_id, permissions=[APIKeyPermission.IDENTITY]
        )
        yield Account(
            user_id=user_id,
            email=email,
            stripe_customer_id=customer_id,
            api_key_id=key.id,
        )
    finally:
        forget_account_state(user_id)
        await AccountDeletionRequest.prisma().delete_many(where={"userId": user_id})
        await AuthUser.prisma().delete_many(where={"id": user_id})
        await User.prisma().delete_many(where={"id": user_id})
        get_user_by_id.cache_delete(user_id)
        is_user_banned.cache_delete(user_id)


@pytest_asyncio.fixture(loop_scope="session")
async def api(account: Account) -> AsyncIterator[httpx.AsyncClient]:
    """Signed in as the account through the real auth dependencies; only the
    token check is overridden, as everywhere in the API tests."""

    def signed_in(request: fastapi.Request) -> dict[str, str]:
        return {"sub": account.user_id, "role": "user", "email": account.email}

    app.dependency_overrides[get_jwt_payload] = signed_in
    try:
        async with _client() as client:
            yield client
    finally:
        app.dependency_overrides.pop(get_jwt_payload, None)


@pytest.fixture
def flag(mocker: MockerFixture) -> AsyncMock:
    return mocker.patch(
        f"{PACKAGE}.routes.is_feature_enabled",
        new_callable=AsyncMock,
        return_value=True,
    )


@pytest.fixture
def posthog(mocker: MockerFixture) -> MagicMock:
    """The client behind the real ``capture_anonymous``."""
    client = MagicMock()
    mocker.patch("backend.util.posthog_client.get_posthog_client", return_value=client)
    return client


@pytest.fixture
def edges(
    mocker: MockerFixture, account: Account, flag: AsyncMock, posthog: MagicMock
) -> Edges:
    creds_manager = mocker.patch(f"{PACKAGE}.connected_apps.creds_manager")
    creds_manager.store.get_all_creds = AsyncMock(return_value=_credentials())
    _no_autopilot_work(mocker)
    return Edges(
        stripe_list=mocker.patch(
            "backend.data.credit.stripe.Subscription.list_async",
            new_callable=AsyncMock,
            side_effect=_stripe_holding(active="sub_paid"),
        ),
        stripe_cancel=mocker.patch(
            "backend.data.credit.stripe.Subscription.cancel_async",
            new_callable=AsyncMock,
        ),
        checkout_list=mocker.patch(
            "backend.data.subscription_checkout.stripe.checkout.Session.list_async",
            new_callable=AsyncMock,
            return_value=_no_open_checkouts(),
        ),
        revoke=mocker.patch(
            f"{PACKAGE}.connected_apps.delete_and_revoke_credentials",
            new_callable=AsyncMock,
        ),
        cleanup=mocker.patch(
            f"{PACKAGE}.connected_apps.cleanup_user_managed_credentials",
            new_callable=AsyncMock,
        ),
        scheduler=_scheduler(mocker, account.user_id),
        mailerlite=mocker.patch(
            "backend.data.user.queue_audience_change",
            new_callable=AsyncMock,
            return_value=NotificationResult(success=True),
        ),
        mailer=mocker.patch(
            "backend.notifications.auth.get_notification_manager_client"
        ).return_value,
    )


# ---------------------------------------------------------------------------
# POST /api/account/deletion
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_both_routes_need_a_signed_in_user():
    async with _client() as anonymous:
        deleted = await anonymous.post(
            DELETE, json={"confirm_email": "a@example.com", "source": "settings"}
        )
        answered = await anonymous.post(FEEDBACK, json=ANSWER)

    assert (deleted.status_code, answered.status_code) == (401, 401)


@pytest.mark.asyncio(loop_scope="session")
async def test_the_account_is_closed_and_left_for_erasure(
    account: Account, api: httpx.AsyncClient, edges: Edges
):
    await _update(
        account, {"topUpConfig": SafeJson({"amount": 2000, "threshold": 500})}
    )

    response = await api.post(DELETE, json=account.confirm())

    assert response.status_code == 200
    request = await _deletion_request(account)
    assert request is not None
    assert _is_scheduled(response, request)
    assert request.source == AccountDeletionSource.SETTINGS
    assert request.completedAt is None
    assert request.eraseAfter == request.requestedAt + timedelta(days=14)
    # Not banned: signing in is how the owner keeps the account.
    auth_user = await AuthUser.prisma().find_unique_or_raise(
        where={"id": account.user_id}
    )
    assert not auth_user.banned
    assert auth_user.banReason is None
    assert await _session_count(account) == 0
    assert await _api_key_status(account) == APIKeyStatus.REVOKED
    user = await User.prisma().find_unique_or_raise(where={"id": account.user_id})
    assert user.briefingFrequency == BriefingFrequency.OFF
    assert user.maxEmailsPerDay == 0
    assert user.marketingOptOutSource == "account_deleted"
    assert user.topUpConfig == {"amount": 0, "threshold": 0}
    edges.checkout_list.assert_awaited_once_with(
        customer=account.stripe_customer_id, status="open", limit=100
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_an_account_without_a_plan_can_delete_itself(
    account: Account,
    api: httpx.AsyncClient,
    edges: Edges,
    posthog: MagicMock,
    mocker: MockerFixture,
):
    """NO_TIER and paywalled: the route is never behind the payment paywall."""
    await _update(account, {"stripeCustomerId": None})
    paywalled = AsyncMock(return_value=True)
    mocker.patch("backend.copilot.rate_limit.is_user_paywalled", new=paywalled)
    mocker.patch("backend.executor.utils.is_user_paywalled", new=paywalled)

    response = await api.post(DELETE, json=account.confirm())

    assert response.status_code == 200
    edges.stripe_list.assert_not_awaited()
    assert await _deletion_request(account) is not None
    [event] = _captured(posthog, COMPLETED)
    assert event["properties"]["plan_state"] == "none"


@pytest.mark.asyncio(loop_scope="session")
async def test_a_different_email_is_refused(
    account: Account, api: httpx.AsyncClient, edges: Edges, posthog: MagicMock
):
    response = await api.post(
        DELETE, json={"confirm_email": f"x{account.email}", "source": "settings"}
    )

    assert response.status_code == 400
    assert response.json() == {"detail": "That email doesn't match your account."}
    assert await _deletion_request(account) is None
    edges.stripe_list.assert_not_awaited()
    edges.revoke.assert_not_awaited()
    posthog.capture.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_a_repeat_call_answers_200_and_does_nothing_more(
    account: Account, api: httpx.AsyncClient, edges: Edges, posthog: MagicMock
):
    first = await api.post(DELETE, json=account.confirm())
    for mock in (edges.stripe_list, edges.revoke, edges.mailer, posthog):
        mock.reset_mock()

    second = await api.post(DELETE, json=account.confirm())

    assert (first.status_code, second.status_code) == (200, 200)
    assert second.json() == first.json()
    assert second.json()["status"] == "scheduled"
    edges.stripe_list.assert_not_awaited()
    edges.revoke.assert_not_awaited()
    edges.mailer.send_email_or_raise.assert_not_called()
    posthog.capture.assert_not_called()
    assert (
        await AccountDeletionRequest.prisma().count(where={"userId": account.user_id})
        == 1
    )


@pytest.mark.asyncio(loop_scope="session")
async def test_every_plan_is_cancelled_now_without_an_invoice(
    account: Account, api: httpx.AsyncClient, edges: Edges
):
    edges.stripe_list.side_effect = _stripe_holding(
        active="sub_paid", trialing="sub_trial", past_due="sub_past_due"
    )

    response = await api.post(DELETE, json=account.confirm())

    assert response.status_code == 200
    assert {call.kwargs["customer"] for call in edges.stripe_list.await_args_list} == {
        account.stripe_customer_id
    }
    cancelled = {
        call.args[0]: call.kwargs for call in edges.stripe_cancel.await_args_list
    }
    assert set(cancelled) == {"sub_paid", "sub_trial", "sub_past_due"}
    for params in cancelled.values():
        assert params["cancellation_details"] == {
            "comment": ACCOUNT_DELETION_CANCELLATION_COMMENT
        }
        assert params.get("invoice_now") is not True
        assert params.get("prorate") is not True
    assert cancelled["sub_trial"]["invoice_now"] is False
    assert cancelled["sub_trial"]["prorate"] is False


@pytest.mark.asyncio(loop_scope="session")
async def test_a_stripe_error_deletes_nothing(
    account: Account, api: httpx.AsyncClient, edges: Edges, posthog: MagicMock
):
    edges.stripe_list.side_effect = stripe.APIConnectionError("Stripe is unreachable")

    response = await api.post(DELETE, json=account.confirm())

    assert response.status_code == 502
    assert response.json() == {
        "detail": "We couldn't cancel your plan. Nothing was deleted. Try again."
    }
    assert await _deletion_request(account) is None
    auth_user = await AuthUser.prisma().find_unique_or_raise(
        where={"id": account.user_id}
    )
    assert not auth_user.banned
    assert await _session_count(account) == 2
    assert await _api_key_status(account) == APIKeyStatus.ACTIVE
    edges.revoke.assert_not_awaited()
    edges.cleanup.assert_not_awaited()
    edges.scheduler.get_execution_schedules.assert_not_awaited()
    edges.mailerlite.assert_not_awaited()
    edges.mailer.send_email_or_raise.assert_not_called()
    posthog.capture.assert_not_called()
    still_open = await api.post(FEEDBACK, json=ANSWER)
    assert still_open.status_code == 204


@pytest.mark.asyncio(loop_scope="session")
async def test_each_connected_app_is_revoked_even_when_one_fails(
    account: Account, api: httpx.AsyncClient, edges: Edges
):
    edges.revoke.side_effect = [RuntimeError("GitHub is down"), None]

    response = await api.post(DELETE, json=account.confirm())

    assert response.status_code == 200
    revoked = [
        (call.args[0], call.args[1].id, call.kwargs)
        for call in edges.revoke.await_args_list
    ]
    assert revoked == [
        (account.user_id, "cred-github", {"force": True, "skip_failed_prunes": True}),
        (account.user_id, "cred-openai", {"force": True, "skip_failed_prunes": True}),
    ]


@pytest.mark.asyncio(loop_scope="session")
async def test_managed_credentials_are_cleaned_up(
    account: Account, api: httpx.AsyncClient, edges: Edges
):
    response = await api.post(DELETE, json=account.confirm())

    assert response.status_code == 200
    edges.cleanup.assert_awaited_once_with(account.user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_schedules_are_removed(
    account: Account, api: httpx.AsyncClient, edges: Edges
):
    response = await api.post(DELETE, json=account.confirm())

    assert response.status_code == 200
    scheduler = edges.scheduler
    scheduler.get_execution_schedules.assert_awaited_once_with(
        user_id=account.user_id, include_paused=True
    )
    removed = sorted(
        call.kwargs["schedule_id"] for call in scheduler.delete_schedule.await_args_list
    )
    assert removed == ["job-1", "job-2"]
    for call in scheduler.delete_schedule.await_args_list:
        assert call.kwargs["user_id"] == account.user_id
    for job in (
        scheduler.remove_morning_briefing_schedule,
        scheduler.delete_nightly_batch_schedule,
        scheduler.delete_community_rebuild_schedule,
    ):
        job.assert_awaited_once_with(user_id=account.user_id)


@pytest.mark.asyncio(loop_scope="session")
async def test_the_old_token_is_refused_afterwards(
    account: Account, api: httpx.AsyncClient, edges: Edges
):
    await api.post(DELETE, json=account.confirm())

    right_after = await api.post(FEEDBACK, json=ANSWER)
    # Another worker has no cached answer: it asks the database.
    forget_account_state(account.user_id)
    on_another_worker = await api.post(FEEDBACK, json=ANSWER)

    assert (right_after.status_code, right_after.json()) == (401, CLOSED)
    assert (on_another_worker.status_code, on_another_worker.json()) == (401, CLOSED)


@pytest.mark.asyncio(loop_scope="session")
async def test_a_deleted_account_is_answered_as_closed_not_banned(
    account: Account, api: httpx.AsyncClient, edges: Edges, real_ban_lookup: None
):
    """Closing the account bans nothing. The API answers it as closed, and the
    deletion route still answers its repeat request."""
    first = await api.post(DELETE, json=account.confirm())
    # Another worker has no cached answers: it reads the ban and the closure.
    is_user_banned.cache_delete(account.user_id)
    forget_account_state(account.user_id)
    repeat = await api.post(DELETE, json=account.confirm())
    elsewhere = await api.post(FEEDBACK, json=ANSWER)

    assert (first.status_code, repeat.status_code) == (200, 200)
    assert (elsewhere.status_code, elsewhere.json()) == (401, CLOSED)
    assert await is_user_banned(account.user_id) is False


@pytest.mark.asyncio(loop_scope="session")
async def test_an_account_banned_for_another_reason_can_delete_itself(
    account: Account, api: httpx.AsyncClient, edges: Edges, real_ban_lookup: None
):
    """Deleting the account is the person's right whatever the ban, and the
    ban keeps its reason, so they can never sign in to keep the account."""
    await AuthUser.prisma().update(
        where={"id": account.user_id}, data={"banned": True, "banReason": "spam"}
    )

    response = await api.post(DELETE, json=account.confirm())
    elsewhere = await api.post(FEEDBACK, json=ANSWER)

    assert response.status_code == 200
    assert response.json()["status"] == "scheduled"
    assert await _deletion_request(account) is not None
    edges.stripe_list.assert_awaited()
    auth_user = await AuthUser.prisma().find_unique(where={"id": account.user_id})
    assert auth_user is not None
    assert (auth_user.banned, auth_user.banReason) == (True, "spam")
    assert elsewhere.status_code == 403
    assert elsewhere.json()["detail"]["code"] == "account_banned"


@pytest.mark.asyncio(loop_scope="session")
async def test_the_confirmation_email_replies_to_the_shared_inbox(
    account: Account, api: httpx.AsyncClient, edges: Edges
):
    response = await api.post(DELETE, json=account.confirm())

    assert response.status_code == 200
    send = edges.mailer.send_email_or_raise
    send.assert_called_once()
    to, subject, html, text = send.call_args.args
    assert to == account.email
    request = await _deletion_request(account)
    assert request is not None
    last_day = request.eraseAfter - timedelta(days=1)
    erase_day = f"{last_day.day} {last_day.strftime('%b %Y')}"
    assert subject == "Your AutoGPT account is closed"
    assert "Your account is closed" in html
    assert (
        "We've closed your AutoGPT account, signed you out and disconnected your "
        f"apps. Changed your mind? Sign in by {erase_day} and choose Keep my "
        "account. After that, everything in your account is erased. If you have "
        "a minute, reply and tell us what we could have done better."
    ) in text
    assert send.call_args.kwargs == {"reply_to": "contact@agpt.co"}


@pytest.mark.asyncio(loop_scope="session")
async def test_one_anonymous_count_with_the_plan_held_before_the_cancel(
    account: Account, api: httpx.AsyncClient, edges: Edges, posthog: MagicMock
):
    await _update(account, {"subscriptionTier": SubscriptionTier.PRO})

    async def cancel_and_sync_the_tier(sub_id: str, **params) -> None:
        # The webhook the cancel sets off can land while the request runs.
        await _update(account, {"subscriptionTier": SubscriptionTier.NO_TIER})

    edges.stripe_cancel.side_effect = cancel_and_sync_the_tier

    response = await api.post(DELETE, json=account.confirm(source="paywall"))

    assert response.status_code == 200
    [event] = _captured(posthog, COMPLETED)
    assert event["properties"] == {
        "plan_state": "pro",
        "surface": "paywall",
        "$process_person_profile": False,
        "environment": ANY,
        "source": "platform",
    }
    _assert_anonymous(event, account)


@pytest.mark.asyncio(loop_scope="session")
async def test_neither_email_is_logged(
    account: Account,
    api: httpx.AsyncClient,
    edges: Edges,
    caplog: pytest.LogCaptureFixture,
):
    typo = f"typo-{account.email}"
    edges.mailer.send_email_or_raise.side_effect = RuntimeError(
        f"Postmark refused {account.email}"
    )

    with caplog.at_level(logging.DEBUG):
        refused = await api.post(
            DELETE, json={"confirm_email": typo, "source": "settings"}
        )
        deleted = await api.post(DELETE, json=account.confirm())

    assert (refused.status_code, deleted.status_code) == (400, 200)
    logged = caplog.text.lower()
    assert account.email not in logged
    assert typo not in logged


@pytest.mark.asyncio(loop_scope="session")
async def test_with_the_flag_off_both_routes_are_hidden(
    account: Account,
    api: httpx.AsyncClient,
    edges: Edges,
    flag: AsyncMock,
    posthog: MagicMock,
):
    flag.return_value = False

    deleted = await api.post(DELETE, json=account.confirm())
    answered = await api.post(FEEDBACK, json=ANSWER)

    assert (deleted.status_code, answered.status_code) == (404, 404)
    assert await _deletion_request(account) is None
    edges.stripe_list.assert_not_awaited()
    posthog.capture.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_the_flag_is_off_when_no_flag_service_answers(
    account: Account,
    api: httpx.AsyncClient,
    posthog: MagicMock,
    monkeypatch: pytest.MonkeyPatch,
    mocker: MockerFixture,
):
    """The real gate, with no local override and no flag vendor (self-hosted,
    CI): closing an account is permanent, so the gate must fail closed."""
    for name in (
        "FORCE_FLAG_SELF_SERVE_ACCOUNT_DELETION",
        "NEXT_PUBLIC_FORCE_FLAG_SELF_SERVE_ACCOUNT_DELETION",
        "FORCE_ALL_FLAGS",
        "NEXT_PUBLIC_FORCE_ALL_FLAGS",
    ):
        monkeypatch.delenv(name, raising=False)
    mocker.patch(
        "backend.util.feature_flag._evaluate_flag_value",
        new_callable=AsyncMock,
        side_effect=lambda key, user_id, default, **_: (default, False),
    )
    stripe_list = mocker.patch(
        "backend.data.credit.stripe.Subscription.list_async", new_callable=AsyncMock
    )

    deleted = await api.post(DELETE, json=account.confirm())
    answered = await api.post(FEEDBACK, json=ANSWER)

    assert (deleted.status_code, answered.status_code) == (404, 404)
    assert await _deletion_request(account) is None
    stripe_list.assert_not_awaited()
    posthog.capture.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_an_admin_acting_as_the_account_can_neither_delete_nor_answer(
    account: Account, flag: AsyncMock, posthog: MagicMock, mocker: MockerFixture
):
    """Impersonation is for support; closing an account is its owner's own act."""
    delete = mocker.patch(f"{PACKAGE}.routes.delete_account", new_callable=AsyncMock)
    answer = mocker.patch(f"{PACKAGE}.routes.record_feedback", new_callable=AsyncMock)
    # No email claim, so the admin's own User row is never provisioned here.
    admin = {"sub": f"admin-{uuid4()}", "role": "admin"}
    app.dependency_overrides[get_jwt_payload] = lambda: admin
    acting_as = {"X-Act-As-User-Id": account.user_id}
    try:
        async with _client() as client:
            deleted = await client.post(
                DELETE, json=account.confirm(), headers=acting_as
            )
            answered = await client.post(FEEDBACK, json=ANSWER, headers=acting_as)
    finally:
        app.dependency_overrides.pop(get_jwt_payload, None)

    assert (deleted.status_code, answered.status_code) == (403, 403)
    assert deleted.json() == {
        "detail": "Account deletion isn't available while impersonating a user."
    }
    delete.assert_not_awaited()
    answer.assert_not_awaited()
    assert await _deletion_request(account) is None
    posthog.capture.assert_not_called()


def test_both_routes_declare_the_impersonation_refusal():
    paths = app.openapi()["paths"]

    for path in (DELETE, FEEDBACK):
        assert paths[path]["post"]["responses"]["403"] == {
            "description": IMPERSONATION_DETAIL
        }


# ---------------------------------------------------------------------------
# GET /api/account/deletion and POST /api/account/deletion/restore
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_an_open_account_has_nothing_scheduled(
    account: Account, api: httpx.AsyncClient, flag: AsyncMock
):
    response = await api.get(STATUS)

    assert response.status_code == 200
    assert response.json() == {"status": "open", "erase_after": None}


@pytest.mark.asyncio(loop_scope="session")
async def test_signing_in_during_the_grace_period_shows_the_erasure_date(
    account: Account, api: httpx.AsyncClient, edges: Edges
):
    await api.post(DELETE, json=account.confirm())
    # Another worker: it asks the database.
    forget_account_state(account.user_id)

    response = await api.get(STATUS)

    request = await _deletion_request(account)
    assert request is not None
    assert response.status_code == 200
    assert _is_scheduled(response, request)


@pytest.mark.asyncio(loop_scope="session")
async def test_keeping_the_account_opens_it_again(
    account: Account, api: httpx.AsyncClient, edges: Edges
):
    """The request is cancelled and the account works again at once; what the
    close stopped stays stopped (the API key stays revoked)."""
    await api.post(DELETE, json=account.confirm())

    kept = await api.post(RESTORE)
    answered = await api.post(FEEDBACK, json=ANSWER)
    status = await api.get(STATUS)

    assert (kept.status_code, kept.json()) == (
        200,
        {"status": "open", "erase_after": None},
    )
    assert answered.status_code == 204
    assert status.json()["status"] == "open"
    assert await _deletion_request(account) is None
    assert await _api_key_status(account) == APIKeyStatus.REVOKED
    user = await User.prisma().find_unique_or_raise(where={"id": account.user_id})
    assert user.maxEmailsPerDay == 3
    # And it can be deleted again, with a new grace period.
    again = await api.post(DELETE, json=account.confirm())
    assert again.status_code == 200
    assert await _deletion_request(account) is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_the_account_cannot_be_kept_after_the_grace_period(
    account: Account, api: httpx.AsyncClient, edges: Edges
):
    await api.post(DELETE, json=account.confirm())
    await AccountDeletionRequest.prisma().update(
        where={"userId": account.user_id},
        data={"eraseAfter": datetime.now(timezone.utc) - timedelta(minutes=1)},
    )

    response = await api.post(RESTORE)

    assert (response.status_code, response.json()) == (
        409,
        {"detail": TOO_LATE_DETAIL},
    )
    assert await _deletion_request(account) is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_the_account_cannot_be_kept_once_its_erasure_started(
    account: Account, api: httpx.AsyncClient, edges: Edges
):
    await api.post(DELETE, json=account.confirm())
    await AccountDeletionRequest.prisma().update(
        where={"userId": account.user_id},
        data={"startedAt": datetime.now(timezone.utc)},
    )

    response = await api.post(RESTORE)

    assert (response.status_code, response.json()) == (
        409,
        {"detail": TOO_LATE_DETAIL},
    )
    assert await _deletion_request(account) is not None


@pytest.mark.asyncio(loop_scope="session")
async def test_the_account_can_be_kept_with_the_flag_off(
    account: Account, api: httpx.AsyncClient, edges: Edges, flag: AsyncMock
):
    """Turning the flag off must not strand someone who closed their account
    while it was on."""
    await api.post(DELETE, json=account.confirm())
    flag.return_value = False

    status = await api.get(STATUS)
    kept = await api.post(RESTORE)

    assert status.json()["status"] == "scheduled"
    assert kept.status_code == 200
    assert await _deletion_request(account) is None


@pytest.mark.asyncio(loop_scope="session")
async def test_an_admin_acting_as_the_account_cannot_keep_it(
    account: Account, edges: Edges, mocker: MockerFixture
):
    restore = mocker.patch(f"{PACKAGE}.routes.restore_account", new_callable=AsyncMock)
    admin = {"sub": f"admin-{uuid4()}", "role": "admin"}
    app.dependency_overrides[get_jwt_payload] = lambda: admin
    try:
        async with _client() as client:
            response = await client.post(
                RESTORE, headers={"X-Act-As-User-Id": account.user_id}
            )
    finally:
        app.dependency_overrides.pop(get_jwt_payload, None)

    assert (response.status_code, response.json()) == (
        403,
        {"detail": IMPERSONATION_DETAIL},
    )
    restore.assert_not_awaited()


# ---------------------------------------------------------------------------
# POST /api/account/deletion/feedback
# ---------------------------------------------------------------------------


@pytest.mark.asyncio(loop_scope="session")
async def test_an_answer_is_counted_without_the_person(
    account: Account, api: httpx.AsyncClient, flag: AsyncMock, posthog: MagicMock
):
    await _update(
        account,
        {"subscriptionTier": SubscriptionTier.MAX, "createdAt": _days_ago(45)},
    )

    response = await api.post(
        FEEDBACK, json={"kind": "answer", "reason": "cost", "text": "  Too pricey "}
    )

    assert response.status_code == 204
    [event] = _captured(posthog, ANSWERED)
    assert event["properties"] == {
        "reason": "cost",
        "text": "Too pricey",
        "plan_state": "max",
        "tenure_bucket": "30-90d",
        "$process_person_profile": False,
        "environment": ANY,
        "source": "platform",
    }
    _assert_anonymous(event, account)


@pytest.mark.asyncio(loop_scope="session")
async def test_an_open_is_counted_with_its_surface_without_the_person(
    account: Account, api: httpx.AsyncClient, flag: AsyncMock, posthog: MagicMock
):
    await _update(
        account,
        {"subscriptionTier": SubscriptionTier.TRIAL, "createdAt": _days_ago(3)},
    )

    response = await api.post(FEEDBACK, json={"kind": "opened", "surface": "paywall"})

    assert response.status_code == 204
    [event] = _captured(posthog, STARTED)
    assert event["properties"] == {
        "surface": "paywall",
        "plan_state": "trial",
        "tenure_bucket": "<7d",
        "$process_person_profile": False,
        "environment": ANY,
        "source": "platform",
    }
    _assert_anonymous(event, account)


@pytest.mark.asyncio(loop_scope="session")
async def test_the_text_is_trimmed_and_capped(
    account: Account, api: httpx.AsyncClient, flag: AsyncMock, posthog: MagicMock
):
    response = await api.post(
        FEEDBACK,
        json={"kind": "answer", "reason": "other", "text": f"  {'a' * 1200}  "},
    )

    assert response.status_code == 204
    [event] = _captured(posthog, ANSWERED)
    assert event["properties"]["text"] == "a" * 1000


@pytest.mark.asyncio(loop_scope="session")
async def test_the_plan_and_tenure_come_from_the_server(
    account: Account, api: httpx.AsyncClient, flag: AsyncMock, posthog: MagicMock
):
    await _update(
        account,
        {"subscriptionTier": SubscriptionTier.BUSINESS, "createdAt": _days_ago(400)},
    )

    response = await api.post(
        FEEDBACK,
        json={
            "kind": "alternative_clicked",
            "reason": "cost",
            "alternative": "billing",
            "plan_state": "none",
            "tenure_bucket": "<7d",
            "user_id": account.user_id,
            "email": account.email,
        },
    )

    assert response.status_code == 204
    [event] = _captured(posthog, CLICKED)
    assert event["properties"] == {
        "reason": "cost",
        "alternative": "billing",
        "plan_state": "managed",
        "tenure_bucket": "90d+",
        "$process_person_profile": False,
        "environment": ANY,
        "source": "platform",
    }
    _assert_anonymous(event, account)


@pytest.mark.asyncio(loop_scope="session")
@pytest.mark.parametrize(
    "body",
    [
        {"kind": "alternative_clicked", "reason": "cost"},
        {"kind": "answer", "reason": "cost", "alternative": "billing"},
        {"kind": "answer", "reason": "bored"},
        {"kind": "goodbye"},
        {"kind": "opened"},
        {"kind": "answer", "reason": "cost", "surface": "settings"},
    ],
    ids=[
        "click_without_alternative",
        "answer_with_alternative",
        "reason",
        "kind",
        "open_without_surface",
        "answer_with_surface",
    ],
)
async def test_an_invalid_body_is_refused(
    account: Account,
    api: httpx.AsyncClient,
    flag: AsyncMock,
    posthog: MagicMock,
    body: dict,
):
    response = await api.post(FEEDBACK, json=body)

    assert response.status_code == 422
    posthog.capture.assert_not_called()


@pytest.mark.asyncio(loop_scope="session")
async def test_answers_past_the_limit_get_a_429(
    account: Account, api: httpx.AsyncClient, flag: AsyncMock, posthog: MagicMock
):
    responses = [
        await api.post(FEEDBACK, json=ANSWER) for _ in range(FEEDBACK_MAX_REQUESTS + 1)
    ]

    statuses = [response.status_code for response in responses]
    assert statuses == [204] * FEEDBACK_MAX_REQUESTS + [429]
    assert responses[-1].json() == {"detail": "Too many answers. Try again later."}
    assert len(_captured(posthog, ANSWERED)) == FEEDBACK_MAX_REQUESTS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app), base_url="http://test"
    )


def _captured(posthog: MagicMock, event: str) -> list[dict]:
    """The keyword arguments of each PostHog ``capture`` call for *event*."""
    return [
        call.kwargs
        for call in posthog.capture.call_args_list
        if call.kwargs.get("event") == event
    ]


def _assert_anonymous(event: dict, account: Account) -> None:
    assert event["distinct_id"] != account.user_id
    sent = str(event).lower()
    assert account.user_id not in sent
    assert account.email not in sent


def _credentials() -> list:
    """Two the user connected, plus a managed, an SDK-default and a system one
    that are not the user's own to revoke."""
    return [
        OAuth2Credentials(
            id="cred-github",
            provider="github",
            title="GitHub",
            access_token=SecretStr("ghp_access_secret"),
            scopes=["repo"],
        ),
        APIKeyCredentials(
            id="cred-managed",
            provider="agent_mail",
            api_key=SecretStr("sk-managed"),
            is_managed=True,
        ),
        APIKeyCredentials(
            id="cred-openai", provider="openai", api_key=SecretStr("sk-user")
        ),
        APIKeyCredentials(
            id="exa-default", provider="exa", api_key=SecretStr("sk-platform")
        ),
        ollama_credentials,
    ]


def _stripe_holding(**subscription_by_status: str) -> Callable[..., MagicMock]:
    """Stripe's list call, holding one subscription in each named status and,
    like Stripe, listing only those in the status a call asks for."""

    def list_by_status(*_args, **kwargs) -> MagicMock:
        status = kwargs["status"]
        sub_id = subscription_by_status.get(status)
        page = MagicMock()
        page.has_more = False
        page.data = (
            [
                stripe.Subscription.construct_from(
                    {"id": sub_id, "status": status, "schedule": None}, "sk_test"
                )
            ]
            if sub_id
            else []
        )
        return page

    return list_by_status


def _no_open_checkouts() -> MagicMock:
    page = MagicMock()
    page.has_more = False
    page.data = []
    return page


def _scheduler(mocker: MockerFixture, user_id: str) -> MagicMock:
    client = MagicMock()
    client.get_execution_schedules = AsyncMock(
        return_value=[_schedule("job-1", user_id), _schedule("job-2", user_id)]
    )
    client.delete_schedule = AsyncMock()
    client.remove_morning_briefing_schedule = AsyncMock()
    client.delete_nightly_batch_schedule = AsyncMock()
    client.delete_community_rebuild_schedule = AsyncMock()
    mocker.patch(f"{PACKAGE}.scheduled_work.get_scheduler_client", return_value=client)
    return client


def _schedule(job_id: str, user_id: str) -> GraphExecutionJobInfo:
    return GraphExecutionJobInfo(
        id=job_id,
        name="Daily digest",
        next_run_time="2026-10-13T09:00:00+00:00",
        schedule_id=job_id,
        user_id=user_id,
        graph_id="graph-1",
        graph_version=1,
        cron="0 9 * * *",
        input_data={},
    )


def _no_autopilot_work(mocker: MockerFixture) -> None:
    """No AutoPilot turn running or queued, and no chat platform linked."""
    module = f"{PACKAGE}.autopilot_work"
    mocker.patch(
        f"{module}.get_running_session_ids", new_callable=AsyncMock, return_value=set()
    )
    mocker.patch(
        f"{module}.list_queued_sessions", new_callable=AsyncMock, return_value=[]
    )
    links = mocker.patch(f"{module}.platform_linking_db").return_value
    links.list_server_links = AsyncMock(return_value=[])
    links.list_user_links = AsyncMock(return_value=[])


async def _update(account: Account, data: UserUpdateInput) -> None:
    await User.prisma().update(where={"id": account.user_id}, data=data)
    get_user_by_id.cache_delete(account.user_id)


def _is_scheduled(response: httpx.Response, request: AccountDeletionRequest) -> bool:
    """The answer names the request's erasure date."""
    body = response.json()
    return body["status"] == "scheduled" and (
        datetime.fromisoformat(body["erase_after"]) == request.eraseAfter
    )


async def _deletion_request(account: Account) -> AccountDeletionRequest | None:
    return await AccountDeletionRequest.prisma().find_unique(
        where={"userId": account.user_id}
    )


async def _session_count(account: Account) -> int:
    return await AuthSession.prisma().count(where={"userId": account.user_id})


async def _api_key_status(account: Account) -> APIKeyStatus:
    key = await APIKey.prisma().find_unique_or_raise(where={"id": account.api_key_id})
    return key.status


def _days_ago(days: int) -> datetime:
    return datetime.now(timezone.utc) - timedelta(days=days)


async def _add_session(user_id: str) -> None:
    await AuthSession.prisma().create(
        data={
            "id": str(uuid4()),
            "token": str(uuid4()),
            "userId": user_id,
            "expiresAt": datetime.now(timezone.utc) + timedelta(days=1),
        }
    )


async def _settle_the_pool() -> None:
    """An earlier test can leave the Prisma pool bound to a closed event loop;
    only the first query on the session loop after that fails."""
    try:
        await User.prisma().count(where={"id": "pool-check"})
    except RuntimeError as e:
        if "Event loop is closed" not in str(e):
            raise
