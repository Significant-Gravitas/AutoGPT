"""The four copilot checkout tools, end to end over the in-process broker."""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import SecretStr

from backend.copilot.model import ChatSession
from backend.copilot.tools import TOOL_REGISTRY
from backend.copilot.tools import browser_checkout as request_tools
from backend.copilot.tools import browser_checkout_status as status_tools
from backend.copilot.tools import browser_checkout_support as support
from backend.copilot.tools.browser_checkout_support import CheckoutResponse
from backend.copilot.tools.models import ErrorResponse
from backend.data.model import OAuth2Credentials
from backend.util.link_checkout import approval, broker_link
from backend.util.link_checkout.refusals import (
    DUPLICATE_REQUEST,
    LINK_ACCOUNT_NOT_CHOSEN,
    LINK_NOT_CONNECTED,
)
from backend.util.settings import BehaveAs


@asynccontextmanager
async def fake_link_credentials(user_id: str, credentials_id: str):
    yield MagicMock(access_token=SecretStr("synthetic-token"))


@pytest.fixture
def tools(local_broker, fake_redis, monkeypatch):
    monkeypatch.setattr(support, "available", lambda: True)
    for module in (request_tools, status_tools):
        monkeypatch.setattr(module, "link_credentials", fake_link_credentials)
    monkeypatch.setattr(request_tools, "validate_url_host", AsyncMock())
    monkeypatch.setattr(
        request_tools, "in_app_approval_allowed", AsyncMock(return_value=False)
    )
    monkeypatch.setattr("backend.copilot.tools.base._record_activity", AsyncMock())
    return local_broker


def session():
    # Not interactive, so the AutoPilot approval gate stays out of the way; the
    # checkout tools carry their own approval (see gate/policy.py).
    return MagicMock(
        spec=ChatSession,
        session_id="chat",
        user_id="owner",
        metadata=MagicMock(origin="scheduled"),
    )


async def run(name: str, user_id: str = "owner", **arguments):
    return await TOOL_REGISTRY[name].execute(user_id, session(), "call", **arguments)


def parsed(result) -> CheckoutResponse:
    return CheckoutResponse.model_validate_json(result.output)


@pytest.mark.asyncio
async def test_link_approval_then_exactly_one_private_attempt(tools, plan):
    created = parsed(await run("browser_request_link_payment", **plan.model_dump()))
    assert created.status == "pending_approval"
    assert created.approval_mode == "link"
    assert created.approval_url.startswith("https://app.link.com/")

    completed = await run(
        "browser_complete_link_payment", checkout_id=created.checkout_id
    )
    assert parsed(completed).status == "submitted"
    assert tools.calls == ["create", "status", "pay"]

    again = await run("browser_complete_link_payment", checkout_id=created.checkout_id)
    assert parsed(again).attempted is True
    assert tools.calls == ["create", "status", "pay", "status"]
    for output in (completed.output, again.output):
        assert "4242424242424242" not in output
        assert "synthetic-token" not in output


@pytest.mark.asyncio
async def test_in_chat_approval_pays_only_after_the_customer_clicks(
    tools, plan, monkeypatch
):
    monkeypatch.setattr(
        request_tools, "in_app_approval_allowed", AsyncMock(return_value=True)
    )
    created = parsed(await run("browser_request_link_payment", **plan.model_dump()))
    assert created.status == "awaiting_approval"
    assert created.approval_mode == "in_app"
    assert created.approval_url == ""
    assert (await approval.read_approval(created.checkout_id)).state == "awaiting"

    early = await run("browser_complete_link_payment", checkout_id=created.checkout_id)
    assert parsed(early).status == "awaiting_approval"
    assert tools.calls == []

    await approval.decide(
        created.checkout_id, "owner", "chat", approve=True, user_agent="Mozilla/5.0"
    )
    paid = await run("browser_complete_link_payment", checkout_id=created.checkout_id)

    assert parsed(paid).status == "submitted"
    assert tools.calls == ["create_delegated", "pay"]


@pytest.mark.asyncio
async def test_a_declined_purchase_is_never_paid(tools, plan, monkeypatch):
    monkeypatch.setattr(
        request_tools, "in_app_approval_allowed", AsyncMock(return_value=True)
    )
    created = parsed(await run("browser_request_link_payment", **plan.model_dump()))
    await approval.decide(
        created.checkout_id, "owner", "chat", approve=False, user_agent=None
    )

    result = parsed(
        await run("browser_complete_link_payment", checkout_id=created.checkout_id)
    )

    assert result.status == "declined"
    assert result.approval_state == "declined"
    assert tools.calls == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change",
    [{"amount": 1}, {"merchant_url": "https://elsewhere.example/checkout"}],
)
async def test_an_approval_for_a_different_purchase_does_not_count(
    tools, plan, monkeypatch, change
):
    monkeypatch.setattr(
        request_tools, "in_app_approval_allowed", AsyncMock(return_value=True)
    )
    created = parsed(await run("browser_request_link_payment", **plan.model_dump()))
    await approval.decide(
        created.checkout_id, "owner", "chat", approve=True, user_agent=None
    )
    view = await approval.read_approval(created.checkout_id)
    tampered = view.model_copy(
        update={"pending": view.pending.model_copy(update=change)}
    )
    monkeypatch.setattr(
        request_tools, "read_approval", AsyncMock(return_value=tampered)
    )

    result = parsed(
        await run("browser_complete_link_payment", checkout_id=created.checkout_id)
    )

    assert result.status == "expired"
    assert tools.calls == []


@pytest.mark.asyncio
async def test_status_is_read_only_and_reset_needs_a_final_status(tools, plan):
    created = parsed(await run("browser_request_link_payment", **plan.model_dump()))
    await run("browser_complete_link_payment", checkout_id=created.checkout_id)

    tools.status = "submitted"
    status = parsed(
        await run("browser_link_payment_status", checkout_id=created.checkout_id)
    )
    assert status.paid is False
    refused = await run("browser_reset_after_payment", checkout_id=created.checkout_id)
    assert '"error":"private_checkout_unavailable"' in refused.output

    tools.status = "succeeded"
    status = parsed(
        await run("browser_link_payment_status", checkout_id=created.checkout_id)
    )
    assert status.paid is True
    reset = parsed(
        await run("browser_reset_after_payment", checkout_id=created.checkout_id)
    )
    assert reset.status == "browser_reset"
    assert tools.calls.count("pay") == 1


@pytest.mark.asyncio
async def test_wrong_user_never_reaches_the_broker(tools, plan):
    await run("browser_request_link_payment", user_id="mallory", **plan.model_dump())
    assert tools.calls == []


@pytest.mark.asyncio
async def test_worker_failure_text_never_reaches_the_tool_output(
    tools, plan, monkeypatch
):
    created = parsed(await run("browser_request_link_payment", **plan.model_dump()))

    async def exploding_worker(job):
        raise RuntimeError("canary-secret-4242424242424242")

    monkeypatch.setattr(broker_link, "run_worker", exploding_worker)
    result = await run("browser_complete_link_payment", checkout_id=created.checkout_id)
    assert "canary-secret" not in result.output
    assert "4242424242424242" not in result.output


@pytest.mark.asyncio
async def test_concurrent_completions_attempt_the_checkout_once(tools, plan):
    created = parsed(await run("browser_request_link_payment", **plan.model_dump()))
    results = await asyncio.gather(
        run("browser_complete_link_payment", checkout_id=created.checkout_id),
        run("browser_complete_link_payment", checkout_id=created.checkout_id),
    )
    assert tools.calls.count("pay") == 1
    statuses = sorted(parsed(result).status for result in results)
    # The loser waits for the lock, finds the attempt recorded, and only
    # reconciles with Link.
    assert statuses == ["approved", "submitted"]
    assert all(parsed(result).attempted for result in results)


def test_hosted_checkout_needs_a_broker_the_registered_client_and_opt_in(
    monkeypatch,
):
    monkeypatch.setattr(
        support, "_settings", MagicMock(config=MagicMock(behave_as=BehaveAs.CLOUD))
    )
    monkeypatch.setattr(support.engine, "enabled", lambda: True)
    monkeypatch.setattr(support.engine, "remote", lambda: False)
    monkeypatch.setattr(support, "STRIPE_LINK_HOSTED_OAUTH_IS_CONFIGURED", True)
    monkeypatch.setenv("COPILOT_LINK_HOSTED_CHECKOUT", "true")
    assert not support.available()

    monkeypatch.setattr(support.engine, "remote", lambda: True)
    assert support.available()

    monkeypatch.setattr(support, "STRIPE_LINK_HOSTED_OAUTH_IS_CONFIGURED", False)
    assert not support.available()

    monkeypatch.setattr(support, "STRIPE_LINK_HOSTED_OAUTH_IS_CONFIGURED", True)
    monkeypatch.setenv("COPILOT_LINK_HOSTED_CHECKOUT", "false")
    assert not support.available()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "metadata,scopes",
    [
        ({}, ["payment_methods.agentic"]),
        (
            {"link_oauth_flow": "authorization_code", "link_client_id": "other"},
            ["payment_methods.agentic"],
        ),
        (
            {"link_oauth_flow": "authorization_code", "link_client_id": "ours"},
            ["userinfo:read"],
        ),
    ],
)
async def test_hosted_payment_refuses_device_other_client_and_unscoped_tokens(
    monkeypatch, metadata, scopes
):
    monkeypatch.setattr(
        support,
        "_settings",
        MagicMock(
            config=MagicMock(behave_as=BehaveAs.CLOUD),
            secrets=MagicMock(stripe_link_client_id="ours"),
        ),
    )
    lease = MagicMock(
        credentials=OAuth2Credentials(
            provider="stripe_link",
            access_token="synthetic",
            scopes=scopes,
            metadata=metadata,
        ),
        validate=AsyncMock(),
        release=AsyncMock(),
    )
    monkeypatch.setattr(
        support._credentials, "acquire_lease", AsyncMock(return_value=lease)
    )
    with pytest.raises(ValueError):
        async with support.link_credentials("owner", "wallet"):
            pytest.fail("Token accepted")
    lease.release.assert_awaited_once()


@pytest.mark.asyncio
async def test_the_chat_approval_records_where_and_what_the_customer_approves(
    tools, plan, monkeypatch
):
    monkeypatch.setattr(
        request_tools, "in_app_approval_allowed", AsyncMock(return_value=True)
    )
    created = parsed(await run("browser_request_link_payment", **plan.model_dump()))

    view = await approval.read_approval(created.checkout_id)

    assert view is not None
    assert view.pending.merchant_url == plan.merchant_url() == created.merchant_url
    assert view.pending.context == plan.context


@pytest.mark.asyncio
async def test_a_refusal_tells_the_agent_why(tools, plan):
    tools.create_error = "link_duplicate"

    result = await run("browser_request_link_payment", **plan.model_dump())

    assert DUPLICATE_REQUEST in result.output


def link_account(credential_id: str, scopes=("payment_methods.agentic",)):
    return OAuth2Credentials(
        id=credential_id,
        provider="stripe_link",
        access_token=SecretStr("synthetic"),
        scopes=list(scopes),
    )


@pytest.fixture
def link_accounts(tools, monkeypatch):
    """The user's stored Link connections, the account picked in the chat, and
    the credential each checkout leased."""
    state = SimpleNamespace(stored=[], picked={}, leased=[])

    async def by_provider(user_id, provider):
        assert (user_id, provider) == ("owner", "stripe_link")
        return state.stored

    async def picks(session_id):
        assert session_id == "chat"
        return state.picked

    @asynccontextmanager
    async def leased_link_credentials(user_id: str, credentials_id: str):
        state.leased.append(credentials_id)
        yield MagicMock(access_token=SecretStr("synthetic-token"))

    monkeypatch.setattr(
        support,
        "_credentials",
        MagicMock(store=MagicMock(get_creds_by_provider=by_provider)),
    )
    monkeypatch.setattr(support, "selected_credentials", picks)
    monkeypatch.setattr(request_tools, "link_credentials", leased_link_credentials)
    return state


@pytest.mark.asyncio
async def test_without_a_credential_the_users_only_link_wallet_pays(
    link_accounts, plan
):
    link_accounts.stored = [
        link_account("wallet-2"),
        link_account("profile-only", scopes=["userinfo:read"]),
    ]

    created = parsed(
        await run(
            "browser_request_link_payment",
            **plan.model_dump(exclude={"credentials_id"}),
        )
    )

    assert created.status == "pending_approval"
    assert link_accounts.leased == ["wallet-2"]


@pytest.mark.asyncio
async def test_among_several_link_wallets_the_one_picked_in_the_chat_pays(
    link_accounts, plan
):
    link_accounts.stored = [link_account("personal"), link_account("work")]
    link_accounts.picked = {"stripe_link": "work"}

    await run(
        "browser_request_link_payment", **plan.model_dump(exclude={"credentials_id"})
    )

    assert link_accounts.leased == ["work"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "stored,refusal",
    [([], LINK_NOT_CONNECTED), (["personal", "work"], LINK_ACCOUNT_NOT_CHOSEN)],
    ids=["none", "several_unpicked"],
)
async def test_without_one_link_wallet_the_agent_is_sent_to_the_connect_card(
    tools, link_accounts, plan, stored, refusal
):
    link_accounts.stored = [link_account(credential_id) for credential_id in stored]

    result = await run(
        "browser_request_link_payment", **plan.model_dump(exclude={"credentials_id"})
    )

    assert ErrorResponse.model_validate_json(result.output).message == refusal
    assert link_accounts.leased == []
    assert tools.calls == []


@pytest.mark.asyncio
async def test_an_invalid_plan_names_what_to_fix_without_echoing_it(tools, plan):
    arguments = plan.model_dump(exclude={"payment_method_id"}) | {
        "context": "a short reason",
        "card_number": "4242424242424242",
    }

    result = await run("browser_request_link_payment", **arguments)

    message = ErrorResponse.model_validate_json(result.output).message
    assert "payment_method_id: Field required" in message
    assert "context: String should have at least 100 characters" in message
    assert "card_number: Extra inputs are not permitted" in message
    assert "4242424242424242" not in result.output
    assert "a short reason" not in result.output
    assert tools.calls == []
