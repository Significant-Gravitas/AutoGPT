from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock

import fastapi
import fastapi.testclient
import pytest
from prisma.enums import APIKeyPermission

from backend.api.external.middleware import require_auth
from backend.api.external.v1.routes import v1_router
from backend.data.auth.base import APIAuthorizationInfo
from backend.util.exceptions import InsufficientBalanceError

app = fastapi.FastAPI()
app.include_router(v1_router)
client = fastapi.testclient.TestClient(app)


@pytest.fixture(autouse=True)
def setup_auth(test_user_id):
    """Override require_auth to return a synthetic API-key principal with all scopes."""

    async def fake_require_auth() -> APIAuthorizationInfo:
        return APIAuthorizationInfo(
            user_id=test_user_id,
            scopes=list(APIKeyPermission),
            type="api_key",
            created_at=datetime.now(timezone.utc),
        )

    app.dependency_overrides[require_auth] = fake_require_auth
    yield
    app.dependency_overrides.clear()


def _stub_block(
    *,
    block_id: str = "00000000-0000-0000-0000-000000000001",
    name: str = "TestBlock",
    disabled: bool = False,
):
    """Build a minimal block stub for get_block(...) replacement.

    Async-iterable execute() yields one (name, value) pair so the route's
    `async for` loop can iterate without touching real block logic.
    """
    block = MagicMock()
    block.id = block_id
    block.name = name
    block.disabled = disabled

    async def _execute(_data):
        yield "result", "ok"

    block.execute = _execute
    return block


def test_zero_balance_returns_402_on_paid_block(monkeypatch: pytest.MonkeyPatch):
    """Zero-credit user calling a paid block must be rejected before execution."""
    block = _stub_block(name="PaidBlock")
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)
    monkeypatch.setattr(
        "backend.executor.utils.block_usage_cost", lambda *_a, **_k: (5, {})
    )
    spend_mock = AsyncMock(
        side_effect=InsufficientBalanceError(
            user_id="test-user-id",
            message="No credits left.",
            balance=0,
            amount=5,
        )
    )
    monkeypatch.setattr(
        "backend.executor.utils.get_user_credit_model",
        AsyncMock(return_value=MagicMock(spend_credits=spend_mock)),
    )

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 402, f"got {response.status_code}: {response.text}"
    spend_mock.assert_awaited_once()


def test_paid_block_charges_then_runs(
    monkeypatch: pytest.MonkeyPatch, test_user_id: str
):
    """Happy path for a paid static-cost block: charge first, then execute."""
    block = _stub_block(name="PaidBlock")
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)
    monkeypatch.setattr(
        "backend.executor.utils.block_usage_cost",
        lambda *_a, **_k: (3, {"matched": True}),
    )
    spend_mock = AsyncMock(return_value=97)
    monkeypatch.setattr(
        "backend.executor.utils.get_user_credit_model",
        AsyncMock(return_value=MagicMock(spend_credits=spend_mock)),
    )

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 200, f"got {response.status_code}: {response.text}"
    assert response.json() == {"result": ["ok"]}
    spend_mock.assert_awaited_once()
    kwargs = spend_mock.await_args.kwargs
    assert kwargs["user_id"] == test_user_id
    assert kwargs["cost"] == 3
    assert kwargs["metadata"].block_id == block.id
    assert kwargs["metadata"].block == "PaidBlock"
    assert kwargs["metadata"].input == {"matched": True}
    assert kwargs["metadata"].reason == "Direct external block execution of PaidBlock"


def test_free_block_runs_without_charging(monkeypatch: pytest.MonkeyPatch):
    """A block with cost == 0 should execute and never call spend_credits."""
    block = _stub_block(name="FreeBlock")
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)
    monkeypatch.setattr(
        "backend.executor.utils.block_usage_cost", lambda *_a, **_k: (0, {})
    )
    spend_mock = AsyncMock()
    monkeypatch.setattr(
        "backend.executor.utils.get_user_credit_model",
        AsyncMock(return_value=MagicMock(spend_credits=spend_mock)),
    )

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 200, f"got {response.status_code}: {response.text}"
    assert response.json() == {"result": ["ok"]}
    spend_mock.assert_not_awaited()


def test_disabled_block_still_403(monkeypatch: pytest.MonkeyPatch):
    """Pre-existing behavior: disabled blocks return 403, not bypassed by new gates."""
    block = _stub_block(name="DisabledBlock", disabled=True)
    monkeypatch.setattr("backend.blocks.get_block", lambda _: block)

    response = client.post(f"/blocks/{block.id}/execute", json={})

    assert response.status_code == 403
    assert "disabled" in response.json()["detail"]


def test_unknown_block_returns_404(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("backend.blocks.get_block", lambda _: None)
    response = client.post("/blocks/00000000-0000-0000-0000-deadbeef/execute", json={})
    assert response.status_code == 404
