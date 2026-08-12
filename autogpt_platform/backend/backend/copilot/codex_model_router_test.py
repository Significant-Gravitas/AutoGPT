from typing import cast
from unittest.mock import AsyncMock

import pytest

from backend.copilot.model_router import resolve_codex_model_route
from backend.data.llm_registry import registry
from backend.integrations.codex.models import CodexModelInfo, CodexReasoningEffort
from backend.integrations.credential_lease import CredentialLease


@pytest.fixture(scope="session", name="server")
def _server_noop():
    return None


@pytest.fixture(scope="session", autouse=True, name="graph_cleanup")
def _graph_cleanup_noop():
    yield


def _model(
    model: str,
    *,
    default: bool = False,
    hidden: bool = False,
    default_effort: CodexReasoningEffort = "medium",
    efforts: list[CodexReasoningEffort] | None = None,
) -> CodexModelInfo:
    return CodexModelInfo(
        model=model,
        display_name=model,
        is_default=default,
        hidden=hidden,
        default_reasoning_effort=default_effort,
        supported_reasoning_efforts=efforts or ["low", "medium", "high", "xhigh"],
        input_modalities=["text"],
    )


def _lease() -> CredentialLease:
    return cast(CredentialLease, object())


@pytest.fixture
def catalog_state():
    old = (
        registry._dynamic_models,
        registry._date_stripped_models,
        registry._routes,
        registry._loaded,
    )
    registry.load_catalog()
    yield
    (
        registry._dynamic_models,
        registry._date_stripped_models,
        registry._routes,
        registry._loaded,
    ) = old


def _transport(monkeypatch, models: list[CodexModelInfo]):
    transport = AsyncMock()
    transport.models.return_value = models
    monkeypatch.setattr(
        "backend.copilot.model_router.get_codex_transport",
        lambda: transport,
    )
    return transport


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "mode,tier,expected_model,expected_effort",
    [
        ("fast", "standard", "gpt-5.6-luna", "low"),
        ("fast", "advanced", "gpt-5.6-terra", "medium"),
        ("thinking", "standard", "gpt-5.6-terra", "high"),
        ("thinking", "advanced", "gpt-5.6-sol", "xhigh"),
    ],
)
async def test_catalog_cells_select_latest_advertised_model(
    monkeypatch,
    catalog_state,
    mode,
    tier,
    expected_model,
    expected_effort,
):
    models = [
        _model("gpt-5.6-luna"),
        _model("gpt-5.6-terra"),
        _model("gpt-5.6-sol", default=True),
    ]
    transport = _transport(monkeypatch, models)
    lease = _lease()

    resolved = await resolve_codex_model_route(
        mode,
        tier,
        lease,
    )

    assert resolved == (expected_model, expected_effort, "catalog")
    transport.models.assert_awaited_once_with(lease)


@pytest.mark.asyncio
async def test_unavailable_catalog_model_uses_visible_account_default(
    monkeypatch,
    catalog_state,
):
    _transport(
        monkeypatch,
        [
            _model("codex-auto-review", hidden=True),
            _model(
                "gpt-5.4",
                default=True,
                default_effort="low",
                efforts=["low"],
            ),
        ],
    )

    resolved = await resolve_codex_model_route(
        "thinking",
        "advanced",
        _lease(),
    )

    assert resolved == ("gpt-5.4", "low", "account_default")


@pytest.mark.asyncio
async def test_no_default_uses_first_visible_account_model(monkeypatch, catalog_state):
    _transport(
        monkeypatch,
        [
            _model("codex-auto-review", hidden=True),
            _model("gpt-5.2", efforts=["medium"]),
        ],
    )

    resolved = await resolve_codex_model_route(
        "fast",
        "standard",
        _lease(),
    )

    assert resolved == ("gpt-5.2", "medium", "account_available")


@pytest.mark.asyncio
async def test_empty_visible_catalog_fails_closed(monkeypatch, catalog_state):
    _transport(monkeypatch, [_model("codex-auto-review", hidden=True, default=True)])

    with pytest.raises(RuntimeError, match="codex_model_unavailable"):
        await resolve_codex_model_route(
            "fast",
            "standard",
            _lease(),
        )


@pytest.mark.asyncio
async def test_disabled_account_default_is_not_used(monkeypatch, catalog_state):
    disabled = registry._dynamic_models["gpt-5.4-2026-03-05"].model_copy(
        update={"is_enabled": False}
    )
    registry._dynamic_models["gpt-5.4-2026-03-05"] = disabled
    registry._date_stripped_models["gpt-5.4"] = disabled
    _transport(
        monkeypatch,
        [
            _model("gpt-5.4", default=True),
            _model("gpt-5.2", efforts=["medium"]),
        ],
    )

    resolved = await resolve_codex_model_route(
        "fast",
        "standard",
        _lease(),
    )

    assert resolved == ("gpt-5.2", "medium", "account_available")
