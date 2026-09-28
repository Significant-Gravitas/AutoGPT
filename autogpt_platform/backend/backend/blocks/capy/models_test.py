"""Tests for Capy model routing: linked providers, rejections and the balance fallback.

The rejection payloads are the ones Capy returned live on 2026-09-28 for an
account with Codex linked but disconnected, no Copilot link, and no Azure
organization account.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.blocks.capy import _api
from backend.blocks.capy._api import CapyAPIError, _error, with_capy_balance_fallback
from backend.blocks.capy._config import TEST_CREDENTIALS, TEST_CREDENTIALS_INPUT
from backend.blocks.capy._models import (
    ModelRoute,
    billed_via,
    is_linked_route,
    resolve_model_id,
)
from backend.blocks.capy._testdata import TEST_PROJECT, TEST_THREAD
from backend.blocks.capy.create_thread import CapyCreateThreadBlock

CODEX_DISCONNECTED = {
    "_tag": "ModelSelection.Rejected",
    "message": "Codex disconnected, reconnect it in Settings",
    "candidates": [],
    "rejection": "disconnected",
    "modelId": "codex/gpt-6-astra",
    "service": "codex",
}
COPILOT_NOT_CONNECTED = {
    "_tag": "ModelSelection.Rejected",
    "message": "Model requires a connection that does not exist yet: "
    "copilot/claude-opus-4-8 (oauth:copilot)",
    "candidates": [],
    "rejection": "not_connected",
    "modelId": "copilot/claude-opus-4-8",
    "service": "copilot",
}
AZURE_MISSING = {
    "_tag": "ModelSelection.Rejected",
    "message": "Model requires a connection that does not exist yet: "
    "azure/claude-fable-5 (org)",
    "candidates": [],
    "rejection": None,
    "modelId": "azure/claude-fable-5",
    "service": None,
}


def _rejected(body: dict[str, Any]) -> CapyAPIError:
    response = MagicMock()
    response.status = 400
    response.json.return_value = body
    return _error(response)


async def _run_create(**inputs) -> dict[str, Any]:
    block = CapyCreateThreadBlock()
    out: dict[str, Any] = {}
    async for name, value in block.run(
        block.input_schema(
            credentials=TEST_CREDENTIALS_INPUT,
            project_id=TEST_PROJECT.id,
            message="Fix it",
            **inputs,
        ),
        credentials=TEST_CREDENTIALS,
    ):
        out[name] = value
    return out


class TestRouting:
    @pytest.mark.parametrize(
        "model_id,route,expected",
        [
            ("gpt-6-astra", ModelRoute.AS_GIVEN, "gpt-6-astra"),
            ("openai/gpt-6-astra", ModelRoute.CODEX, "codex/gpt-6-astra"),
            ("gpt-6-astra", ModelRoute.CODEX, "codex/gpt-6-astra"),
            ("codex/gpt-6-astra", ModelRoute.CAPY_BALANCE, "openai/gpt-6-astra"),
            ("muse-spark-1.3", ModelRoute.CAPY_BALANCE, "meta/muse-spark-1.3"),
            (
                "copilot/claude-opus-4-8",
                ModelRoute.CAPY_BALANCE,
                "anthropic/claude-opus-4-8",
            ),
            ("supergrok/grok-4.5", ModelRoute.CAPY_BALANCE, "xai/grok-4.5"),
            (
                "anthropic/claude-sonnet-5",
                ModelRoute.CAPY_BALANCE,
                "anthropic/claude-sonnet-5",
            ),
            ("new-family-1", ModelRoute.CAPY_BALANCE, "new-family-1"),
            ("grok-4.5", ModelRoute.SUPERGROK, "supergrok/grok-4.5"),
            ("claude-opus-4-8", ModelRoute.COPILOT, "copilot/claude-opus-4-8"),
            ("", ModelRoute.CODEX, ""),
        ],
    )
    def test_resolve_model_id(self, model_id: str, route: ModelRoute, expected: str):
        assert resolve_model_id(model_id, route) == expected

    def test_billed_via(self):
        assert billed_via("codex/gpt-6-astra") == "Codex (ChatGPT subscription)"
        assert billed_via("supergrok/grok-4.5") == "SuperGrok subscription"
        assert billed_via("azure/claude-fable-5") == "Azure organization account"
        assert billed_via("openai/gpt-6-astra") == "Capy balance"
        assert billed_via("gpt-6-astra") == "Capy balance"
        assert billed_via(None) == ""

    def test_is_linked_route(self):
        assert is_linked_route("copilot/gpt-5.5")
        assert not is_linked_route("anthropic/claude-opus-5-5")
        assert not is_linked_route("gpt-6-astra")


class TestRejectionMessages:
    def test_disconnected_says_reconnect(self):
        err = _rejected(CODEX_DISCONNECTED)

        assert err.rejection == "disconnected"
        assert err.service == "codex"
        assert "Codex (ChatGPT subscription) linked in Capy is disconnected" in str(err)
        assert "reconnect it" in str(err)

    def test_not_connected_says_link_it(self):
        err = _rejected(COPILOT_NOT_CONNECTED)

        assert "no GitHub Copilot subscription is linked" in str(err)
        assert "service-user key" in str(err)

    def test_org_route_without_rejection_code(self):
        err = _rejected(AZURE_MISSING)

        assert "Azure organization account" in str(err)
        assert "not set up" in str(err)


class TestBalanceFallback:
    async def test_falls_back_when_the_linked_provider_is_down(self):
        call = AsyncMock(side_effect=[_rejected(CODEX_DISCONNECTED), "thread"])

        result, used = await with_capy_balance_fallback(
            call, "codex/gpt-6-astra", fallback=True
        )

        assert (result, used) == ("thread", "openai/gpt-6-astra")
        assert [c.args[0] for c in call.await_args_list] == [
            "codex/gpt-6-astra",
            "openai/gpt-6-astra",
        ]

    async def test_no_fallback_unless_asked(self):
        call = AsyncMock(side_effect=_rejected(CODEX_DISCONNECTED))

        with pytest.raises(CapyAPIError):
            await with_capy_balance_fallback(call, "codex/gpt-6-astra", fallback=False)
        assert call.await_count == 1

    async def test_no_fallback_for_a_balance_model(self):
        unknown = _rejected(
            {"_tag": "ModelSelection.Rejected", "message": "Unknown model: x"}
        )
        call = AsyncMock(side_effect=unknown)

        with pytest.raises(CapyAPIError):
            await with_capy_balance_fallback(call, "openai/x", fallback=True)
        assert call.await_count == 1

    async def test_other_errors_are_not_retried(self):
        call = AsyncMock(side_effect=CapyAPIError(403, "capy/Forbidden", "no"))

        with pytest.raises(CapyAPIError):
            await with_capy_balance_fallback(call, "codex/gpt-6-astra", fallback=True)
        assert call.await_count == 1


class TestCreateThreadRouting:
    async def test_route_and_fallback_reach_the_api(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        create = AsyncMock(side_effect=[_rejected(CODEX_DISCONNECTED), TEST_THREAD])
        monkeypatch.setattr(_api.CapyClient, "create_thread", create)
        monkeypatch.setattr(_api, "Requests", MagicMock())

        out = await _run_create(
            model_id="gpt-6-astra",
            model_route=ModelRoute.CODEX,
            fall_back_to_capy_balance=True,
            request_id="job-7",
        )

        first, second = (c.kwargs for c in create.await_args_list)
        assert first["model_id"] == "codex/gpt-6-astra"
        assert first["request_id"] == "job-7"
        assert second["model_id"] == "openai/gpt-6-astra"
        assert second["request_id"] == "job-7-capy-balance"
        assert out["model_id"] == "openai/gpt-6-astra"
        assert out["billed_via"] == "Capy balance"

    async def test_project_default_model_sends_no_model(
        self, monkeypatch: pytest.MonkeyPatch
    ):
        create = AsyncMock(return_value=TEST_THREAD)
        monkeypatch.setattr(_api.CapyClient, "create_thread", create)
        monkeypatch.setattr(_api, "Requests", MagicMock())

        out = await _run_create()

        assert create.await_args is not None
        assert create.await_args.kwargs["model_id"] == ""
        assert out["model_id"] == ""
        assert out["billed_via"] == ""
