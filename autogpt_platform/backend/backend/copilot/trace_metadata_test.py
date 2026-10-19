import hashlib
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
import sentry_sdk
from langfuse import propagate_attributes
from sentry_sdk.envelope import Envelope
from sentry_sdk.transport import Transport

from backend.copilot import service, trace_metadata
from backend.util import metrics
from backend.util.settings import Config


@pytest.mark.parametrize("release", ["build-a", "  v2026.10.19\n", "", " \n"])
def test_trace_release_matches_deployment_config(
    monkeypatch: pytest.MonkeyPatch, release: str
) -> None:
    monkeypatch.setenv("SENTRY_RELEASE", release)
    monkeypatch.setattr(trace_metadata.settings, "config", Config())

    metadata = trace_metadata.copilot_trace_metadata("base prompt")

    if release.strip():
        assert metadata["release"] == release.strip()
    else:
        assert "release" not in metadata


def test_prompt_hash_identifies_exact_resolved_base_prompt() -> None:
    prompt = "You are Otto.\n日本語"

    first = trace_metadata.copilot_trace_metadata(prompt)
    repeated = trace_metadata.copilot_trace_metadata(prompt)
    changed = trace_metadata.copilot_trace_metadata(prompt + "\nNew instruction.")

    assert first["base_prompt_sha256"] == hashlib.sha256(prompt.encode()).hexdigest()
    assert first == repeated
    assert first["base_prompt_sha256"] != changed["base_prompt_sha256"]
    assert prompt not in first.values()


@pytest.mark.asyncio
@pytest.mark.parametrize("fails", [False, True], ids=["success", "error"])
async def test_separate_title_generation_carries_release(
    monkeypatch: pytest.MonkeyPatch, fails: bool
) -> None:
    monkeypatch.setattr(trace_metadata.settings.config, "sentry_release", "build-title")
    complete = AsyncMock(
        side_effect=RuntimeError("title model unavailable") if fails else None,
        return_value=MagicMock(choices=[]),
    )
    monkeypatch.setattr(service, "call_provider_openai_compat_sync", complete)
    monkeypatch.setattr(service, "_get_aux_client", MagicMock())
    attributes = MagicMock(wraps=propagate_attributes)
    monkeypatch.setattr(service, "propagate_attributes", attributes)
    await service._generate_session_title("Please organize my notes")
    complete.assert_awaited_once()
    assert attributes.call_args.kwargs["metadata"] == {"release": "build-title"}


class _EventCapture(Transport):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[dict[str, Any]] = []

    def capture_envelope(self, envelope: Envelope) -> None:
        self.events.extend(
            item.payload.json
            for item in envelope.items
            if item.type in ("event", "transaction") and item.payload.json is not None
        )


def test_sentry_events_and_transactions_match_langfuse_release(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SENTRY_RELEASE", " build-from-deploy\n")
    config = Config()
    monkeypatch.setattr(trace_metadata.settings, "config", config)
    monkeypatch.setattr(metrics.settings, "config", config)
    monkeypatch.setattr(metrics, "_running_under_pytest", lambda: False)
    init = MagicMock()
    monkeypatch.setattr(metrics, "_sentry_init", init)
    metrics.sentry_init()

    capture = _EventCapture()
    options = init.call_args.kwargs | {
        "dsn": "https://key@o1.ingest.us.sentry.io/1",
        "transport": capture,
        "integrations": [],
        "default_integrations": False,
        "auto_enabling_integrations": False,
    }
    client = sentry_sdk.Client(**options)
    try:
        with sentry_sdk.new_scope() as scope:
            scope.set_client(client)
            sentry_sdk.capture_exception(RuntimeError("release-stamp test"))
            with sentry_sdk.start_transaction(name="copilot.turn", sampled=True):
                pass
    finally:
        client.close()

    assert len(capture.events) == 2
    assert (
        {event["release"] for event in capture.events}
        == {trace_metadata.copilot_trace_metadata()["release"]}
        == {"build-from-deploy"}
    )
