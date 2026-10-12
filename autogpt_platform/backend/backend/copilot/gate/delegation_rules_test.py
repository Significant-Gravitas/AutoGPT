"""``ask_before_external``: outward calls ask in every mode, for Otto and the
threads Otto delegated, while the user keeps the toggle on."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot.context import reset_consult_budget
from backend.copilot.delegation_settings import DelegationSettings
from backend.copilot.gate import delegation_rules
from backend.copilot.gate.policy import Effect, Verdict
from backend.copilot.model import ChatSession


def _session(expert_id: str | None = None, delegated_by: str | None = None):
    session = ChatSession.new("u1", dry_run=False, expert_id=expert_id)
    if delegated_by is not None:
        session.metadata.delegated_by_session_id = delegated_by
    return session


@pytest.fixture
def settings(monkeypatch):
    db = MagicMock()
    db.get_delegation_settings = AsyncMock(return_value=DelegationSettings())
    monkeypatch.setattr(delegation_rules, "delegation_db", lambda: db)
    reset_consult_budget()
    return db


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["ask_first", "auto", "unsupervised"])
async def test_an_outward_call_asks_in_every_mode_for_otto(settings, mode):
    verdict = await delegation_rules.verdict_for(
        mode, Effect.EXTERNAL, "u1", _session()
    )

    assert verdict is Verdict.ASK


@pytest.mark.asyncio
async def test_a_thread_otto_delegated_asks_too(settings):
    thread = _session(expert_id="expert-b", delegated_by="otto-chat")

    verdict = await delegation_rules.verdict_for(
        "unsupervised", Effect.EXTERNAL, "u1", thread
    )

    assert verdict is Verdict.ASK


@pytest.mark.asyncio
async def test_an_experts_own_chat_follows_its_mode(settings):
    verdict = await delegation_rules.verdict_for(
        "unsupervised", Effect.EXTERNAL, "u1", _session(expert_id="expert-b")
    )

    assert verdict is Verdict.RUN
    settings.get_delegation_settings.assert_not_awaited()


@pytest.mark.asyncio
async def test_with_the_toggle_off_the_mode_table_decides(settings):
    settings.get_delegation_settings.return_value = DelegationSettings(
        ask_before_external=False
    )

    verdict = await delegation_rules.verdict_for(
        "unsupervised", Effect.EXTERNAL, "u1", _session()
    )

    assert verdict is Verdict.RUN


@pytest.mark.asyncio
async def test_other_effects_keep_the_mode_table(settings):
    verdict = await delegation_rules.verdict_for(
        "unsupervised", Effect.PLATFORM, "u1", _session()
    )

    assert verdict is Verdict.RUN
    settings.get_delegation_settings.assert_not_awaited()


@pytest.mark.asyncio
async def test_settings_are_read_once_per_turn(settings):
    for _ in range(3):
        await delegation_rules.verdict_for(
            "unsupervised", Effect.EXTERNAL, "u1", _session()
        )
    settings.get_delegation_settings.assert_awaited_once()

    reset_consult_budget()
    await delegation_rules.verdict_for(
        "unsupervised", Effect.EXTERNAL, "u1", _session()
    )
    assert settings.get_delegation_settings.await_count == 2


@pytest.mark.asyncio
async def test_unreadable_settings_keep_asking(settings):
    settings.get_delegation_settings.side_effect = RuntimeError("down")

    verdict = await delegation_rules.verdict_for(
        "unsupervised", Effect.EXTERNAL, "u1", _session()
    )

    assert verdict is Verdict.ASK
