"""Tests for CoPilot executor utils (queue config, message models, logging)."""

from collections.abc import Iterator
from contextlib import asynccontextmanager, contextmanager
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot import usage_activation_fixtures
from backend.copilot.continuation import KEEP_GOING_NOTE
from backend.copilot.credential_selection import CredentialPin
from backend.copilot.executor import utils
from backend.copilot.executor.utils import (
    COPILOT_CANCEL_EXCHANGE,
    COPILOT_EXECUTION_EXCHANGE,
    COPILOT_EXECUTION_QUEUE_NAME,
    COPILOT_EXECUTION_ROUTING_KEY,
    CancelCoPilotEvent,
    CoPilotExecutionEntry,
    CoPilotLogMetadata,
    ScheduledTurnOrigin,
    create_copilot_queue_config,
    declare_pod_cancel_queue,
)
from backend.copilot.permissions import CopilotPermissions
from backend.copilot.prompt_origin import PromptOrigin, split_prompt_origin
from backend.copilot.tree import ALL_TOOL_NAMES, root_envelope
from backend.copilot.trial_cost_context import TrialCostContext


@pytest.mark.asyncio
async def test_schedule_chat_turn_redispatches_admitted_persisted_message() -> None:
    slot = MagicMock(admitted=True)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    append = AsyncMock()
    dispatch = AsyncMock()
    with (
        patch.object(utils, "acquire_turn_slot", new=acquire),
        patch("backend.copilot.model.append_and_save_message", new=append),
        patch.object(utils, "dispatch_turn", new=dispatch),
    ):
        turn_id = await utils.schedule_chat_turn(
            session_id="s1",
            user_id="u1",
            message="kickoff",
            message_already_persisted=True,
        )

    assert turn_id is not None
    append.assert_not_awaited()
    dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_schedule_chat_turn_skips_concurrent_persisted_redispatch() -> None:
    slot = MagicMock(admitted=False)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    dispatch = AsyncMock()
    with (
        patch.object(utils, "acquire_turn_slot", new=acquire),
        patch.object(utils, "dispatch_turn", new=dispatch),
    ):
        turn_id = await utils.schedule_chat_turn(
            session_id="s1",
            user_id="u1",
            message="kickoff",
            message_already_persisted=True,
        )

    assert turn_id is None
    dispatch.assert_not_awaited()


class TestCoPilotExecutionEntry:
    def test_basic_fields(self):
        entry = CoPilotExecutionEntry(
            session_id="s1",
            user_id="u1",
            message="hello",
        )
        assert entry.session_id == "s1"
        assert entry.user_id == "u1"
        assert entry.message == "hello"
        assert entry.is_user_message is True
        assert not hasattr(entry, "mode")
        assert entry.context is None
        assert entry.file_ids is None

    def test_optional_fields(self):
        entry = CoPilotExecutionEntry(
            session_id="s1",
            user_id="u1",
            message="test",
            turn_id="t1",
            context={"url": "https://example.com"},
            file_ids=["f1", "f2"],
            is_user_message=False,
        )
        assert entry.turn_id == "t1"
        assert entry.context == {"url": "https://example.com"}
        assert entry.file_ids == ["f1", "f2"]
        assert entry.is_user_message is False

    def test_serialization_roundtrip(self):
        entry = CoPilotExecutionEntry(
            session_id="s1",
            user_id="u1",
            message="hello",
        )
        json_str = entry.model_dump_json()
        restored = CoPilotExecutionEntry.model_validate_json(json_str)
        assert restored == entry

    def test_unattended_survives_the_queue_and_defaults_to_watched(self):
        entry = CoPilotExecutionEntry(
            session_id="s1", user_id="u1", message="hi", unattended=True
        )
        restored = CoPilotExecutionEntry.model_validate_json(entry.model_dump_json())
        assert restored.unattended is True
        # A message queued by a worker without the field is a watched turn.
        legacy = CoPilotExecutionEntry.model_validate_json(
            '{"session_id": "s1", "user_id": "u1", "message": "hi"}'
        )
        assert legacy.unattended is False

    def test_impersonated_survives_the_queue_and_defaults_to_false(self):
        entry = CoPilotExecutionEntry(
            session_id="s1", user_id="u1", message="hi", impersonated=True
        )
        restored = CoPilotExecutionEntry.model_validate_json(entry.model_dump_json())
        assert restored.impersonated is True
        legacy = CoPilotExecutionEntry.model_validate_json(
            '{"session_id": "s1", "user_id": "u1", "message": "hi"}'
        )
        assert legacy.impersonated is False

    def test_credential_pins_survive_the_queue_and_default_to_none(self):
        pin = CredentialPin(id="exa-new", title="Work key")
        entry = CoPilotExecutionEntry(
            session_id="s1", user_id="u1", message="hi", credential_pins={"exa": pin}
        )
        restored = CoPilotExecutionEntry.model_validate_json(entry.model_dump_json())
        assert restored.credential_pins == {"exa": pin}
        legacy = CoPilotExecutionEntry.model_validate_json(
            '{"session_id": "s1", "user_id": "u1", "message": "hi"}'
        )
        assert legacy.credential_pins == {}


class TestCancelCoPilotEvent:
    def test_basic(self):
        event = CancelCoPilotEvent(session_id="s1")
        assert event.session_id == "s1"

    def test_serialization(self):
        event = CancelCoPilotEvent(session_id="s1")
        restored = CancelCoPilotEvent.model_validate_json(event.model_dump_json())
        assert restored.session_id == "s1"


class TestCreateCopilotQueueConfig:
    def test_returns_valid_config(self):
        config = create_copilot_queue_config()
        assert len(config.exchanges) == 2
        assert len(config.queues) == 1

    def test_execution_queue_properties(self):
        config = create_copilot_queue_config()
        exec_queue = next(
            q for q in config.queues if q.name == COPILOT_EXECUTION_QUEUE_NAME
        )
        assert exec_queue.durable is True
        assert exec_queue.exchange == COPILOT_EXECUTION_EXCHANGE
        assert exec_queue.routing_key == COPILOT_EXECUTION_ROUTING_KEY

    def test_config_declares_no_shared_cancel_queue(self):
        """A queue in this config is declared by every holder, the publishing
        API included, and shared by every consumer — which is what made the
        fanout deliver each cancel to one pod."""
        config = create_copilot_queue_config()
        assert COPILOT_CANCEL_EXCHANGE in config.exchanges
        assert all(q.exchange != COPILOT_CANCEL_EXCHANGE for q in config.queues)


class TestDeclarePodCancelQueue:
    def test_queue_is_exclusive_auto_delete_and_bound_to_the_fanout(self):
        channel = MagicMock()
        name = declare_pod_cancel_queue(channel, "executor-1")
        channel.queue_declare.assert_called_once_with(
            queue=name, durable=False, exclusive=True, auto_delete=True
        )
        channel.queue_bind.assert_called_once_with(
            queue=name, exchange=COPILOT_CANCEL_EXCHANGE.name, routing_key=""
        )

    def test_each_pod_gets_its_own_queue(self):
        names = {
            declare_pod_cancel_queue(MagicMock(), executor_id)
            for executor_id in ("executor-1", "executor-2", "executor-1")
        }
        assert len(names) == 3


class TestCoPilotLogMetadata:
    def test_creates_logger_with_metadata(self):
        import logging

        base_logger = logging.getLogger("test")
        log = CoPilotLogMetadata(base_logger, session_id="s1", user_id="u1")
        assert log is not None

    def test_filters_none_values(self):
        import logging

        base_logger = logging.getLogger("test")
        log = CoPilotLogMetadata(
            base_logger, session_id="s1", user_id=None, turn_id="t1"
        )
        assert log is not None


@pytest.mark.asyncio
@pytest.mark.parametrize("from_chat_platform", [True, False])
async def test_a_chat_platform_dispatch_queues_a_tainted_envelope(
    from_chat_platform: bool,
) -> None:
    """The envelope carries the taint from dispatch, not only the entry's own
    field: an executor that predates the field still runs a bot's prompt
    tainted, in a chat started on the web as much as in the bot's own."""
    slot = MagicMock(admitted=True)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    queue = MagicMock(publish_message=AsyncMock())
    with (
        patch.object(utils, "acquire_turn_slot", new=acquire),
        patch.object(utils, "admit_turn", new=AsyncMock()),
        patch.object(
            utils,
            "capture_cost_context",
            new=AsyncMock(return_value=TrialCostContext(user_id="u1", trial_id=None)),
        ),
        patch(
            "backend.copilot.model.append_and_save_message",
            new=AsyncMock(return_value=MagicMock()),
        ),
        patch("backend.copilot.tracking.track_user_message", new=MagicMock()),
        patch("backend.copilot.stream_registry.create_session", new=AsyncMock()),
        patch(
            "backend.util.clients.get_async_copilot_queue",
            new=AsyncMock(return_value=queue),
        ),
    ):
        await utils.schedule_chat_turn(
            session_id="s1",
            user_id="u1",
            message="hi from the bot",
            from_chat_platform=from_chat_platform,
        )

    call = queue.publish_message.await_args
    assert call is not None
    entry = utils.CoPilotExecutionEntry.model_validate_json(call.kwargs["message"])
    assert entry.envelope is not None
    assert entry.envelope.tainted is from_chat_platform
    assert entry.from_chat_platform is from_chat_platform


@pytest.mark.asyncio
async def test_schedule_chat_turn_saves_the_spoken_words_it_dispatches() -> None:
    # The engine dedups the dispatched message against the row saved here,
    # so the two must match; when they diverged the turn was saved twice.
    # The voice instruction travels on the turn, not in either of them.
    slot = MagicMock(admitted=True)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    append = AsyncMock()
    dispatch = AsyncMock()
    with (
        patch.object(utils, "acquire_turn_slot", new=acquire),
        patch("backend.copilot.model.append_and_save_message", new=append),
        patch("backend.copilot.tracking.track_user_message", new=MagicMock()),
        patch.object(utils, "dispatch_turn", new=dispatch),
    ):
        await utils.schedule_chat_turn(
            session_id="s1",
            user_id="u1",
            message="what did I run yesterday",
            voice=True,
        )

    persisted = append.await_args.args[1].content
    assert persisted == dispatch.await_args.kwargs["message"]
    assert persisted == "what did I run yesterday"
    assert dispatch.await_args_list[-1].kwargs["voice"] is True


@pytest.mark.asyncio
async def test_schedule_chat_turn_leaves_a_typed_message_alone() -> None:
    slot = MagicMock(admitted=True)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    append = AsyncMock()
    dispatch = AsyncMock()
    with (
        patch.object(utils, "acquire_turn_slot", new=acquire),
        patch("backend.copilot.model.append_and_save_message", new=append),
        patch("backend.copilot.tracking.track_user_message", new=MagicMock()),
        patch.object(utils, "dispatch_turn", new=dispatch),
    ):
        await utils.schedule_chat_turn(
            session_id="s1",
            user_id="u1",
            message="what did I run yesterday",
        )

    assert append.await_args.args[1].content == "what did I run yesterday"
    assert dispatch.await_args.kwargs["message"] == "what did I run yesterday"
    assert dispatch.await_args.kwargs["impersonated"] is False


@pytest.mark.asyncio
async def test_schedule_chat_turn_hands_impersonation_to_the_turn() -> None:
    slot = MagicMock(admitted=True)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    dispatch = AsyncMock()
    with (
        patch.object(utils, "acquire_turn_slot", new=acquire),
        patch("backend.copilot.model.append_and_save_message", new=AsyncMock()),
        patch("backend.copilot.tracking.track_user_message", new=MagicMock()),
        patch.object(utils, "dispatch_turn", new=dispatch),
    ):
        await utils.schedule_chat_turn(
            session_id="s1", user_id="u1", message="hi", impersonated=True
        )

    assert dispatch.await_args.kwargs["impersonated"] is True


@pytest.mark.asyncio
async def test_schedule_chat_turn_tracks_the_session_channel() -> None:
    # A web message in a session opened from Discord still counts as Discord,
    # the same as the engines report it for turns they save themselves.
    slot = MagicMock(admitted=True)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    tracked = MagicMock()
    with (
        patch.object(utils, "acquire_turn_slot", new=acquire),
        patch("backend.copilot.model.append_and_save_message", new=AsyncMock()),
        patch("backend.copilot.tracking.track_user_message", new=tracked),
        patch.object(utils, "dispatch_turn", new=AsyncMock()),
    ):
        await utils.schedule_chat_turn(
            session_id="s1",
            user_id="u1",
            message="hello",
            expert_id="expert-1",
            session_origin="interactive",
            session_source_platform="discord",
        )

    tracked.assert_called_once_with(
        user_id="u1",
        session_id="s1",
        message_length=5,
        expert_id="expert-1",
        origin="interactive",
        source_platform="discord",
    )


@pytest.mark.asyncio
async def test_schedule_chat_turn_leaves_an_already_saved_message_alone() -> None:
    # The row was saved by an earlier call; changing the text now would put
    # the two out of step, which is the duplicate this guard exists to
    # prevent.
    slot = MagicMock(admitted=True)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    dispatch = AsyncMock()
    with (
        patch.object(utils, "acquire_turn_slot", new=acquire),
        patch.object(utils, "dispatch_turn", new=dispatch),
    ):
        await utils.schedule_chat_turn(
            session_id="s1",
            user_id="u1",
            message="kick it off",
            message_already_persisted=True,
            voice=True,
        )

    assert dispatch.await_args.kwargs["message"] == "kick it off"


@contextmanager
def _scheduling(*, admitted: bool) -> Iterator[tuple[AsyncMock, AsyncMock]]:
    """``schedule_chat_turn`` over a slot this call did or didn't admit."""
    slot = MagicMock(admitted=admitted)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    append = AsyncMock()
    dispatch = AsyncMock()
    with (
        patch.object(utils, "acquire_turn_slot", new=acquire),
        patch("backend.copilot.model.append_and_save_message", new=append),
        patch("backend.copilot.tracking.track_user_message", new=MagicMock()),
        patch.object(utils, "dispatch_turn", new=dispatch),
    ):
        yield append, dispatch


@pytest.mark.asyncio
async def test_schedule_chat_turn_needing_a_fresh_slot_saves_nothing_on_a_held_chat():
    # Another turn reserved the chat first, so this call only refreshed its
    # slot: saving the note and dispatching would run two turns at once.
    with _scheduling(admitted=False) as (append, dispatch):
        turn_id = await utils.schedule_chat_turn(
            session_id="s1",
            user_id="u1",
            message=KEEP_GOING_NOTE,
            is_user_message=False,
            require_fresh_slot=True,
        )

    assert turn_id is None
    append.assert_not_awaited()
    dispatch.assert_not_awaited()


@pytest.mark.asyncio
async def test_schedule_chat_turn_needing_a_fresh_slot_dispatches_when_admitted():
    with _scheduling(admitted=True) as (append, dispatch):
        turn_id = await utils.schedule_chat_turn(
            session_id="s1",
            user_id="u1",
            message=KEEP_GOING_NOTE,
            is_user_message=False,
            require_fresh_slot=True,
        )

    assert turn_id is not None
    append.assert_awaited_once()
    dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_schedule_chat_turn_still_saves_and_dispatches_on_a_refreshed_slot():
    # /stream doesn't ask for a fresh slot; its behaviour is unchanged.
    with _scheduling(admitted=False) as (append, dispatch):
        turn_id = await utils.schedule_chat_turn(
            session_id="s1", user_id="u1", message="and the totals"
        )

    assert turn_id is not None
    append.assert_awaited_once()
    dispatch.assert_awaited_once()


@pytest.mark.asyncio
async def test_schedule_chat_turn_dedupes_content_by_default():
    with _scheduling(admitted=True) as (append, _dispatch):
        await utils.schedule_chat_turn(
            session_id="s1", user_id="u1", message="and the totals"
        )

    assert append.await_args.kwargs["dedupe_content"] is True


@pytest.mark.asyncio
async def test_schedule_chat_turn_passes_content_dedupe_off_to_the_save():
    with _scheduling(admitted=True) as (append, _dispatch):
        await utils.schedule_chat_turn(
            session_id="s1",
            user_id="u1",
            message=KEEP_GOING_NOTE,
            is_user_message=False,
            dedupe_content=False,
        )

    assert append.await_args.kwargs["dedupe_content"] is False


class TestNarrowPermissions:
    """The envelope's tool set becomes the turn's whitelist. The encoding of
    "no tools at all" is the trap: it is a blacklist of every tool, and a
    merge that hardcodes the whitelist flag reads it back as its opposite.
    """

    def test_a_locked_envelope_stays_locked_when_merged_with_a_block_filter(
        self,
    ) -> None:
        locked = root_envelope("t").model_copy(update={"tools": frozenset()})
        caller = CopilotPermissions(blocks=["some-block"], blocks_exclude=False)

        merged = utils._narrow_permissions(caller, locked)

        assert merged is not None
        # Deny-all is encoded as "blacklist everything"; flipping the flag
        # would turn the same list into "whitelist everything".
        assert merged.tools_exclude is True
        assert merged.effective_allowed_tools(ALL_TOOL_NAMES) == set()
        assert merged.blocks == ["some-block"]

    def test_a_narrowed_envelope_merges_as_a_whitelist(self) -> None:
        narrowed = root_envelope("t").model_copy(
            update={"tools": frozenset({"read_workspace_file"})}
        )
        caller = CopilotPermissions(blocks=["some-block"], blocks_exclude=False)

        merged = utils._narrow_permissions(caller, narrowed)

        assert merged is not None
        assert merged.tools_exclude is False
        assert merged.effective_allowed_tools(ALL_TOOL_NAMES) == {"read_workspace_file"}


def _scheduled_entry(**scheduled_fields) -> CoPilotExecutionEntry:
    return CoPilotExecutionEntry(
        session_id="sess-1",
        user_id="user-1",
        message="Check the inbox.",
        is_user_message=False,
        message_metadata={"from_session_id": "s-2"},
        scheduled=ScheduledTurnOrigin(**scheduled_fields),
    )


def test_row_metadata_stamps_a_follow_up_prompt() -> None:
    fired_at = datetime(2026, 10, 5, 9, 0, tzinfo=UTC)
    entry = _scheduled_entry(schedule_id="job-1", fired_at=fired_at)

    rest, origin = split_prompt_origin(entry.row_metadata())

    assert rest == {"from_session_id": "s-2"}
    assert origin == PromptOrigin(
        kind="scheduled_followup", schedule_id="job-1", fired_at=fired_at
    )
    assert entry.message_metadata == {"from_session_id": "s-2"}


def test_row_metadata_marks_a_routine_prompt() -> None:
    entry = _scheduled_entry(schedule_id="job-1", routine_id="routine-1")

    _, origin = split_prompt_origin(entry.row_metadata())

    assert origin is not None
    assert origin.kind == "scheduled_routine"
    assert origin.routine_id == "routine-1"
    assert origin.fired_at is None


def test_row_metadata_leaves_unscheduled_turns_alone() -> None:
    entry = CoPilotExecutionEntry(
        session_id="sess-1",
        user_id="user-1",
        message="hi",
        message_metadata={"from_session_id": "s-2"},
    )

    assert entry.row_metadata() is entry.message_metadata


@pytest.mark.asyncio
async def test_dispatch_turn_puts_impersonation_on_the_queued_entry() -> None:
    queue = MagicMock(publish_message=AsyncMock())
    with (
        patch.object(
            utils,
            "_admitted_turn_envelope",
            new=AsyncMock(return_value=root_envelope("t1")),
        ),
        patch("backend.copilot.stream_registry.create_session", new=AsyncMock()),
        patch(
            "backend.util.clients.get_async_copilot_queue",
            new=AsyncMock(return_value=queue),
        ),
    ):
        await utils.dispatch_turn(
            MagicMock(),
            session_id="s1",
            user_id="u1",
            turn_id="t1",
            message="hi",
            impersonated=True,
        )

    entry = CoPilotExecutionEntry.model_validate_json(
        queue.publish_message.await_args.kwargs["message"]
    )
    assert entry.impersonated is True


@pytest.mark.asyncio
async def test_schedule_turn_hands_impersonation_to_the_turn() -> None:
    slot = MagicMock(admitted=True)

    @asynccontextmanager
    async def acquire(*_args, **_kwargs):
        yield slot

    dispatch = AsyncMock()
    with (
        patch(
            "backend.copilot.turn_queue.count_inflight_turns",
            new=AsyncMock(return_value=0),
        ),
        patch.object(utils, "acquire_turn_slot", new=acquire),
        patch.object(utils, "dispatch_turn", new=dispatch),
    ):
        await utils.schedule_turn(
            session_id="s1", user_id="u1", turn_id="t1", message="hi"
        )
        assert dispatch.await_args.kwargs["impersonated"] is False

        await utils.schedule_turn(
            session_id="s1",
            user_id="u1",
            turn_id="t2",
            message="hi",
            impersonated=True,
        )
        assert dispatch.await_args.kwargs["impersonated"] is True


@pytest.mark.asyncio
async def test_queue_spawned_turn_stores_impersonation_with_the_queued_row() -> None:
    enqueue = AsyncMock()
    with (
        patch(
            "backend.copilot.turn_queue.count_inflight_turns",
            new=AsyncMock(return_value=0),
        ),
        patch.object(
            utils,
            "_admitted_turn_envelope",
            new=AsyncMock(return_value=root_envelope("t1")),
        ),
        patch("backend.copilot.turn_queue.enqueue_turn", new=enqueue),
    ):
        await utils.queue_spawned_turn(
            session_id="s1",
            user_id="u1",
            message="hi",
            tool_call_id="sub:parent",
            tool_name="run_sub_session",
            llm_auth_provider="platform",
            llm_credential_id=None,
            permissions=None,
            spawn=None,
            message_metadata=None,
            impersonated=True,
        )

    assert enqueue.await_args.kwargs["impersonated"] is True


usage_snapshot = usage_activation_fixtures.usage_snapshot
