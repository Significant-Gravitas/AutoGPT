import inspect
from datetime import datetime, timezone

from prisma.enums import (
    DreamPassPhase,
    DreamPassRoute,
    DreamPassStatus,
    DreamPassTrigger,
)
from pydantic import TypeAdapter

from backend.copilot.dream.fetch import DreamInput
from backend.copilot.dream.schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DreamOperations,
    DreamPassUsage,
    PhaseUsage,
)
from backend.util.json import to_dict

from .db_manager import DatabaseManager, DatabaseManagerAsyncClient
from .dream_pass import (
    DreamPassApplied,
    DreamPassOperations,
    DreamPassRecord,
    DreamPassUpdate,
    DreamPhaseOutputs,
)


def test_async_client_exposes_chat_methods() -> None:
    assert hasattr(DatabaseManagerAsyncClient, "delete_chat_session")
    assert hasattr(DatabaseManagerAsyncClient, "set_turn_duration")
    assert hasattr(DatabaseManager, "update_chat_session_llm_route")
    assert hasattr(DatabaseManagerAsyncClient, "update_chat_session_llm_route")


def test_bot_analytics_methods_registered() -> None:
    for method in (
        "record_bot_event",
        "record_guild_joined",
        "mark_guild_left",
        "sync_guild_presence",
    ):
        assert hasattr(DatabaseManager, method)
        assert hasattr(DatabaseManagerAsyncClient, method)


def test_add_store_agent_rpc_request_schema_is_constructible() -> None:
    manager = DatabaseManager()
    manager._create_fastapi_endpoint(manager.add_store_agent_to_library)


_DREAM_PASS_METHODS = (
    "create_dream_pass",
    "update_dream_pass",
    "get_dream_pass",
    "get_dream_pass_for_user",
    "list_open_dream_passes",
    "list_dream_passes",
)


def test_dream_pass_methods_registered() -> None:
    for method in _DREAM_PASS_METHODS:
        assert hasattr(DatabaseManager, method)
        assert hasattr(DatabaseManagerAsyncClient, method)


def test_dream_pass_models_survive_the_rpc_round_trip() -> None:
    """The dream store in the scheduler and the batch executor reaches the
    table over this RPC: an update must arrive, and a record come back, as
    they left (JSON encoded one way, validated from the signature the other)."""
    now = datetime(2026, 9, 26, 3, 0, tzinfo=timezone.utc)
    bundle = DreamInput(
        user_id="u1",
        group_id="user_u1",
        window_start=now,
        window_end=now,
        known_fact_uuids={"f1", "f2"},
    )
    usage = DreamPassUsage(
        phases=[PhaseUsage(phase="consolidate", model="claude-sonnet-5")],
        discount_applied=0.5,
    )
    outputs = DreamPhaseOutputs(
        consolidate=ConsolidationOutput(
            facts=[ConsolidatedFact(content="Nick ships on Fridays", confidence=0.8)]
        )
    )
    operations = DreamPassOperations(
        planned=DreamOperations(summary_for_user="ok"),
        applied=DreamPassApplied(consolidated_count=1),
    )
    update = DreamPassUpdate(
        status=DreamPassStatus.SUBMITTED,
        phase=DreamPassPhase.CONSOLIDATE,
        input_bundle=bundle,
        phase_outputs=outputs,
        operations=operations,
        usage=usage,
        lease_expires_at=now,
    )
    manager = DatabaseManager()
    endpoint = manager._create_fastapi_endpoint(manager.update_dream_pass)
    body = inspect.signature(endpoint).parameters["body"].annotation
    sent = body.model_validate(to_dict({"pass_id": "p1", "update": update}))
    assert sent.update == update

    record = DreamPassRecord(
        id="p1",
        user_id="u1",
        expert_id=None,
        scope_key="u1",
        route=DreamPassRoute.ANTHROPIC_BATCH,
        trigger=DreamPassTrigger.EVAL,
        phase=DreamPassPhase.CONSOLIDATE,
        status=DreamPassStatus.SUBMITTED,
        skip_reason=None,
        cancel_generation=0,
        provider_batch_id="msgbatch_1",
        lease_token="tok",
        lease_expires_at=now,
        input_bundle=bundle,
        phase_outputs=outputs,
        operations=operations,
        usage=usage,
        window_start=now,
        window_end=now,
        error=None,
        created_at=now,
        started_at=now,
        submitted_at=now,
        applied_at=None,
        completed_at=None,
        updated_at=now,
    )
    returned = TypeAdapter(DreamPassRecord | None).validate_python(to_dict(record))
    assert returned == record
