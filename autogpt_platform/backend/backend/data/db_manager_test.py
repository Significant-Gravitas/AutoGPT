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
from backend.copilot.dream.pass_record import (
    cancelled,
    cleanup_finished,
    expired,
    reaped,
)
from backend.copilot.dream.schemas import (
    ConsolidatedFact,
    ConsolidationOutput,
    DreamOperations,
    DreamPassUsage,
    PhaseUsage,
)
from backend.util.json import to_dict
from backend.util.service import EXPOSED_FLAG

from .db_manager import DatabaseManager, DatabaseManagerAsyncClient
from .dream_pass_models import (
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


def test_every_exposed_method_has_a_constructible_rpc_schema() -> None:
    """The service builds them all as it starts. One it can't build stops the
    database manager, and every run and call waiting on it hangs."""
    manager = DatabaseManager()
    failures = []
    for name, method in vars(DatabaseManager).items():
        if not getattr(method, EXPOSED_FLAG, False):
            continue
        try:
            manager._create_fastapi_endpoint(method)
        except Exception as error:
            failures.append(f"{name}: {type(error).__name__}: {error}")

    assert not failures, "\n".join(failures)


_DREAM_PASS_METHODS = (
    "create_dream_pass",
    "update_dream_pass",
    "get_dream_pass",
    "get_dream_pass_for_user",
    "list_open_dream_passes",
    "list_dream_passes",
    "list_expired_dream_passes",
    "list_dream_pass_cleanups",
    "delete_old_dream_passes",
    "erase_dream_passes",
    "add_dream_narrative",
    "blank_dream_narratives",
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
    for stop in (
        expired("stale", not_updated_since=now.replace(microsecond=123000)),
        cancelled("testing", owner_user_id="u1"),
        reaped("lapsed", not_updated_since=now),
        cleanup_finished(),
    ):
        sent = body.model_validate(to_dict({"pass_id": "p1", "update": stop}))
        assert sent.update == stop
        assert sent.update.clear == stop.clear != frozenset()

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
        cleanup_pending_at=now,
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


def test_the_reaper_and_retention_arguments_survive_the_rpc_round_trip() -> None:
    """The scheduler lists and deletes passes over this RPC: an aware cutoff
    and a limit must arrive as they left. The endpoints are built from the
    class's functions, as the service registers them."""
    cutoff = datetime(2026, 9, 26, 3, 0, 0, 123000, tzinfo=timezone.utc)
    manager = DatabaseManager()
    # Both scans also carry where the reaper's scan cycle stands.
    open_scan = {"after_lapse": cutoff, "after_id": "p1"}
    cleanup_scan = {"marked_through": cutoff, "after_mark": cutoff, "after_id": "p1"}
    for method, args in (
        (
            DatabaseManager.list_expired_dream_passes,
            {"expired_before": cutoff, "limit": 7, **open_scan},
        ),
        (
            DatabaseManager.list_dream_pass_cleanups,
            {"due_before": cutoff, "limit": 7, **cleanup_scan},
        ),
        (
            DatabaseManager.delete_old_dream_passes,
            {"created_before": cutoff, "limit": 7},
        ),
    ):
        endpoint = manager._create_fastapi_endpoint(method)
        body = inspect.signature(endpoint).parameters["body"].annotation
        sent = body.model_validate(to_dict(args))
        assert sent.model_dump() == args


def test_the_audience_consumer_can_read_the_opt_out() -> None:
    assert hasattr(DatabaseManager, "is_marketing_opted_out")
    assert hasattr(DatabaseManagerAsyncClient, "is_marketing_opted_out")


def test_the_executor_can_record_a_scheduled_run() -> None:
    """The copilot executor has no Prisma of its own. A run's outcome, the
    deepest payload here, crosses this RPC; a schema that cannot be built
    would only show as a warning in the executor's log."""
    assert hasattr(DatabaseManagerAsyncClient, "record_scheduled_run")
    manager = DatabaseManager()
    manager._create_fastapi_endpoint(manager.record_scheduled_run)


def test_the_executor_can_forget_a_deleted_scheduled_chat() -> None:
    """Otto's delete_schedule tool runs in the copilot executor, which has no
    Prisma: a direct call there fails, and its cleanup with it."""
    assert hasattr(DatabaseManagerAsyncClient, "forget_scheduled_chat")
    manager = DatabaseManager()
    manager._create_fastapi_endpoint(manager.forget_scheduled_chat)


def test_the_scheduler_can_post_a_note_into_a_chat() -> None:
    """The scheduler has no Prisma of its own; a skipped run's note goes
    through this RPC."""
    assert hasattr(DatabaseManagerAsyncClient, "append_session_message")
    manager = DatabaseManager()
    manager._create_fastapi_endpoint(manager.append_session_message)


def test_memory_schedule_methods_registered() -> None:
    """The scheduler and copilot-executor have no Prisma client; the
    schedule registry reaches its rows through these."""
    for method in (
        "get_scope_schedule",
        "claim_scope_schedule",
        "record_scope_jobs",
        "set_scope_state",
        "forget_scope_job",
        "record_scope_run",
        "list_user_scope_schedules",
        "list_scope_schedules",
        "existing_user_ids",
    ):
        assert hasattr(DatabaseManager, method)
        assert hasattr(DatabaseManagerAsyncClient, method)


def test_memory_schedule_rpc_request_schemas_are_constructible() -> None:
    manager = DatabaseManager()
    manager._create_fastapi_endpoint(manager.claim_scope_schedule)
    manager._create_fastapi_endpoint(manager.record_scope_run)
    manager._create_fastapi_endpoint(manager.existing_user_ids)
