"""A scheduled run that fails the workflow's checks when it fires (#17245).

The schedule is kept and the run skipped, as for the scheduler's other skips
(a paused expert, a paywalled or banned owner): a skip can be undone, and a
delete can't. Its owner is told once per problem, through the Alert family,
and the schedule keeps what they were told (``RunsSkipped``), so one that
fires every minute doesn't tell them every minute. The next run that starts,
or an edit, clears it and withdraws the alert.

A schedule saved before schedules had ids (2025-11, #11374) has nothing to key
its alert by: it is kept and skipped with a warning, and its owner isn't told.
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING
from urllib.parse import quote

from pydantic import BaseModel

from backend.blocks import get_block
from backend.data.model import provider_label
from backend.executor.utils import (
    CRED_ERR_UNKNOWN_PREFIX,
    is_credential_validation_error_message,
)
from backend.notifications.alert_causes import ScheduleProblem, ScheduleSkippedCause
from backend.util.clients import get_database_manager_async_client, get_scheduler_client
from backend.util.exceptions import BlockDisabledError, GraphValidationError

if TYPE_CHECKING:
    from backend.executor.scheduler import GraphExecutionJobArgs

logger = logging.getLogger(__name__)


class RunsSkipped(BaseModel):
    """Why a schedule's runs are being skipped, as its owner was told."""

    problem: ScheduleProblem
    # Which fields fail and how, so another problem of the same kind is news.
    fingerprint: str
    since: datetime


class _Diagnosis(BaseModel):
    problem: ScheduleProblem
    fingerprint: str
    removed_key_ids: list[str] = []
    step_block_id: str | None = None


def alert_key(schedule_id: str) -> str:
    return f"schedule_skipped:{schedule_id}"


async def skip_invalid_run(
    args: "GraphExecutionJobArgs", error: GraphValidationError
) -> None:
    """The run failed the workflow's checks: a key, an input."""
    await _skip(args, _diagnose_validation(error))


async def skip_run_of_disabled_step(
    args: "GraphExecutionJobArgs", error: BlockDisabledError
) -> None:
    """The workflow's version this schedule runs uses a switched-off block."""
    await _skip(
        args,
        _Diagnosis(
            problem="step_disabled",
            fingerprint=_fingerprint("step_disabled", [error.block_id or ""]),
            step_block_id=error.block_id,
        ),
    )


async def note_run_started(args: "GraphExecutionJobArgs") -> None:
    """A run started, so what its schedule's owner was told is over. Best
    effort: a failure leaves the note on the schedule, and the next run that
    starts tries again."""
    if args.runs_skipped is None or args.schedule_id is None:
        return
    try:
        await get_database_manager_async_client().resolve_alert_condition(
            args.user_id, alert_key(args.schedule_id)
        )
        await get_scheduler_client().clear_runs_skipped(
            schedule_id=args.schedule_id, user_id=args.user_id
        )
    except Exception:
        logger.warning(
            f"Could not clear the skipped-runs notice of schedule "
            f"{args.schedule_id}; the next run that starts tries again",
            exc_info=True,
        )


async def withdraw_alert(user_id: str, schedule_id: str) -> None:
    """The schedule was edited or deleted, so an alert about it is out of
    date. Best effort: one left open is about a schedule that changed."""
    try:
        await get_database_manager_async_client().resolve_alert_condition(
            user_id, alert_key(schedule_id)
        )
    except Exception:
        logger.warning(
            f"Could not withdraw the skipped-runs alert of schedule {schedule_id}",
            exc_info=True,
        )


async def _skip(args: "GraphExecutionJobArgs", diagnosis: _Diagnosis) -> None:
    told = args.runs_skipped
    if told is not None and told.fingerprint == diagnosis.fingerprint:
        logger.info(
            f"Skipped a scheduled run of graph #{args.graph_id} (schedule "
            f"{args.schedule_id}): {diagnosis.problem}; its owner was told"
        )
        return
    logger.warning(
        f"Skipped a scheduled run of graph #{args.graph_id} (schedule "
        f"{args.schedule_id}): {diagnosis.problem}; the schedule is kept"
    )
    if args.schedule_id is None:
        return
    notice = RunsSkipped(
        problem=diagnosis.problem,
        fingerprint=diagnosis.fingerprint,
        since=told.since if told else datetime.now(timezone.utc),
    )
    try:
        await _raise_alert(args, args.schedule_id, diagnosis)
        recorded = await get_scheduler_client().mark_runs_skipped(
            user_id=args.user_id, fired=args, runs_skipped=notice
        )
    except Exception:
        # Not recorded, so the next run that fails tells them again; the
        # alert engine keeps that to one email.
        logger.warning(
            f"Could not tell the owner of schedule {args.schedule_id} why its "
            "run was skipped; the next run tries again",
            exc_info=True,
        )
        return
    if not recorded:
        # Edited or deleted since this run read it, so the alert is about
        # inputs it no longer has. A run of the new ones decides again.
        await withdraw_alert(args.user_id, args.schedule_id)


async def _raise_alert(
    args: "GraphExecutionJobArgs", schedule_id: str, diagnosis: _Diagnosis
) -> None:
    db = get_database_manager_async_client()
    graph = await db.get_graph_metadata(args.graph_id, args.graph_version)
    removed = next(
        (
            meta
            for meta in args.input_credentials.values()
            if meta.id in diagnosis.removed_key_ids
        ),
        None,
    )
    step = get_block(diagnosis.step_block_id) if diagnosis.step_block_id else None
    cause = ScheduleSkippedCause(
        cta_path=await _cta_path(args, schedule_id, diagnosis.problem),
        agent=graph.name if graph and graph.name else "Your workflow",
        problem=diagnosis.problem,
        schedule=args.agent_name,
        provider=_provider_name(removed.provider.value) if removed else None,
        key_title=removed.title if removed else None,
        step=step.display_name if step else None,
    )
    await db.raise_alert_condition(
        user_id=args.user_id,
        cause=cause.cause,
        cause_key=alert_key(schedule_id),
        data=cause.model_dump(mode="json"),
    )


async def _cta_path(
    args: "GraphExecutionJobArgs", schedule_id: str, problem: ScheduleProblem
) -> str:
    """The step is replaced in the builder; anything else starts from the
    schedule, which the Library lists with its workflow."""
    if problem == "step_disabled":
        return f"/build?flowID={quote(args.graph_id)}"
    library_agent_id = (
        await get_database_manager_async_client().get_library_agent_id_by_graph_id(
            args.user_id, args.graph_id
        )
    )
    if library_agent_id is None:
        return "/library"
    return (
        f"/library/agents/{quote(library_agent_id)}"
        f"?activeTab=scheduled&activeItem={quote(schedule_id)}"
    )


def _diagnose_validation(error: GraphValidationError) -> _Diagnosis:
    """A removed key first, as the error the issue names; then any other
    key the check refused; then the inputs."""
    failing = sorted(
        (node_id, field, message)
        for node_id, errors in error.node_errors.items()
        for field, message in errors.items()
    )
    removed = sorted(
        {
            message.removeprefix(CRED_ERR_UNKNOWN_PREFIX)
            for _, _, message in failing
            if message.startswith(CRED_ERR_UNKNOWN_PREFIX)
        }
    )
    problem: ScheduleProblem
    if removed:
        problem = "key_removed"
    elif any(is_credential_validation_error_message(m) for _, _, m in failing):
        problem = "key_unusable"
    else:
        problem = "inputs_invalid"
    # Fields, not messages: a key store that can't be read puts its own
    # error text in the message, which differs on every run.
    fields = [f"{node_id}/{field}" for node_id, field, _ in failing]
    return _Diagnosis(
        problem=problem,
        fingerprint=_fingerprint(problem, fields + removed),
        removed_key_ids=removed,
    )


def _fingerprint(problem: ScheduleProblem, parts: list[str]) -> str:
    text = json.dumps([problem, parts])
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _provider_name(provider: str) -> str:
    label = provider_label(provider)
    return label if label != provider else provider.replace("_", " ").title()
