"""Find the schedule a new one would duplicate (SECRT-2681).

Asked twice for the same job, AutoPilot used to set it up twice, and both
copies ran. The creating tools check here first and hand back the schedule
that already does the work instead of adding a second.

Only active schedules count: a paused one does not run, and a one-shot that
has fired is spent, so setting the same thing up again is a real request.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime, timedelta
from typing import Any

from backend.copilot.credential_selection import CredentialPins
from backend.copilot.model import ChatSession
from backend.data.model import CredentialsMetaInput
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo

from .manage_schedules import _is_in_session_scope

# A one-shot's delay is counted from the moment of the call, so asking twice
# for "noon" a few minutes apart lands the two a few minutes apart. Within this
# window, the same message to the same chat is the same request.
ONE_SHOT_DUPLICATE_WINDOW = timedelta(minutes=10)

Tenant = tuple[str | None, str | None]


async def find_duplicate_followup(
    scheduler: Any,
    user_id: str,
    session: ChatSession,
    *,
    message: str,
    target_session_id: str | None,
    cron: str | None,
    run_at: datetime | None,
    timezone: str,
    credential_pins: CredentialPins,
) -> CopilotTurnJobInfo | None:
    """A follow-up that already sends this message to the same chat (or to a
    fresh one), on the same accounts, on the same cron in the same timezone,
    or once within ``ONE_SHOT_DUPLICATE_WINDOW`` of the same time. Spacing in
    the message or the cron doesn't make it a different one. A routine's jobs
    are left out: the routine is the record to change, through its own tool."""
    jobs = await scheduler.get_execution_schedules(user_id=user_id, kind="copilot_turn")
    for job in jobs:
        if (
            not isinstance(job, CopilotTurnJobInfo)
            or job.routine_id
            or not _is_same_scope(job, session)
            or (
                session.expert_id is None
                and _tenant(job.organization_id, job.team_id)
                != _tenant(session.organization_id, session.team_id)
            )
            or (job.session_id or None) != (target_session_id or None)
            or not _same_words(job.message, message)
            or _pinned_ids(job.credential_pins) != _pinned_ids(credential_pins)
        ):
            continue
        if cron is not None:
            if _same_words(job.cron, cron) and job.timezone == timezone:
                return job
        elif (
            job.cron is None
            and job.run_at is not None
            and run_at is not None
            and abs(job.run_at - run_at) <= ONE_SHOT_DUPLICATE_WINDOW
        ):
            return job
    return None


async def find_duplicate_graph_schedule(
    scheduler: Any,
    user_id: str,
    session: ChatSession,
    *,
    graph_id: str,
    graph_version: int,
    cron: str,
    timezone: str,
    inputs: dict[str, Any],
    input_defaults: dict[str, Any],
    credentials: dict[str, CredentialsMetaInput],
    tenant: Tenant,
    default_tenant: Callable[[], Awaitable[Tenant]],
) -> GraphExecutionJobInfo | None:
    """A schedule of the same workflow version, on the same cron in the same
    timezone and tenant, with the same inputs and the same accounts. Inputs
    are compared as a run sees them: defaults filled in, and an input left
    out the same as one given as nothing. Its name is only a label, so a
    different one doesn't make it a different schedule.

    A job saved before schedules carried a tenant runs in the owner's default
    team, so *default_tenant* is asked for that, and only when one turns up."""
    jobs = await scheduler.get_execution_schedules(
        user_id=user_id, graph_id=graph_id, kind="graph"
    )
    wanted_accounts = _account_ids(credentials)
    wanted_inputs = _as_run_sees(inputs, input_defaults)
    for job in jobs:
        if not (
            isinstance(job, GraphExecutionJobInfo)
            and _is_same_scope(job, session)
            and job.graph_id == graph_id
            and job.graph_version == graph_version
            and _same_words(job.cron, cron)
            and job.timezone == timezone
            and _as_run_sees(job.input_data, input_defaults) == wanted_inputs
            and _account_ids(job.input_credentials) == wanted_accounts
        ):
            continue
        if session.expert_id is not None:
            return job
        job_tenant = _tenant(job.organization_id, job.team_id)
        if job_tenant[0] is None:
            job_tenant = _tenant(*await default_tenant())
        if job_tenant == _tenant(*tenant):
            return job
    return None


def _is_same_scope(
    job: GraphExecutionJobInfo | CopilotTurnJobInfo, session: ChatSession
) -> bool:
    """Personal AutoPilot can see its experts' schedules, but an expert's
    schedule is not a copy of the same work set up for the account. An
    expert's jobs are moved to her private tenancy when they are made, so
    for those the expert is what identifies them; without one, the callers
    also compare the organization and team."""
    return _is_in_session_scope(job, session) and job.expert_id == session.expert_id


def _tenant(organization_id: str | None, team_id: str | None) -> Tenant:
    return organization_id or None, team_id or None


def _as_run_sees(inputs: dict[str, Any], defaults: dict[str, Any]) -> dict[str, Any]:
    merged = {**defaults, **inputs}
    return {
        name: value
        for name, value in merged.items()
        if value is not None or defaults.get(name) is not None
    }


def _same_words(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and a.split() == b.split()


def _account_ids(credentials: dict[str, CredentialsMetaInput]) -> dict[str, str]:
    return {field: account.id for field, account in credentials.items()}


def _pinned_ids(pins: CredentialPins) -> dict[str, str]:
    return {provider: pin.id for provider, pin in pins.items()}
