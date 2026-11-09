"""Find the schedule a new one would duplicate (SECRT-2681).

Asked twice for the same job, AutoPilot used to set it up twice, and both
copies ran. The creating tools check here first and hand back the schedule
that already does the work instead of adding a second.

Only active schedules count: a paused one does not run, and a one-shot that
has fired is spent, so setting the same thing up again is a real request.
"""

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
            or not _is_same_owner(
                job, session, session.organization_id, session.team_id
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
    organization_id: str | None,
    team_id: str | None,
) -> GraphExecutionJobInfo | None:
    """A schedule of the same workflow version, on the same cron in the same
    timezone, with the same inputs and the same accounts. Inputs are compared
    with the workflow's defaults filled in, since a run fills them the same
    way. Its name is only a label, so a different one doesn't make it a
    different schedule."""
    jobs = await scheduler.get_execution_schedules(
        user_id=user_id, graph_id=graph_id, kind="graph"
    )
    wanted_accounts = _account_ids(credentials)
    wanted_inputs = {**input_defaults, **inputs}
    return next(
        (
            job
            for job in jobs
            if isinstance(job, GraphExecutionJobInfo)
            and _is_same_owner(job, session, organization_id, team_id)
            and job.graph_id == graph_id
            and job.graph_version == graph_version
            and _same_words(job.cron, cron)
            and job.timezone == timezone
            and {**input_defaults, **job.input_data} == wanted_inputs
            and _account_ids(job.input_credentials) == wanted_accounts
        ),
        None,
    )


def _is_same_owner(
    job: GraphExecutionJobInfo | CopilotTurnJobInfo,
    session: ChatSession,
    organization_id: str | None,
    team_id: str | None,
) -> bool:
    # Personal AutoPilot can see its experts' schedules, but an expert's
    # schedule is not a copy of the same work set up for the account.
    if not _is_in_session_scope(job, session) or job.expert_id != session.expert_id:
        return False
    # An expert's jobs are moved to her own private tenancy when they are
    # made, so the expert is what identifies them. Without one, the same work
    # set up in another organization or team is that tenant's own schedule.
    return session.expert_id is not None or (
        (job.organization_id or None, job.team_id or None)
        == (organization_id or None, team_id or None)
    )


def _same_words(a: str | None, b: str | None) -> bool:
    return a is not None and b is not None and a.split() == b.split()


def _account_ids(credentials: dict[str, CredentialsMetaInput]) -> dict[str, str]:
    return {field: account.id for field, account in credentials.items()}


def _pinned_ids(pins: CredentialPins) -> dict[str, str]:
    return {provider: pin.id for provider, pin in pins.items()}
