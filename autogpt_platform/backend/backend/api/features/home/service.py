import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import Any
from zoneinfo import ZoneInfo

from pydantic import BaseModel, ValidationError

from backend.api.features.executions.activity_gate import (
    hide_activity_summaries_if_disabled,
)
from backend.api.features.executions.review.model import PendingHumanReviewModel
from backend.api.features.experts import experts_db
from backend.api.features.experts.models import Expert
from backend.api.features.library import db as library_db
from backend.copilot.briefing.models import BriefingContent
from backend.data import briefing as briefing_db
from backend.data import execution as execution_db
from backend.data import human_review as review_db
from backend.data import user as user_db
from backend.data.credit import get_credit_model
from backend.data.execution import GraphExecutionMeta
from backend.data.execution_cost_summary import (
    UserExecutionCostSummary,
    get_user_cost_summary,
)
from backend.executor.scheduler import CopilotTurnJobInfo, GraphExecutionJobInfo
from backend.util.clients import get_scheduler_client
from backend.util.feature_flag import Flag, is_feature_enabled
from backend.util.timezone_utils import get_user_timezone_or_utc

from .briefing import without_summaries
from .compose import compose_home_dashboard
from .models import HomeDashboardResponse

logger = logging.getLogger(__name__)

_EXECUTION_LIMIT = 300
_REVIEW_LIMIT = 100


class HomeSourceData(BaseModel):
    experts: list[Expert]
    executions: list[GraphExecutionMeta]
    reviews: list[PendingHumanReviewModel]
    cost_summary: UserExecutionCostSummary
    schedules: list[GraphExecutionJobInfo | CopilotTurnJobInfo]
    credits_balance: int | None
    timezone_name: str


async def build_home_dashboard(
    *,
    user_id: str,
    organization_id: str | None = None,
) -> HomeDashboardResponse:
    now = datetime.now(timezone.utc)
    week_start = now - timedelta(days=7)
    data = await _load_home_source_data(
        user_id=user_id,
        organization_id=organization_id,
        now=now,
        week_start=week_start,
    )
    graph_ids = list(
        {execution.graph_id for execution in data.executions}
        | {review.graph_id for review in data.reviews}
    )
    # Both depend on the gathered data (graph ids / timezone) but not on each
    # other, so the briefing read costs no extra round-trip.
    library_refs, persisted_briefing = await asyncio.gather(
        library_db.get_library_agent_refs_by_graph_ids(user_id, graph_ids),
        _persisted_briefing(user_id=user_id, timezone_name=data.timezone_name),
    )

    return compose_home_dashboard(
        now=now,
        experts=data.experts,
        executions=data.executions,
        reviews=data.reviews,
        schedules=data.schedules,
        library_refs=library_refs,
        cost_summary=data.cost_summary,
        credits_balance=data.credits_balance,
        timezone_name=data.timezone_name,
        persisted_briefing=persisted_briefing,
    )


async def _persisted_briefing(
    *, user_id: str, timezone_name: str
) -> BriefingContent | None:
    """Today's stored briefing, or None when home should compute live instead.

    None covers every way the anchor can be missing: no row yet (a pre-9am
    signup), a job that failed, and — mirroring `briefings/routes.py` — stored
    content written by a different composer version that no longer validates.
    """
    briefing_date = datetime.now(ZoneInfo(timezone_name)).date()
    try:
        record = await briefing_db.get_briefing_for_date(user_id, briefing_date)
    except Exception:
        logger.warning("Home could not load the briefing for user %s", user_id[:12])
        return None
    if record is None:
        return None
    try:
        content = BriefingContent.model_validate(record.content)
    except ValidationError:
        logger.warning(
            "Briefing %s failed to validate against BriefingContent; "
            "home is composing its card live instead",
            record.id,
        )
        return None
    if await is_feature_enabled(Flag.AI_ACTIVITY_STATUS, user_id):
        return content
    return without_summaries(content)


async def _load_home_source_data(
    *,
    user_id: str,
    organization_id: str | None,
    now: datetime,
    week_start: datetime,
) -> HomeSourceData:
    experts_task = asyncio.create_task(experts_db.list_experts(user_id))
    executions_task = asyncio.create_task(
        execution_db.get_graph_executions(
            user_id=user_id,
            created_time_gte=week_start,
            created_time_lte=now,
            limit=_EXECUTION_LIMIT,
        )
    )
    reviews_task = asyncio.create_task(
        review_db.get_pending_reviews_for_user(user_id, page=1, page_size=_REVIEW_LIMIT)
    )
    cost_summary_task = asyncio.create_task(
        get_user_cost_summary(user_id=user_id, since=week_start, until=now)
    )
    schedules_task = asyncio.create_task(_get_schedules(user_id=user_id))
    credits_task = asyncio.create_task(
        _get_credits(user_id=user_id, organization_id=organization_id)
    )
    user_task = asyncio.create_task(user_db.get_user_by_id(user_id))
    # Gather rather than awaiting one by one: a failure in the first task would
    # otherwise leave the rest detached with their exceptions never retrieved.
    started: list[asyncio.Task[Any]] = [
        experts_task,
        executions_task,
        reviews_task,
        cost_summary_task,
        schedules_task,
        credits_task,
        user_task,
    ]
    await asyncio.gather(*started)
    user = user_task.result()
    return HomeSourceData(
        experts=experts_task.result(),
        executions=await hide_activity_summaries_if_disabled(
            executions_task.result(), user_id
        ),
        reviews=reviews_task.result(),
        cost_summary=cost_summary_task.result(),
        schedules=schedules_task.result(),
        credits_balance=credits_task.result(),
        timezone_name=get_user_timezone_or_utc(user.timezone if user else None),
    )


async def _get_schedules(
    *,
    user_id: str,
) -> list[GraphExecutionJobInfo | CopilotTurnJobInfo]:
    # Deliberately owner-scoped: executions, reviews and cost totals on this page
    # are all personal, so org/team schedules would surface upcoming runs whose
    # outcomes and approvals could never appear anywhere else on the dashboard.
    try:
        return await get_scheduler_client().get_execution_schedules(user_id=user_id)
    except Exception:
        logger.warning("Home could not load schedules for user %s", user_id[:12])
        return []


async def _get_credits(*, user_id: str, organization_id: str | None) -> int | None:
    try:
        model = await get_credit_model(user_id, organization_id)
        return await model.get_credits(user_id, organization_id)
    except Exception:
        logger.warning("Home could not load credits for user %s", user_id[:12])
        return None
