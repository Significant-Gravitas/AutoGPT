import asyncio
import logging
from datetime import datetime, timedelta, timezone
from typing import NamedTuple

from backend.api.features.executions.review.model import PendingHumanReviewModel
from backend.api.features.experts import experts_db
from backend.api.features.experts.models import Expert
from backend.api.features.library import db as library_db
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
from backend.util.timezone_utils import get_user_timezone_or_utc

from .compose import compose_home_dashboard
from .models import HomeDashboardResponse

logger = logging.getLogger(__name__)

_EXECUTION_LIMIT = 300
_REVIEW_LIMIT = 100


class HomeSourceData(NamedTuple):
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
    team_ids: list[str] | None = None,
) -> HomeDashboardResponse:
    now = datetime.now(timezone.utc)
    week_start = now - timedelta(days=7)
    data = await _load_home_source_data(
        user_id=user_id,
        organization_id=organization_id,
        team_ids=team_ids or [],
        now=now,
        week_start=week_start,
    )
    graph_ids = list(
        {execution.graph_id for execution in data.executions}
        | {review.graph_id for review in data.reviews}
    )
    library_refs = await library_db.get_library_agent_refs_by_graph_ids(
        user_id, graph_ids
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
    )


async def _load_home_source_data(
    *,
    user_id: str,
    organization_id: str | None,
    team_ids: list[str],
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
    schedules_task = asyncio.create_task(
        _get_schedules(
            user_id=user_id,
            organization_id=organization_id,
            team_ids=team_ids,
        )
    )
    credits_task = asyncio.create_task(
        _get_credits(user_id=user_id, organization_id=organization_id)
    )
    user_task = asyncio.create_task(user_db.get_user_by_id(user_id))
    experts = await experts_task
    executions = await executions_task
    reviews = await reviews_task
    cost_summary = await cost_summary_task
    schedules = await schedules_task
    credits = await credits_task
    user = await user_task
    return HomeSourceData(
        experts=experts,
        executions=executions,
        reviews=reviews,
        cost_summary=cost_summary,
        schedules=schedules,
        credits_balance=credits,
        timezone_name=get_user_timezone_or_utc(user.timezone if user else None),
    )


async def _get_schedules(
    *, user_id: str, organization_id: str | None, team_ids: list[str]
) -> list[GraphExecutionJobInfo | CopilotTurnJobInfo]:
    try:
        return await get_scheduler_client().get_execution_schedules(
            user_id=user_id,
            organization_id=organization_id,
            team_ids=team_ids,
        )
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
