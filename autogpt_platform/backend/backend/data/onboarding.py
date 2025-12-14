import re
from datetime import datetime, timedelta, timezone
from typing import Any, Literal, Optional
from zoneinfo import ZoneInfo

import prisma
import pydantic
from prisma.enums import OnboardingStep
from prisma.models import UserOnboarding
from prisma.types import UserOnboardingCreateInput, UserOnboardingUpdateInput

from backend.data import execution as execution_db
from backend.data.credit import get_user_credit_model
from backend.data.notification_bus import (
    AsyncRedisNotificationEventBus,
    NotificationEvent,
)
from backend.data.user import get_user_by_id
from backend.server.model import OnboardingNotificationPayload
from backend.server.v2.store.model import StoreAgentDetails
from backend.util.cache import cached
from backend.util.json import SafeJson
from backend.util.timezone_utils import get_user_timezone_or_utc

# Mapping from user reason id to categories to search for when choosing agent to show
REASON_MAPPING: dict[str, list[str]] = {
    "content_marketing": ["writing", "marketing", "creative"],
    "business_workflow_automation": ["business", "productivity"],
    "data_research": ["data", "research"],
    "ai_innovation": ["development", "research"],
    "personal_productivity": ["personal", "productivity"],
}
POINTS_AGENT_COUNT = 50  # Number of agents to calculate points for
MIN_AGENT_COUNT = 2  # Minimum number of marketplace agents to enable onboarding

FrontendOnboardingStep = Literal[
    OnboardingStep.WELCOME,
    OnboardingStep.USAGE_REASON,
    OnboardingStep.INTEGRATIONS,
    OnboardingStep.AGENT_CHOICE,
    OnboardingStep.AGENT_NEW_RUN,
    OnboardingStep.AGENT_INPUT,
    OnboardingStep.CONGRATS,
    OnboardingStep.MARKETPLACE_VISIT,
    OnboardingStep.BUILDER_OPEN,
]


class UserOnboardingUpdate(pydantic.BaseModel):
    walletShown: Optional[bool] = None
    notified: Optional[list[OnboardingStep]] = None
    usageReason: Optional[str] = None
    integrations: Optional[list[str]] = None
    otherIntegrations: Optional[str] = None
    selectedStoreListingVersionId: Optional[str] = None
    agentInput: Optional[dict[str, Any]] = None
    onboardingAgentExecutionId: Optional[str] = None


async def get_user_onboarding(user_id: str):
    return await UserOnboarding.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": UserOnboardingCreateInput(userId=user_id),
            "update": {},
        },
    )


async def reset_user_onboarding(user_id: str):
    return await UserOnboarding.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": UserOnboardingCreateInput(userId=user_id),
            "update": {
                "completedSteps": [],
                "walletShown": False,
                "notified": [],
                "usageReason": None,
                "integrations": [],
                "otherIntegrations": None,
                "selectedStoreListingVersionId": None,
                "agentInput": prisma.Json({}),
                "onboardingAgentExecutionId": None,
                "agentRuns": 0,
                "lastRunAt": None,
                "consecutiveRunDays": 0,
            },
        },
    )


async def update_user_onboarding(user_id: str, data: UserOnboardingUpdate):
    update: UserOnboardingUpdateInput = {}
    onboarding = await get_user_onboarding(user_id)
    if data.walletShown:
        update["walletShown"] = data.walletShown
    if data.notified is not None:
        update["notified"] = list(set(data.notified + onboarding.notified))
    if data.usageReason is not None:
        update["usageReason"] = data.usageReason
    if data.integrations is not None:
        update["integrations"] = data.integrations
    if data.otherIntegrations is not None:
        update["otherIntegrations"] = data.otherIntegrations
    if data.selectedStoreListingVersionId is not None:
        update["selectedStoreListingVersionId"] = data.selectedStoreListingVersionId
    if data.agentInput is not None:
        update["agentInput"] = SafeJson(data.agentInput)
    if data.onboardingAgentExecutionId is not None:
        update["onboardingAgentExecutionId"] = data.onboardingAgentExecutionId

    return await UserOnboarding.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id, **update},
            "update": update,
        },
    )


async def _reward_user(user_id: str, onboarding: UserOnboarding, step: OnboardingStep):
    reward = 0
    match step:
        # Reward user when they clicked New Run during onboarding
        # This is because they need credits before scheduling a run (next step)
        # This is seen as a reward for the GET_RESULTS step in the wallet
        case OnboardingStep.AGENT_NEW_RUN:
            reward = 300
        case OnboardingStep.MARKETPLACE_VISIT:
            reward = 100
        case OnboardingStep.MARKETPLACE_ADD_AGENT:
            reward = 100
        case OnboardingStep.MARKETPLACE_RUN_AGENT:
            reward = 100
        case OnboardingStep.BUILDER_SAVE_AGENT:
            reward = 100
        case OnboardingStep.RE_RUN_AGENT:
            reward = 100
        case OnboardingStep.SCHEDULE_AGENT:
            reward = 100
        case OnboardingStep.RUN_AGENTS:
            reward = 300
        case OnboardingStep.RUN_3_DAYS:
            reward = 100
        case OnboardingStep.TRIGGER_WEBHOOK:
            reward = 100
        case OnboardingStep.RUN_14_DAYS:
            reward = 300
        case OnboardingStep.RUN_AGENTS_100:
            reward = 300

    if reward == 0:
        return

    # Skip if already rewarded
    if step in onboarding.rewardedFor:
        return

    user_credit_model = await get_user_credit_model(user_id)
    await user_credit_model.onboarding_reward(user_id, reward, step)
    await UserOnboarding.prisma().update(
        where={"userId": user_id},
        data={
            "rewardedFor": list(set(onboarding.rewardedFor + [step])),
        },
    )


async def complete_onboarding_step(user_id: str, step: OnboardingStep):
    """
    Completes the specified onboarding step for the user if not already completed.
    """
    onboarding = await get_user_onboarding(user_id)
    if step not in onboarding.completedSteps:
        await UserOnboarding.prisma().update(
            where={"userId": user_id},
            data={
                "completedSteps": list(set(onboarding.completedSteps + [step])),
            },
        )
        await _reward_user(user_id, onboarding, step)
        await _send_onboarding_notification(user_id, step)


async def _send_onboarding_notification(
    user_id: str, step: OnboardingStep | None, event: str = "step_completed"
):
    """
    Sends an onboarding notification to the user.
    """
    payload = OnboardingNotificationPayload(
        type="onboarding",
        event=event,
        step=step,
    )
    await AsyncRedisNotificationEventBus().publish(
        NotificationEvent(user_id=user_id, payload=payload)
    )


async def complete_re_run_agent(user_id: str, graph_id: str) -> None:
    """
    Complete RE_RUN_AGENT step when a user runs a graph they've run before.
    Keeps overhead low by only counting executions if the step is still pending.
    """
    onboarding = await get_user_onboarding(user_id)
    if OnboardingStep.RE_RUN_AGENT in onboarding.completedSteps:
        return

    # Includes current execution, so count > 1 means there was at least one prior run.
    previous_exec_count = await execution_db.get_graph_executions_count(
        user_id=user_id, graph_id=graph_id
    )
    if previous_exec_count > 1:
        await complete_onboarding_step(user_id, OnboardingStep.RE_RUN_AGENT)


def _clean_and_split(text: str) -> list[str]:
    """
    Removes all special characters from a string, truncates it to 100 characters,
    and splits it by whitespace and commas.

    Args:
        text (str): The input string.

    Returns:
        list[str]: A list of cleaned words.
    """
    # Remove all special characters (keep only alphanumeric and whitespace)
    cleaned_text = re.sub(r"[^a-zA-Z0-9\s,]", "", text.strip()[:100])

    # Split by whitespace and commas
    words = re.split(r"[\s,]+", cleaned_text)

    # Remove empty strings from the list
    words = [word.lower() for word in words if word]

    return words


def _calculate_points(
    agent, categories: list[str], custom: list[str], integrations: list[str]
) -> int:
    """
    Calculates the total points for an agent based on the specified criteria.

    Args:
        agent: The agent object.
        categories (list[str]): List of categories to match.
        words (list[str]): List of words to match in the description.

    Returns:
        int: Total points for the agent.
    """
    points = 0

    # 1. Category Matches
    matched_categories = sum(
        1 for category in categories if category in agent.categories
    )
    points += matched_categories * 100

    # 2. Description Word Matches
    description_words = agent.description.split()  # Split description into words
    matched_words = sum(1 for word in custom if word in description_words)
    points += matched_words * 100

    matched_words = sum(1 for word in integrations if word in description_words)
    points += matched_words * 50

    # 3. Featured Bonus
    if agent.featured:
        points += 50

    # 4. Rating Bonus
    points += agent.rating * 10

    # 5. Runs Bonus
    runs_points = min(agent.runs / 1000 * 100, 100)  # Cap at 100 points
    points += runs_points

    return int(points)


def _normalize_datetime(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _calculate_consecutive_run_days(
    last_run_at: datetime | None, current_consecutive_days: int, user_timezone: str
) -> tuple[datetime, int]:
    tz = ZoneInfo(user_timezone)
    local_now = datetime.now(tz)
    normalized_last_run = _normalize_datetime(last_run_at)

    if normalized_last_run is None:
        return local_now.astimezone(timezone.utc), 1

    last_run_local = normalized_last_run.astimezone(tz)
    last_run_date = last_run_local.date()
    today = local_now.date()

    if last_run_date == today:
        return local_now.astimezone(timezone.utc), current_consecutive_days

    if last_run_date == today - timedelta(days=1):
        return local_now.astimezone(timezone.utc), current_consecutive_days + 1

    return local_now.astimezone(timezone.utc), 1


def _get_run_milestone_steps(
    new_run_count: int, consecutive_days: int
) -> list[OnboardingStep]:
    milestones: list[OnboardingStep] = []
    if new_run_count >= 10:
        milestones.append(OnboardingStep.RUN_AGENTS)
    if new_run_count >= 100:
        milestones.append(OnboardingStep.RUN_AGENTS_100)
    if consecutive_days >= 3:
        milestones.append(OnboardingStep.RUN_3_DAYS)
    if consecutive_days >= 14:
        milestones.append(OnboardingStep.RUN_14_DAYS)
    return milestones


async def _get_user_timezone(user_id: str) -> str:
    user = await get_user_by_id(user_id)
    return get_user_timezone_or_utc(user.timezone if user else None)


async def increment_runs(user_id: str):
    """
    Increment a user's run counters and trigger any onboarding milestones.
    """
    user_timezone = await _get_user_timezone(user_id)
    onboarding = await get_user_onboarding(user_id)
    new_run_count = onboarding.agentRuns + 1
    last_run_at, consecutive_run_days = _calculate_consecutive_run_days(
        onboarding.lastRunAt, onboarding.consecutiveRunDays, user_timezone
    )

    await UserOnboarding.prisma().update(
        where={"userId": user_id},
        data={
            "agentRuns": {"increment": 1},
            "lastRunAt": last_run_at,
            "consecutiveRunDays": consecutive_run_days,
        },
    )

    milestones = _get_run_milestone_steps(new_run_count, consecutive_run_days)
    new_steps = [step for step in milestones if step not in onboarding.completedSteps]

    for step in new_steps:
        await complete_onboarding_step(user_id, step)
    # Send progress notification if no steps were completed, so client refetches onboarding state
    if not new_steps:
        await _send_onboarding_notification(user_id, None, event="increment_runs")


async def get_recommended_agents(user_id: str) -> list[StoreAgentDetails]:
    user_onboarding = await get_user_onboarding(user_id)
    categories = REASON_MAPPING.get(user_onboarding.usageReason or "", [])

    where_clause: dict[str, Any] = {}

    custom = _clean_and_split((user_onboarding.usageReason or "").lower())

    if categories:
        where_clause["OR"] = [
            {"categories": {"has": category}} for category in categories
        ]
    else:
        where_clause["OR"] = [
            {"description": {"contains": word, "mode": "insensitive"}}
            for word in custom
        ]

    where_clause["OR"] += [
        {"description": {"contains": word, "mode": "insensitive"}}
        for word in user_onboarding.integrations
    ]

    where_clause["is_available"] = True

    # Try to take only agents that are available and allowed for onboarding
    storeAgents = await prisma.models.StoreAgent.prisma().find_many(
        where={
            "is_available": True,
            "useForOnboarding": True,
        },
        order=[
            {"featured": "desc"},
            {"runs": "desc"},
            {"rating": "desc"},
        ],
        take=100,
    )

    # If not enough agents found, relax the useForOnboarding filter
    if len(storeAgents) < 2:
        storeAgents = await prisma.models.StoreAgent.prisma().find_many(
            where=prisma.types.StoreAgentWhereInput(**where_clause),
            order=[
                {"featured": "desc"},
                {"runs": "desc"},
                {"rating": "desc"},
            ],
            take=100,
        )

    # Calculate points for the first X agents and choose the top 2
    agent_points = []
    for agent in storeAgents[:POINTS_AGENT_COUNT]:
        points = _calculate_points(
            agent, categories, custom, user_onboarding.integrations
        )
        agent_points.append((agent, points))

    agent_points.sort(key=lambda x: x[1], reverse=True)
    recommended_agents = [agent for agent, _ in agent_points[:2]]

    return [
        StoreAgentDetails(
            store_listing_version_id=agent.storeListingVersionId,
            slug=agent.slug,
            agent_name=agent.agent_name,
            agent_video=agent.agent_video or "",
            agent_output_demo=agent.agent_output_demo or "",
            agent_image=agent.agent_image,
            creator=agent.creator_username,
            creator_avatar=agent.creator_avatar,
            sub_heading=agent.sub_heading,
            description=agent.description,
            categories=agent.categories,
            runs=agent.runs,
            rating=agent.rating,
            versions=agent.versions,
            last_updated=agent.updated_at,
        )
        for agent in recommended_agents
    ]


@cached(maxsize=1, ttl_seconds=300)  # Cache for 5 minutes since this rarely changes
async def onboarding_enabled() -> bool:
    """
    Check if onboarding should be enabled based on store agent count.
    Cached to prevent repeated slow database queries.
    """
    # Use a more efficient query that stops counting after finding enough agents
    count = await prisma.models.StoreAgent.prisma().count(take=MIN_AGENT_COUNT + 1)
    # Onboarding is enabled if there are at least 2 agents in the store
    return count >= MIN_AGENT_COUNT
