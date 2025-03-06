import re
from typing import Any, Optional

import prisma
import pydantic
from prisma import Json
from prisma.models import (
    AgentGraph,
    AgentGraphExecution,
    StoreListingVersion,
    UserOnboarding,
)
from prisma.types import UserOnboardingUpdateInput

from backend.server.v2.library.db import set_is_deleted_for_library_agent
from backend.server.v2.store.db import get_store_agent_details
from backend.server.v2.store.model import StoreAgentDetails

# Mapping from user reason id to categories to search for when choosing agent to show
REASON_MAPPING: dict[str, list[str]] = {
    "content_marketing": ["writing", "marketing", "creative"],
    "business_workflow_automation": ["business", "productivity"],
    "data_research": ["data", "research"],
    "ai_innovation": ["development", "research"],
    "personal_productivity": ["personal", "productivity"],
}


class UserOnboardingUpdate(pydantic.BaseModel):
    step: int
    usageReason: Optional[str] = None
    integrations: list[str] = pydantic.Field(default_factory=list)
    otherIntegrations: Optional[str] = None
    selectedAgentCreator: Optional[str] = None
    selectedAgentSlug: Optional[str] = None
    agentInput: Optional[dict[str, Any]] = None
    isCompleted: bool = False


async def get_user_onboarding(user_id: str):
    return await UserOnboarding.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id},  # type: ignore
            "update": {},
        },
    )


async def update_user_onboarding(user_id: str, data: UserOnboardingUpdate):
    # Get the user onboarding data
    user_onboarding = await get_user_onboarding(user_id)
    update: UserOnboardingUpdateInput = {
        "step": data.step,
        "isCompleted": data.isCompleted,
    }
    if data.usageReason:
        update["usageReason"] = data.usageReason
    if data.integrations:
        update["integrations"] = data.integrations
    if data.otherIntegrations:
        update["otherIntegrations"] = data.otherIntegrations
    if data.selectedAgentSlug and data.selectedAgentCreator:
        update["selectedAgentSlug"] = data.selectedAgentSlug
        update["selectedAgentCreator"] = data.selectedAgentCreator
        # Check if slug changes
        if (
            user_onboarding.selectedAgentCreator
            and user_onboarding.selectedAgentSlug
            and user_onboarding.selectedAgentSlug != data.selectedAgentSlug
        ):
            store_agent = await get_store_agent_details(
                user_onboarding.selectedAgentCreator, user_onboarding.selectedAgentSlug
            )
            store_listing = await StoreListingVersion.prisma().find_unique_or_raise(
                where={"id": store_agent.store_listing_version_id}
            )
            agent_graph = await AgentGraph.prisma().find_first(
                where={"id": store_listing.agentId, "version": store_listing.version}
            )
            execution_count = await AgentGraphExecution.prisma().count(
                where={
                    "userId": user_id,
                    "agentGraphId": store_listing.agentId,
                    "agentGraphVersion": store_listing.version,
                }
            )
            # If there was no execution and graph doesn't belong to the user,
            # mark the agent as deleted
            if execution_count == 0 and agent_graph and agent_graph.userId != user_id:
                await set_is_deleted_for_library_agent(
                    user_id, store_listing.agentId, store_listing.agentVersion, True
                )
    if data.agentInput:
        update["agentInput"] = Json(data.agentInput)

    return await UserOnboarding.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id, **update},  # type: ignore
            "update": update,
        },
    )


def clean_and_split(text: str) -> list[str]:
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


def calculate_points(
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


async def get_recommended_agents(user_id: str) -> list[StoreAgentDetails]:
    user_onboarding = await get_user_onboarding(user_id)
    categories = REASON_MAPPING.get(user_onboarding.usageReason or "", [])

    where_clause: dict[str, Any] = {}

    custom = clean_and_split((user_onboarding.usageReason or "").lower())

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

    agents = await prisma.models.StoreAgent.prisma().find_many(
        where=prisma.types.StoreAgentWhereInput(**where_clause),
        order=[
            {"featured": "desc"},
            {"runs": "desc"},
            {"rating": "desc"},
        ],
    )

    if len(agents) < 2:
        agents += await prisma.models.StoreAgent.prisma().find_many(
            where={
                "listing_id": {"not_in": [agent.listing_id for agent in agents]},
            },
            order=[
                {"featured": "desc"},
                {"runs": "desc"},
                {"rating": "desc"},
            ],
            take=2 - len(agents),
        )

    # Calculate points for the first 30 agents and choose the top 2
    agent_points = []
    for agent in agents[:50]:
        points = calculate_points(
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
