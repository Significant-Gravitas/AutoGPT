import re
from typing import Any, Optional

import prisma
import pydantic
from prisma import Json
from prisma.enums import OnboardingStep
from prisma.models import UserOnboarding
from prisma.types import UserOnboardingUpdateInput

from backend.data.block import get_blocks
from backend.data.graph import GraphModel
from backend.data.model import CredentialsMetaInput
from backend.server.v2.store.model import StoreAgentDetails

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


class UserOnboardingUpdate(pydantic.BaseModel):
    completedSteps: Optional[list[OnboardingStep]] = None
    usageReason: Optional[str] = None
    integrations: Optional[list[str]] = None
    otherIntegrations: Optional[str] = None
    selectedStoreListingVersionId: Optional[str] = None
    agentInput: Optional[dict[str, Any]] = None


async def get_user_onboarding(user_id: str):
    return await UserOnboarding.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id},  # type: ignore
            "update": {},
        },
    )


async def update_user_onboarding(user_id: str, data: UserOnboardingUpdate):
    update: UserOnboardingUpdateInput = {}
    if data.completedSteps is not None:
        update["completedSteps"] = list(set(data.completedSteps))
    if data.usageReason is not None:
        update["usageReason"] = data.usageReason
    if data.integrations is not None:
        update["integrations"] = data.integrations
    if data.otherIntegrations is not None:
        update["otherIntegrations"] = data.otherIntegrations
    if data.selectedStoreListingVersionId is not None:
        update["selectedStoreListingVersionId"] = data.selectedStoreListingVersionId
    if data.agentInput is not None:
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


def get_credentials_blocks() -> dict[str, str]:
    # Returns a dictionary of block id to credentials field name
    creds: dict[str, str] = {}
    blocks = get_blocks()
    for id, block in blocks.items():
        for field_name, field_info in block().input_schema.model_fields.items():
            if field_info.annotation == CredentialsMetaInput:
                creds[id] = field_name
    return creds


CREDENTIALS_FIELDS: dict[str, str] = get_credentials_blocks()


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

    storeAgents = await prisma.models.StoreAgent.prisma().find_many(
        where=prisma.types.StoreAgentWhereInput(**where_clause),
        order=[
            {"featured": "desc"},
            {"runs": "desc"},
            {"rating": "desc"},
        ],
        take=100,
    )

    agentListings = await prisma.models.StoreListingVersion.prisma().find_many(
        where={
            "id": {"in": [agent.storeListingVersionId for agent in storeAgents]},
        },
        include={"Agent": True},
    )

    for listing in agentListings:
        agent = listing.Agent
        if agent is None:
            continue
        graph = GraphModel.from_db(agent)
        # Remove agents with empty input schema
        if not graph.input_schema:
            storeAgents = [
                a for a in storeAgents if a.storeListingVersionId != listing.id
            ]
            continue

        # Remove agents with empty credentials
        # Get nodes from this agent that have credentials
        nodes = await prisma.models.AgentNode.prisma().find_many(
            where={
                "agentGraphId": agent.id,
                "agentBlockId": {"in": list(CREDENTIALS_FIELDS.keys())},
            },
        )
        for node in nodes:
            block_id = node.agentBlockId
            field_name = CREDENTIALS_FIELDS[block_id]
            # If there are no credentials or they are empty, remove the agent
            # FIXME ignores default values
            if (
                field_name not in node.constantInput
                or node.constantInput[field_name] is None
            ):
                storeAgents = [
                    a for a in storeAgents if a.storeListingVersionId != listing.id
                ]
                break

    # If there are less than 2 agents, add more agents to the list
    if len(storeAgents) < 2:
        storeAgents += await prisma.models.StoreAgent.prisma().find_many(
            where={
                "listing_id": {"not_in": [agent.listing_id for agent in storeAgents]},
            },
            order=[
                {"featured": "desc"},
                {"runs": "desc"},
                {"rating": "desc"},
            ],
            take=2 - len(storeAgents),
        )

    # Calculate points for the first X agents and choose the top 2
    agent_points = []
    for agent in storeAgents[:POINTS_AGENT_COUNT]:
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


async def onboarding_enabled() -> bool:
    count = await prisma.models.StoreAgent.prisma().count(take=MIN_AGENT_COUNT + 1)

    # Onboading is enabled if there are at least 2 agents in the store
    return count >= MIN_AGENT_COUNT
