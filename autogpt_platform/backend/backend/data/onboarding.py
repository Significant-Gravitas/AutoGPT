from typing import Any, List, Optional

import pydantic
from prisma import Json
from prisma.models import (
    AgentGraph,
    AgentGraphExecution,
    StoreListingVersion,
    UserOnboarding,
)
from prisma.types import UserOnboardingCreateInput, UserOnboardingUpdateInput

from backend.server.v2.library.db import remove_agent_from_library
from backend.server.v2.store.db import get_store_agent_details


class UserOnboardingUpdate(pydantic.BaseModel):
    step: int
    usageReason: Optional[str] = None
    integrations: List[str] = pydantic.Field(default_factory=list)
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
    update: UserOnboardingCreateInput | UserOnboardingUpdateInput = {
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
            user_onboarding.selectedAgentSlug
            and user_onboarding.selectedAgentSlug != data.selectedAgentSlug
        ):
            store_agent = await get_store_agent_details(
                data.selectedAgentCreator, data.selectedAgentSlug
            )
            store_listing = await StoreListingVersion.prisma().find_unique_or_raise(
                where={"id": store_agent.store_listing_version_id}
            )
            agent_graph = await AgentGraph.prisma().find_unique_or_raise(
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
            # remove that agent from the user's library
            if execution_count == 0 and agent_graph.userId != user_id:
                await remove_agent_from_library(
                    user_id, store_listing.agentId, store_listing.agentVersion
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
