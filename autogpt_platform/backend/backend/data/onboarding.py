from typing import Any, List, Optional

import pydantic
from prisma import Json
from prisma.models import UserOnboarding
from prisma.types import UserOnboardingCreateInput, UserOnboardingUpdateInput


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
    if data.selectedAgentCreator:
        update["selectedAgentCreator"] = data.selectedAgentCreator
    if data.selectedAgentSlug:
        update["selectedAgentSlug"] = data.selectedAgentSlug
    if data.agentInput:
        update["agentInput"] = Json(data.agentInput)

    return await UserOnboarding.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id, **update},  # type: ignore
            "update": update,
        },
    )
