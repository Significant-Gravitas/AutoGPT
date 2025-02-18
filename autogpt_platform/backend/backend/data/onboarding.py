from prisma.models import UserOnboarding
from prisma.types import UserOnboardingCreateInput, UserOnboardingUpdateInput


async def get_user_onboarding(user_id: str):
    return await UserOnboarding.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id},
            "update": {},
        },
    )


async def update_user_onboarding(
    user_id: str, data: UserOnboardingUpdateInput | UserOnboardingCreateInput
):
    return await UserOnboarding.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id, **data},
            "update": data,
        },
    )
