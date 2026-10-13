"""The role picked in the onboarding wizard, kept as it was picked.

The wizard (frontend onboarding/steps/RoleStep.tsx) offers seven roles and
Other, which takes typed text. POST /onboarding/profile receives the option ID,
or for Other the typed text in its place, and the pick is kept on
`UserOnboarding`. The copy in the business understanding is no record of it:
AutoPilot's add_understanding tool, the brain dump extraction and Tally all
rewrite that field.

GTM segments on the pick, under the label people saw, in MailerLite (`role`,
`role_other`) and PostHog (`onboarding_role`, `onboarding_role_other`).
"""

from prisma.models import UserOnboarding
from pydantic import BaseModel

# Each option ID the wizard sends, and the label people saw.
ROLE_LABELS = {
    "Founder/CEO": "Founder / CEO",
    "Operations": "Operations",
    "Sales/BD": "Sales / BD",
    "Marketing": "Marketing",
    "Product/PM": "Product / PM",
    "Engineering": "Engineering",
    "HR/People": "HR / People",
}
OTHER = "Other"
OTHER_MAX_CHARS = 100


class OnboardingRole(BaseModel):
    # One of `ROLE_LABELS`' option IDs, or Other.
    choice: str
    # What was typed after picking Other; None for every other choice.
    other: str | None = None

    @property
    def label(self) -> str:
        return ROLE_LABELS.get(self.choice, OTHER)

    @classmethod
    def from_answer(cls, answer: str) -> "OnboardingRole":
        """The wizard sends the option ID, or for Other what was typed. Typing
        an option's exact ID under Other reads as picking that option."""
        text = answer.strip()
        if text in ROLE_LABELS:
            return cls(choice=text)
        return cls(choice=OTHER, other=text[:OTHER_MAX_CHARS].strip() or None)

    @classmethod
    def from_understanding(cls, user_role: str | None) -> "OnboardingRole | None":
        """The pick, for an account the wizard kept none for, from the copy in
        its business understanding. An exact option ID can only have come
        from the wizard. Anything else is Other's typed text or a rewrite,
        and the two can't be told apart, so it gives no pick."""
        if user_role in ROLE_LABELS:
            return cls(choice=user_role)
        return None


async def save_onboarding_role(user_id: str, role: OnboardingRole) -> None:
    await UserOnboarding.prisma().upsert(
        where={"userId": user_id},
        data={
            "create": {"userId": user_id, "role": role.choice, "roleOther": role.other},
            "update": {"role": role.choice, "roleOther": role.other},
        },
    )


async def get_onboarding_role(user_id: str) -> OnboardingRole | None:
    """The pick the wizard kept, or None for an account that never reached
    the end of it."""
    row = await UserOnboarding.prisma().find_unique(where={"userId": user_id})
    if row is None or not row.role:
        return None
    return OnboardingRole(choice=row.role, other=row.roleOther)
