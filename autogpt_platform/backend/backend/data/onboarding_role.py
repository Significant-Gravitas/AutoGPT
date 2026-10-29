"""The role picked in the onboarding wizard, kept as it was picked.

The wizard (frontend onboarding/steps/RoleStep.tsx) offers seven roles and
Other, which takes typed text. The pick is kept on `UserOnboarding` as soon as
the person moves on from the role step, from the wizard draft that step saves
(`from_wizard`, PATCH /onboarding), and again by POST /onboarding/profile at
the end of the wizard, which receives the option ID, or for Other the typed
text in its place. Most people who stop at the paywall never reach that end.
The copy in the business understanding is no record of it: AutoPilot's
add_understanding tool, the brain dump extraction and Tally all rewrite that
field.

GTM segments on the pick, under the label people saw, in MailerLite (`role`,
`role_other`) and PostHog (`onboarding_role`, `onboarding_role_other`).
"""

from prisma.models import UserOnboarding
from pydantic import BaseModel

from backend.data.onboarding_wizard import OnboardingWizardProgress

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
    def from_wizard(cls, progress: OnboardingWizardProgress) -> "OnboardingRole | None":
        """The pick a saved wizard draft holds once the person has moved on
        from the role step with it, read the way the wizard's profile submit
        reads it. None while they are still on that step, so a pick being
        changed, or Other's text being typed, is never taken."""
        if "role" not in progress.completedSteps or progress.currentStep == "role":
            return None
        answer = progress.otherRole if progress.role == OTHER else progress.role
        if not answer.strip():
            return None
        return cls.from_answer(answer)

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


def kept_role(row: UserOnboarding | None) -> OnboardingRole | None:
    """The pick kept on an onboarding row, or None if it has none."""
    if row is None or not row.role:
        return None
    return OnboardingRole(choice=row.role, other=row.roleOther)


async def get_onboarding_role(user_id: str) -> OnboardingRole | None:
    """The pick the wizard kept, or None for an account that never moved on
    from the role step."""
    return kept_role(
        await UserOnboarding.prisma().find_unique(where={"userId": user_id})
    )
