"""Copy for the copilot home's onboarding greeting.

Path A (they talked) gets a generated greeting — see ``intro.py``. Path B
is static: there is nothing to reflect back but the role the wizard asked
for. The card under the greeting asks for the dump, so the greeting itself
asks nothing. The suggested prompts under it are the generic set from
``intro.fallback_prompts``.
"""

from backend.data.onboarding_role import ROLE_LABELS, OnboardingRole

PATH_B_GREETING = (
    "I'm your Head of AI, and I build the team that does the work you don't "
    "want to. I haven't heard what your week looks like yet."
)


def path_b_greeting(role: OnboardingRole | None) -> str:
    # An "Other" role is free text that need not read as a role, so only the
    # wizard's own choices go into the sentence.
    if role is None or role.choice not in ROLE_LABELS:
        return PATH_B_GREETING
    return (
        "I'm your Head of AI, and I build the team that does the work you "
        f"don't want to. You told me your role is {role.label}, but I haven't "
        "heard what your week looks like yet."
    )
