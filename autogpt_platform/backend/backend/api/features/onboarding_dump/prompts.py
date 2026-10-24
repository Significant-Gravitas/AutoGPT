"""Copy for the copilot home's onboarding greeting.

Path A (they talked) gets a generated greeting — see ``intro.py``. Path B
is static: there is nothing to reflect back but the role the wizard asked
for. The card under the greeting asks for the dump, so the greeting itself
asks nothing. The suggested prompts under it are the generic set from
``intro.fallback_prompts``.
"""

PATH_B_GREETING = (
    "I'm your Head of AI, and I build the team that does the work you don't "
    "want to. I haven't heard what your week looks like yet."
)


def path_b_greeting(user_role: str | None) -> str:
    # An "Other" role is free text that need not read as a role, so only the
    # wizard's own choices go into the sentence.
    role = _WIZARD_ROLE_LABELS.get((user_role or "").strip())
    if not role:
        return PATH_B_GREETING
    return (
        "I'm your Head of AI, and I build the team that does the work you "
        f"don't want to. You told me your role is {role}, but I haven't "
        "heard what your week looks like yet."
    )


# The role step's ids (what the wizard stores) and the labels it shows.
_WIZARD_ROLE_LABELS = {
    "Founder/CEO": "Founder / CEO",
    "Operations": "Operations",
    "Sales/BD": "Sales / BD",
    "Marketing": "Marketing",
    "Product/PM": "Product / PM",
    "Engineering": "Engineering",
    "HR/People": "HR / People",
}
