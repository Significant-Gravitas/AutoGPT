"""What a task's activity line says beside its text: the connected account a
step used and the skill it followed.

Both ride in the line's ``data`` column (``account``, ``skill``) and are read
back here, so a row written before they existed, or with a value of the wrong
shape, simply has neither.
"""

from typing import Any

from pydantic import BaseModel, Field, ValidationError

ACCOUNT_KEY = "account"
SKILL_KEY = "skill"
LABEL_MAX_CHARS = 80


class TaskActivityAccount(BaseModel):
    """A connected account, as the person who connected it knows it. Never a
    secret, a credential id or a scope."""

    service: str = Field(description="The service's slug, e.g. google.")
    name: str = Field(description="The service's display name, e.g. Google.")
    label: str | None = Field(
        default=None, description="The account's email, handle or title."
    )


def account_from_data(data: dict[str, Any]) -> TaskActivityAccount | None:
    raw = data.get(ACCOUNT_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        return TaskActivityAccount.model_validate(raw)
    except ValidationError:
        return None


def skill_from_data(data: dict[str, Any]) -> str | None:
    skill = data.get(SKILL_KEY)
    if not isinstance(skill, str):
        return None
    return skill.strip() or None


def meta_data(account: TaskActivityAccount | None, skill: str | None) -> dict[str, Any]:
    """The ``data`` keys a tool call's line carries for its account and skill."""
    return {
        **({ACCOUNT_KEY: account.model_dump(exclude_none=True)} if account else {}),
        **({SKILL_KEY: skill} if skill else {}),
    }
