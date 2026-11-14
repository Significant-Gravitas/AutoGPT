"""Audited content-check allowances for exact occurrences in one skill version."""

from typing import Any

from pydantic import BaseModel, Field, ValidationError

from backend.util.json import SafeJson


class SkillContentApproval(BaseModel):
    file_path: str
    pattern_class: str
    match_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    count: int = Field(ge=1, le=100_000)
    approved_by_user_id: str


def parse_content_approvals(value: Any) -> list[SkillContentApproval]:
    if not isinstance(value, list):
        return []
    try:
        return [SkillContentApproval.model_validate(item) for item in value]
    except (ValidationError, TypeError, ValueError):
        return []


def content_approvals_json(approvals: list[SkillContentApproval]) -> SafeJson:
    return SafeJson([approval.model_dump() for approval in approvals])
