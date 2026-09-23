from typing import Protocol, TypedDict

from pydantic import BaseModel

from .presentation_defaults import MANAGED_PRESENTATION


class TemplatePresentation(TypedDict):
    avatarUrl: str | None
    bio: str | None
    identity: str
    tagline: str | None


class TemplateLike(Protocol):
    name: str
    isTemplate: bool
    avatarUrl: str | None
    bio: str | None
    identity: str
    tagline: str | None


class PresentationLike(Protocol):
    jobTitle: str | None
    tagline: str | None
    bio: str | None
    categories: list[str]


class PresentationBaseline(BaseModel):
    jobTitle: str | None
    tagline: str | None
    bio: str | None
    categories: list[str]


def template_presentation(row: TemplateLike) -> TemplatePresentation:
    values = TemplatePresentation(
        avatarUrl=row.avatarUrl, bio=row.bio, identity=row.identity, tagline=row.tagline
    )
    if row.isTemplate:
        for field, (previous, replacement) in MANAGED_PRESENTATION.get(
            row.name, {}
        ).items():
            if values[field] == previous:
                values[field] = replacement
    return values


def presentation_changes(
    current: PresentationLike,
    previous: PresentationLike,
    replacement: PresentationLike,
) -> dict[str, str | list[str] | None]:
    return {
        field: getattr(replacement, field)
        for field in ("jobTitle", "tagline", "bio", "categories")
        if getattr(current, field) == getattr(previous, field)
        and getattr(current, field) != getattr(replacement, field)
    }
