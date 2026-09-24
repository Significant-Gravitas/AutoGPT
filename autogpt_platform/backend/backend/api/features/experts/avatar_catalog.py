from pathlib import Path
from typing import Literal

from pydantic import BaseModel

AvatarCategory = Literal[
    "marketing",
    "sales",
    "finance",
    "support",
    "operations",
    "research",
    "content",
    "development",
]


class AvatarPreset(BaseModel):
    id: AvatarCategory
    label: str
    hex: str
    color: str
    url: str


class AvatarCatalog(BaseModel):
    revision: int
    avatars: list[AvatarPreset]
    legacy: dict[str, str]


CATALOG = AvatarCatalog.model_validate_json(
    Path(__file__).with_suffix(".json").read_text()
)
PRESETS = {avatar.id: avatar for avatar in CATALOG.avatars}
DEFAULT_AVATAR_URL = PRESETS["content"].url


def resolve_avatar_url(url: str | None) -> str | None:
    if not url:
        return url
    if url in CATALOG.legacy:
        return CATALOG.legacy[url]
    if url.startswith("/avatars/notion/") and url.endswith(".svg"):
        return DEFAULT_AVATAR_URL
    return url
