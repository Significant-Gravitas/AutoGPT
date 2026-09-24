from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

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


AvatarColor = Literal[
    "terracotta",
    "ochre",
    "sage",
    "coral",
    "slate",
    "olive",
    "stone",
    "charcoal",
    "rust",
    "rose",
    "seaglass",
    "pine",
    "denim",
    "sand",
    "bluegray",
    "redclay",
    "apricot",
    "moss",
    "chalk",
    "copper",
    "mist",
    "ink",
    "umber",
    "lagoon",
    "limestone",
    "fern",
]


class AvatarColorOption(BaseModel):
    id: AvatarColor
    label: str
    hex: str


class CategoryAvatar(BaseModel):
    url: str
    hex: str


class BuiltinAvatar(BaseModel):
    primary_category: AvatarCategory
    variants: dict[AvatarCategory, CategoryAvatar]
    previous_urls: list[str] = Field(default_factory=list)
    previous_url: str
    id: str
    name: str
    url: str
    color_id: AvatarColor


class AvatarPreset(BaseModel):
    id: AvatarCategory
    label: str
    hex: str
    color: str
    url: str
    color_id: AvatarColor


class AvatarCatalog(BaseModel):
    revision: int
    avatars: list[AvatarPreset]
    legacy: dict[str, str]
    colors: list[AvatarColorOption]
    identities: list[BuiltinAvatar]


CATALOG = AvatarCatalog.model_validate_json(
    Path(__file__).with_suffix(".json").read_text()
)
PRESETS = {avatar.id: avatar for avatar in CATALOG.avatars}
COLORS = {color.id: color for color in CATALOG.colors}
DEFAULT_AVATAR_URL = PRESETS["content"].url


def resolve_avatar_url(url: str | None) -> str | None:
    if not url:
        return url
    if url in CATALOG.legacy:
        return CATALOG.legacy[url]
    if url.startswith("/avatars/notion/") and url.endswith(".svg"):
        return DEFAULT_AVATAR_URL
    return url


def resolve_builtin_avatar_url(name: str, url: str | None) -> str | None:
    avatar = next((a for a in CATALOG.identities if a.name == name), None)
    if avatar and (
        url in [avatar.previous_url, *avatar.previous_urls]
        or resolve_avatar_url(url) == avatar.url
    ):
        return avatar.url
    return url
