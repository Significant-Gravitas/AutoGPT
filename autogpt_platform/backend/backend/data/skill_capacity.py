import re

MAX_SKILLS_PER_EXPERT = 150
SKILL_ORIGIN_USER = "user"
SKILL_ORIGIN_MARKETPLACE = "marketplace"
SKILL_ORIGINS = frozenset({SKILL_ORIGIN_USER, SKILL_ORIGIN_MARKETPLACE})
SKILL_ORIGIN_LABELS = {
    SKILL_ORIGIN_USER: "saved",
    SKILL_ORIGIN_MARKETPLACE: "installed",
}
SKILL_ORIGIN_METADATA_KEY = "skill_origin"

_ROOT_PATH = re.compile(r"^(/skills|/experts/[^/]+/skills)/[^/]+/SKILL\.md$")


class SkillLimitError(Exception):
    """The skill owner's published roots have reached capacity."""


class SkillOwnedError(Exception):
    """A platform install would replace a skill saved by its owner."""


def normalize_skill_origin(value: object) -> str | None:
    return value if isinstance(value, str) and value in SKILL_ORIGINS else None


def skill_origin(metadata: dict | None) -> str | None:
    return normalize_skill_origin((metadata or {}).get(SKILL_ORIGIN_METADATA_KEY))


def skill_owner_folder(path: str) -> str | None:
    match = _ROOT_PATH.fullmatch(path)
    return match.group(1) if match else None
