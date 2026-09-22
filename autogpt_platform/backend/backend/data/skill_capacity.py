import re

MAX_SKILLS_PER_EXPERT = 150

_ROOT_PATH = re.compile(r"^(/skills|/experts/[^/]+/skills)/[^/]+/SKILL\.md$")


class SkillLimitError(Exception):
    """The skill owner's published roots have reached capacity."""


def skill_owner_folder(path: str) -> str | None:
    match = _ROOT_PATH.fullmatch(path)
    return match.group(1) if match else None
