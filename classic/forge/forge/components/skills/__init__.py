"""Agent Skills (SKILL.md) support for Classic AutoGPT."""

from .skill_component import SkillComponent
from .skill_model import Skill, SkillConfiguration, SkillLoadLevel, SkillMetadata
from .skill_parser import SkillParseError, discover_skills

__all__ = [
    "SkillComponent",
    "SkillConfiguration",
    "Skill",
    "SkillLoadLevel",
    "SkillMetadata",
    "SkillParseError",
    "discover_skills",
]
