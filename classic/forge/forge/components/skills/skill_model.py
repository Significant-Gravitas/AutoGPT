"""Data models for Agent Skills (SKILL.md) support."""

import re
from enum import IntEnum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class SkillLoadLevel(IntEnum):
    """Progressive disclosure levels for skill loading."""

    METADATA = 1  # Always loaded (~100 tokens/skill)
    FULL_CONTENT = 2  # Loaded when triggered (~500-5000 tokens)
    ADDITIONAL = 3  # On-demand files


class SkillMetadata(BaseModel):
    """Metadata parsed from SKILL.md frontmatter."""

    model_config = ConfigDict(populate_by_name=True)

    name: str = Field(
        ...,
        description="Skill name (lowercase, alphanumeric, hyphens, max 64 chars)",
        max_length=64,
    )
    description: str = Field(
        ...,
        description="Skill description (max 1024 chars)",
        max_length=1024,
    )
    license: Optional[str] = Field(
        default=None,
        description="License for the skill",
    )
    allowed_tools: Optional[list[str]] = Field(
        default=None,
        alias="allowed-tools",
        description="List of tools this skill is allowed to use",
    )
    author: Optional[str] = Field(
        default=None,
        description="Skill author",
    )
    version: Optional[str] = Field(
        default=None,
        description="Skill version",
    )
    tags: Optional[list[str]] = Field(
        default=None,
        description="Tags for categorizing the skill",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate skill name follows spec: lowercase, alphanumeric, hyphens."""
        if not re.match(r"^[a-z0-9-]+$", v):
            raise ValueError(
                "Skill name must be lowercase, alphanumeric, "
                "and may contain hyphens only"
            )
        return v


class Skill(BaseModel):
    """Represents a loaded skill with its metadata and content."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    path: Path = Field(
        ...,
        description="Path to the skill directory",
    )
    metadata: SkillMetadata = Field(
        ...,
        description="Parsed skill metadata from frontmatter",
    )
    content: Optional[str] = Field(
        default=None,
        description="Full SKILL.md content (body, excluding frontmatter)",
    )
    additional_files: dict[str, str] = Field(
        default_factory=dict,
        description="Additional files loaded on demand",
    )
    load_level: SkillLoadLevel = Field(
        default=SkillLoadLevel.METADATA,
        description="Current load level of the skill",
    )

    @property
    def skill_md_path(self) -> Path:
        """Path to the SKILL.md file."""
        return self.path / "SKILL.md"

    def list_additional_files(self) -> list[str]:
        """List available additional files in the skill directory."""
        files = []
        if self.path.exists():
            for item in self.path.iterdir():
                if item.is_file() and item.name != "SKILL.md":
                    files.append(item.name)
        return files


class SkillConfiguration(BaseModel):
    """Configuration for the SkillComponent."""

    skill_directories: list[Path] = Field(
        default_factory=lambda: [
            Path(".autogpt/skills"),
            Path.home() / ".autogpt/skills",
        ],
        description="Directories to search for skills",
    )
    max_loaded_skills: int = Field(
        default=5,
        description="Maximum number of skills that can be fully loaded at once",
        ge=1,
        le=20,
    )
