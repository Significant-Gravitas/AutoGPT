"""Parsing utilities for SKILL.md files."""

import logging
import re
from pathlib import Path
from typing import Optional

import yaml
from pydantic import ValidationError

from .skill_model import Skill, SkillLoadLevel, SkillMetadata

logger = logging.getLogger(__name__)


class SkillParseError(Exception):
    """Raised when a SKILL.md file cannot be parsed."""

    pass


def _extract_frontmatter(content: str) -> tuple[Optional[str], str]:
    """
    Extract YAML frontmatter from markdown content.

    Args:
        content: Full markdown content with optional frontmatter

    Returns:
        Tuple of (frontmatter_yaml, body_content).
        frontmatter_yaml is None if no frontmatter found.
    """
    # Match frontmatter: starts with ---, ends with ---
    pattern = r"^---\s*\n(.*?)\n---\s*\n?(.*)"
    match = re.match(pattern, content, re.DOTALL)

    if match:
        return match.group(1), match.group(2)
    return None, content


def parse_skill_md(skill_path: Path) -> Skill:
    """
    Parse a SKILL.md file and return a Skill with metadata only (Level 1).

    Args:
        skill_path: Path to the skill directory containing SKILL.md

    Returns:
        Skill object with metadata loaded

    Raises:
        SkillParseError: If the SKILL.md file cannot be parsed
        FileNotFoundError: If SKILL.md doesn't exist
    """
    skill_md_file = skill_path / "SKILL.md"

    if not skill_md_file.exists():
        raise FileNotFoundError(f"SKILL.md not found at {skill_md_file}")

    try:
        content = skill_md_file.read_text(encoding="utf-8")
    except Exception as e:
        raise SkillParseError(f"Failed to read SKILL.md: {e}") from e

    frontmatter_yaml, body = _extract_frontmatter(content)

    if frontmatter_yaml is None:
        raise SkillParseError(
            f"SKILL.md at {skill_path} missing required YAML frontmatter"
        )

    try:
        frontmatter_data = yaml.safe_load(frontmatter_yaml)
    except yaml.YAMLError as e:
        raise SkillParseError(f"Invalid YAML frontmatter in {skill_path}: {e}") from e

    if not isinstance(frontmatter_data, dict):
        raise SkillParseError(
            f"YAML frontmatter in {skill_path} must be a mapping, "
            f"got {type(frontmatter_data).__name__}"
        )

    # Handle nested metadata field if present
    if "metadata" in frontmatter_data:
        metadata_section = frontmatter_data.pop("metadata")
        if isinstance(metadata_section, dict):
            # Merge metadata fields into top level (author, version, tags)
            for key in ["author", "version", "tags"]:
                if key in metadata_section and key not in frontmatter_data:
                    frontmatter_data[key] = metadata_section[key]

    try:
        metadata = SkillMetadata(**frontmatter_data)
    except ValidationError as e:
        raise SkillParseError(
            f"Invalid metadata in SKILL.md at {skill_path}: {e}"
        ) from e

    return Skill(
        path=skill_path,
        metadata=metadata,
        content=None,  # Content not loaded at Level 1
        load_level=SkillLoadLevel.METADATA,
    )


def load_skill_content(skill: Skill) -> Skill:
    """
    Load the full content of a skill's SKILL.md (Level 2).

    Args:
        skill: Skill object with metadata loaded

    Returns:
        Updated Skill object with full content loaded

    Raises:
        SkillParseError: If content cannot be loaded
    """
    if skill.load_level == SkillLoadLevel.FULL_CONTENT:
        return skill  # Already loaded

    skill_md_file = skill.path / "SKILL.md"

    try:
        content = skill_md_file.read_text(encoding="utf-8")
    except Exception as e:
        raise SkillParseError(f"Failed to read SKILL.md: {e}") from e

    _, body = _extract_frontmatter(content)

    skill.content = body.strip()
    skill.load_level = SkillLoadLevel.FULL_CONTENT

    return skill


def load_skill_file(skill: Skill, filename: str) -> str:
    """
    Load an additional file from a skill directory (Level 3).

    Args:
        skill: Skill object
        filename: Name of the file to load

    Returns:
        Content of the file

    Raises:
        FileNotFoundError: If file doesn't exist
        SkillParseError: If file cannot be read
        ValueError: If filename is invalid (attempts path traversal)
    """
    # Security: prevent path traversal
    if ".." in filename or filename.startswith("/"):
        raise ValueError(f"Invalid filename: {filename}")

    file_path = skill.path / filename

    # Verify the resolved path is still within the skill directory
    try:
        file_path.resolve().relative_to(skill.path.resolve())
    except ValueError:
        raise ValueError(f"Invalid filename: {filename}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {filename}")

    if not file_path.is_file():
        raise ValueError(f"Not a file: {filename}")

    try:
        content = file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise SkillParseError(f"Failed to read {filename}: {e}") from e

    # Cache the loaded file
    skill.additional_files[filename] = content
    skill.load_level = SkillLoadLevel.ADDITIONAL

    return content


def discover_skills(directories: list[Path]) -> list[Skill]:
    """
    Discover all skills in the given directories.

    Args:
        directories: List of directories to search for skills

    Returns:
        List of Skill objects with metadata loaded (Level 1)
    """
    skills: list[Skill] = []
    seen_names: set[str] = set()

    for directory in directories:
        if not directory.exists():
            logger.debug(f"Skill directory does not exist: {directory}")
            continue

        if not directory.is_dir():
            logger.warning(f"Skill path is not a directory: {directory}")
            continue

        for item in directory.iterdir():
            if not item.is_dir():
                continue

            skill_md = item / "SKILL.md"
            if not skill_md.exists():
                continue

            try:
                skill = parse_skill_md(item)

                # Skip duplicates (first occurrence wins)
                if skill.metadata.name in seen_names:
                    logger.warning(
                        f"Duplicate skill name '{skill.metadata.name}' "
                        f"found at {item}, skipping"
                    )
                    continue

                seen_names.add(skill.metadata.name)
                skills.append(skill)
                logger.debug(f"Discovered skill: {skill.metadata.name} at {item}")

            except (SkillParseError, FileNotFoundError) as e:
                logger.warning(f"Failed to parse skill at {item}: {e}")
                continue

    return skills
