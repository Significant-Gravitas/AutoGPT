"""
SkillComponent - Provides Agent Skills (SKILL.md) support for Classic AutoGPT.

This component implements the open Agent Skills standard, enabling modular,
progressively-loaded capabilities via markdown-based skill files.

See: https://platform.claude.com/docs/en/agents-and-tools/agent-skills/overview
"""

import logging
from typing import Iterator, Optional

from forge.agent.components import ConfigurableComponent
from forge.agent.protocols import CommandProvider, DirectiveProvider, MessageProvider
from forge.command import Command, command
from forge.llm.providers import ChatMessage
from forge.models.json_schema import JSONSchema

from .skill_model import Skill, SkillConfiguration, SkillLoadLevel
from .skill_parser import (
    SkillParseError,
    discover_skills,
    load_skill_content,
    load_skill_file,
)

logger = logging.getLogger(__name__)


class SkillComponent(
    DirectiveProvider,
    MessageProvider,
    CommandProvider,
    ConfigurableComponent[SkillConfiguration],
):
    """
    Component that provides Agent Skills support.

    Skills are modular capabilities defined by SKILL.md files. They use
    progressive disclosure to minimize token usage:
    - Level 1: Metadata always loaded (~100 tokens/skill)
    - Level 2: Full SKILL.md loaded when triggered (~500-5000 tokens)
    - Level 3: Additional files loaded on demand
    """

    config_class = SkillConfiguration

    def __init__(self, config: Optional[SkillConfiguration] = None):
        ConfigurableComponent.__init__(self, config)
        # All discovered skills (Level 1 - metadata only)
        self._available_skills: dict[str, Skill] = {}
        # Skills with full content loaded (Level 2+)
        self._loaded_skills: dict[str, Skill] = {}
        # Discover skills on initialization
        self._discover_skills()

    def _discover_skills(self) -> None:
        """Discover all skills in configured directories."""
        skills = discover_skills(self.config.skill_directories)
        self._available_skills = {skill.metadata.name: skill for skill in skills}
        logger.info(
            f"Discovered {len(self._available_skills)} skills: "
            f"{list(self._available_skills.keys())}"
        )

    # -------------------------------------------------------------------------
    # DirectiveProvider methods
    # -------------------------------------------------------------------------

    def get_resources(self) -> Iterator[str]:
        if self._available_skills:
            yield (
                "You have access to modular skills that provide specialized "
                "capabilities. Use `list_skills` to see available skills, "
                "and `load_skill` to activate one when needed."
            )

    def get_best_practices(self) -> Iterator[str]:
        if self._available_skills:
            yield (
                "Only load skills when you actually need their capabilities. "
                "Unload skills when you're done to conserve context space."
            )
            yield (
                "Before implementing complex functionality, check if a skill "
                "already provides the capability you need."
            )

    # -------------------------------------------------------------------------
    # MessageProvider methods
    # -------------------------------------------------------------------------

    def get_messages(self) -> Iterator[ChatMessage]:
        # Always provide skill catalog if skills are available
        if self._available_skills:
            catalog_lines = ["## Available Skills"]
            for name, skill in self._available_skills.items():
                loaded_marker = " [LOADED]" if name in self._loaded_skills else ""
                catalog_lines.append(
                    f"- **{name}**{loaded_marker}: {skill.metadata.description}"
                )
            yield ChatMessage.user("\n".join(catalog_lines))

        # Provide loaded skill content
        for name, skill in self._loaded_skills.items():
            if skill.load_level >= SkillLoadLevel.FULL_CONTENT and skill.content:
                skill_content = [f"## Skill: {name}"]
                skill_content.append("")
                skill_content.append(skill.content)

                # Show available additional files
                additional_files = skill.list_additional_files()
                if additional_files:
                    skill_content.append("")
                    skill_content.append("### Additional Files Available")
                    for f in additional_files:
                        loaded = " [loaded]" if f in skill.additional_files else ""
                        skill_content.append(f"- `{f}`{loaded}")

                yield ChatMessage.user("\n".join(skill_content))

    # -------------------------------------------------------------------------
    # CommandProvider methods
    # -------------------------------------------------------------------------

    def get_commands(self) -> Iterator[Command]:
        if self._available_skills:
            yield self.list_skills
            yield self.load_skill
            if self._loaded_skills:
                yield self.unload_skill
                yield self.read_skill_file

    @command(
        names=["list_skills"],
        description="List all available skills with their descriptions",
        parameters={},
    )
    def list_skills(self) -> str:
        """List all available skills with their metadata.

        Returns:
            str: Formatted list of available skills
        """
        if not self._available_skills:
            return "No skills available. Skills can be added to .autogpt/skills/"

        lines = ["Available Skills:", ""]
        for name, skill in self._available_skills.items():
            loaded = " [LOADED]" if name in self._loaded_skills else ""
            lines.append(f"**{name}**{loaded}")
            lines.append(f"  Description: {skill.metadata.description}")
            if skill.metadata.author:
                lines.append(f"  Author: {skill.metadata.author}")
            if skill.metadata.version:
                lines.append(f"  Version: {skill.metadata.version}")
            if skill.metadata.tags:
                lines.append(f"  Tags: {', '.join(skill.metadata.tags)}")
            lines.append("")

        return "\n".join(lines)

    @command(
        names=["load_skill"],
        description="Load a skill's full content to use its capabilities",
        parameters={
            "skill_name": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The name of the skill to load",
                required=True,
            )
        },
    )
    def load_skill(self, skill_name: str) -> str:
        """Load a skill's full content (Level 2).

        Args:
            skill_name: The name of the skill to load

        Returns:
            str: Status message indicating success or failure
        """
        if skill_name not in self._available_skills:
            available = ", ".join(self._available_skills.keys())
            return f"Skill '{skill_name}' not found. Available skills: {available}"

        if skill_name in self._loaded_skills:
            return f"Skill '{skill_name}' is already loaded."

        # Check if we've hit the max loaded skills limit
        if len(self._loaded_skills) >= self.config.max_loaded_skills:
            loaded = ", ".join(self._loaded_skills.keys())
            return (
                f"Cannot load skill '{skill_name}': maximum of "
                f"{self.config.max_loaded_skills} skills already loaded ({loaded}). "
                f"Unload a skill first using `unload_skill`."
            )

        skill = self._available_skills[skill_name]
        try:
            loaded_skill = load_skill_content(skill)
            self._loaded_skills[skill_name] = loaded_skill
            self._available_skills[skill_name] = loaded_skill

            return (
                f"Skill '{skill_name}' loaded successfully. "
                f"Its instructions are now available in the context."
            )
        except SkillParseError as e:
            logger.error(f"Failed to load skill '{skill_name}': {e}")
            return f"Failed to load skill '{skill_name}': {e}"

    @command(
        names=["unload_skill"],
        description="Unload a skill to free up context space",
        parameters={
            "skill_name": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The name of the skill to unload",
                required=True,
            )
        },
    )
    def unload_skill(self, skill_name: str) -> str:
        """Unload a skill to free context space.

        Args:
            skill_name: The name of the skill to unload

        Returns:
            str: Status message indicating success or failure
        """
        if skill_name not in self._loaded_skills:
            if skill_name in self._available_skills:
                return f"Skill '{skill_name}' is not currently loaded."
            return f"Skill '{skill_name}' not found."

        del self._loaded_skills[skill_name]

        # Reset the skill to metadata-only state
        skill = self._available_skills[skill_name]
        skill.content = None
        skill.additional_files.clear()
        skill.load_level = SkillLoadLevel.METADATA

        return f"Skill '{skill_name}' has been unloaded."

    @command(
        names=["read_skill_file"],
        description="Read an additional file from a loaded skill",
        parameters={
            "skill_name": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The name of the skill",
                required=True,
            ),
            "filename": JSONSchema(
                type=JSONSchema.Type.STRING,
                description="The name of the file to read",
                required=True,
            ),
        },
    )
    def read_skill_file(self, skill_name: str, filename: str) -> str:
        """Read an additional file from a skill (Level 3).

        Args:
            skill_name: The name of the skill
            filename: The name of the file to read

        Returns:
            str: The content of the file or an error message
        """
        if skill_name not in self._loaded_skills:
            if skill_name in self._available_skills:
                return (
                    f"Skill '{skill_name}' must be loaded first. "
                    f'Use `load_skill("{skill_name}")` to load it.'
                )
            return f"Skill '{skill_name}' not found."

        skill = self._loaded_skills[skill_name]

        # Check if already loaded
        if filename in skill.additional_files:
            return skill.additional_files[filename]

        # List available files for better error messages
        available_files = skill.list_additional_files()
        if filename not in available_files:
            if available_files:
                return (
                    f"File '{filename}' not found in skill '{skill_name}'. "
                    f"Available files: {', '.join(available_files)}"
                )
            return f"Skill '{skill_name}' has no additional files."

        try:
            content = load_skill_file(skill, filename)
            return content
        except (FileNotFoundError, ValueError, SkillParseError) as e:
            logger.error(f"Failed to read file '{filename}' from skill: {e}")
            return f"Failed to read file '{filename}': {e}"

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------

    def refresh_skills(self) -> None:
        """Re-discover skills from configured directories."""
        self._loaded_skills.clear()
        self._discover_skills()

    @property
    def available_skills(self) -> dict[str, Skill]:
        """Get all discovered skills."""
        return self._available_skills.copy()

    @property
    def loaded_skills(self) -> dict[str, Skill]:
        """Get currently loaded skills."""
        return self._loaded_skills.copy()
