"""Tests for Agent Skills (SKILL.md) support."""

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from .skill_component import SkillComponent
from .skill_model import SkillConfiguration, SkillLoadLevel, SkillMetadata
from .skill_parser import (
    SkillParseError,
    _extract_frontmatter,
    discover_skills,
    load_skill_content,
    load_skill_file,
    parse_skill_md,
)


class TestSkillMetadata:
    """Tests for SkillMetadata model validation."""

    def test_valid_metadata(self):
        """Test creating valid metadata."""
        meta = SkillMetadata(
            name="test-skill",
            description="A test skill",
            author="Test Author",
            version="1.0.0",
            tags=["test", "example"],
        )
        assert meta.name == "test-skill"
        assert meta.description == "A test skill"

    def test_name_validation_lowercase(self):
        """Test that name must be lowercase."""
        with pytest.raises(ValueError, match="lowercase"):
            SkillMetadata(name="Test-Skill", description="A test")

    def test_name_validation_no_spaces(self):
        """Test that name cannot contain spaces."""
        with pytest.raises(ValueError, match="lowercase"):
            SkillMetadata(name="test skill", description="A test")

    def test_name_validation_alphanumeric_hyphens(self):
        """Test that name can only have alphanumeric and hyphens."""
        with pytest.raises(ValueError, match="alphanumeric"):
            SkillMetadata(name="test_skill", description="A test")

    def test_name_max_length(self):
        """Test that name cannot exceed 64 characters."""
        with pytest.raises(ValueError):
            SkillMetadata(name="a" * 65, description="A test")

    def test_description_max_length(self):
        """Test that description cannot exceed 1024 characters."""
        with pytest.raises(ValueError):
            SkillMetadata(name="test", description="a" * 1025)

    def test_allowed_tools_alias(self):
        """Test that allowed-tools YAML field maps to allowed_tools."""
        # Simulate what pydantic does when parsing YAML with the alias field name
        data = {
            "name": "test",
            "description": "test",
            "allowed-tools": ["tool1", "tool2"],
        }
        meta = SkillMetadata.model_validate(data)
        assert meta.allowed_tools == ["tool1", "tool2"]


class TestFrontmatterExtraction:
    """Tests for YAML frontmatter extraction."""

    def test_extract_valid_frontmatter(self):
        """Test extracting valid frontmatter."""
        content = """---
name: test-skill
description: A test skill
---
# Skill Content
Here is the body."""

        yaml_part, body = _extract_frontmatter(content)
        assert yaml_part is not None
        assert "name: test-skill" in yaml_part
        assert body.strip().startswith("# Skill Content")

    def test_no_frontmatter(self):
        """Test content without frontmatter."""
        content = "# Just markdown\nNo frontmatter here."

        yaml_part, body = _extract_frontmatter(content)
        assert yaml_part is None
        assert body == content

    def test_incomplete_frontmatter(self):
        """Test frontmatter without closing ---."""
        content = """---
name: test
# No closing delimiter
Body content"""

        yaml_part, _ = _extract_frontmatter(content)
        # Without the closing delimiter, it's not valid frontmatter
        assert yaml_part is None


class TestSkillParser:
    """Tests for SKILL.md parsing."""

    def test_parse_valid_skill(self):
        """Test parsing a valid SKILL.md file."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "test-skill"
            skill_dir.mkdir()

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(
                """---
name: test-skill
description: A test skill for validation
author: Test Author
version: 1.0.0
tags:
  - test
  - example
---
# Test Skill

This is the skill content.
"""
            )

            skill = parse_skill_md(skill_dir)
            assert skill.metadata.name == "test-skill"
            assert skill.metadata.description == "A test skill for validation"
            assert skill.metadata.author == "Test Author"
            assert skill.load_level == SkillLoadLevel.METADATA
            assert skill.content is None  # Not loaded at level 1

    def test_parse_missing_skill_md(self):
        """Test parsing when SKILL.md doesn't exist."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "empty-skill"
            skill_dir.mkdir()

            with pytest.raises(FileNotFoundError):
                parse_skill_md(skill_dir)

    def test_parse_missing_frontmatter(self):
        """Test parsing SKILL.md without frontmatter."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "no-frontmatter"
            skill_dir.mkdir()

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text("# Just content, no frontmatter")

            with pytest.raises(SkillParseError, match="frontmatter"):
                parse_skill_md(skill_dir)

    def test_parse_invalid_yaml(self):
        """Test parsing SKILL.md with invalid YAML."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "bad-yaml"
            skill_dir.mkdir()

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(
                """---
name: [invalid: yaml
description: broken
---
Content"""
            )

            with pytest.raises(SkillParseError, match="Invalid YAML"):
                parse_skill_md(skill_dir)

    def test_parse_missing_required_fields(self):
        """Test parsing SKILL.md with missing required fields."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "missing-fields"
            skill_dir.mkdir()

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(
                """---
name: test-skill
---
Missing description"""
            )

            with pytest.raises(SkillParseError, match="Invalid metadata"):
                parse_skill_md(skill_dir)

    def test_parse_nested_metadata(self):
        """Test parsing SKILL.md with nested metadata section."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "nested-meta"
            skill_dir.mkdir()

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(
                """---
name: nested-skill
description: Has nested metadata
metadata:
  author: Nested Author
  version: 2.0.0
---
Content"""
            )

            skill = parse_skill_md(skill_dir)
            assert skill.metadata.name == "nested-skill"
            assert skill.metadata.author == "Nested Author"
            assert skill.metadata.version == "2.0.0"


class TestSkillLoading:
    """Tests for progressive skill loading."""

    def test_load_skill_content(self):
        """Test loading full skill content (Level 2)."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "load-test"
            skill_dir.mkdir()

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(
                """---
name: load-test
description: Test loading
---
# Full Content

This should be loaded."""
            )

            skill = parse_skill_md(skill_dir)
            assert skill.content is None
            assert skill.load_level == SkillLoadLevel.METADATA

            loaded = load_skill_content(skill)
            assert loaded.content is not None
            assert "Full Content" in loaded.content
            assert loaded.load_level == SkillLoadLevel.FULL_CONTENT

    def test_load_skill_content_idempotent(self):
        """Test that loading content twice doesn't change anything."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "idempotent"
            skill_dir.mkdir()

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(
                """---
name: idempotent
description: Test
---
Content"""
            )

            skill = parse_skill_md(skill_dir)
            loaded1 = load_skill_content(skill)
            loaded2 = load_skill_content(loaded1)
            assert loaded1.content == loaded2.content

    def test_load_skill_file(self):
        """Test loading additional files (Level 3)."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "extra-files"
            skill_dir.mkdir()

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(
                """---
name: extra-files
description: Has extra files
---
Content"""
            )

            reference_md = skill_dir / "reference.md"
            reference_md.write_text("# Reference\nExtra content here.")

            skill = parse_skill_md(skill_dir)
            content = load_skill_file(skill, "reference.md")
            assert "Reference" in content
            assert "reference.md" in skill.additional_files

    def test_load_skill_file_path_traversal(self):
        """Test that path traversal is prevented."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "traversal-test"
            skill_dir.mkdir()

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(
                """---
name: traversal-test
description: Test
---
Content"""
            )

            skill = parse_skill_md(skill_dir)

            with pytest.raises(ValueError, match="Invalid filename"):
                load_skill_file(skill, "../../../etc/passwd")

            with pytest.raises(ValueError, match="Invalid filename"):
                load_skill_file(skill, "/etc/passwd")

    def test_list_additional_files(self):
        """Test listing additional files in skill directory."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "list-files"
            skill_dir.mkdir()

            skill_md = skill_dir / "SKILL.md"
            skill_md.write_text(
                """---
name: list-files
description: Test
---
Content"""
            )

            (skill_dir / "file1.md").write_text("File 1")
            (skill_dir / "file2.txt").write_text("File 2")

            skill = parse_skill_md(skill_dir)
            files = skill.list_additional_files()

            assert "file1.md" in files
            assert "file2.txt" in files
            assert "SKILL.md" not in files  # Excluded


class TestSkillDiscovery:
    """Tests for skill discovery."""

    def test_discover_skills(self):
        """Test discovering skills in directories."""
        with TemporaryDirectory() as tmp_dir:
            skills_dir = Path(tmp_dir) / "skills"
            skills_dir.mkdir()

            # Create two valid skills
            for name in ["skill-a", "skill-b"]:
                skill_dir = skills_dir / name
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text(
                    f"""---
name: {name}
description: Skill {name}
---
Content for {name}"""
                )

            # Create a directory without SKILL.md (should be ignored)
            (skills_dir / "not-a-skill").mkdir()

            skills = discover_skills([skills_dir])
            names = [s.metadata.name for s in skills]

            assert len(skills) == 2
            assert "skill-a" in names
            assert "skill-b" in names

    def test_discover_skills_duplicates(self):
        """Test that duplicate skill names are handled (first wins)."""
        with TemporaryDirectory() as tmp_dir:
            dir1 = Path(tmp_dir) / "dir1"
            dir2 = Path(tmp_dir) / "dir2"
            dir1.mkdir()
            dir2.mkdir()

            for d, desc in [(dir1, "First"), (dir2, "Second")]:
                skill_dir = d / "duplicate-skill"
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text(
                    f"""---
name: duplicate-skill
description: {desc} skill
---
Content"""
                )

            skills = discover_skills([dir1, dir2])
            assert len(skills) == 1
            assert skills[0].metadata.description == "First skill"

    def test_discover_skills_nonexistent_directory(self):
        """Test that nonexistent directories are handled gracefully."""
        skills = discover_skills([Path("/nonexistent/path")])
        assert skills == []


class TestSkillComponent:
    """Tests for the SkillComponent."""

    def test_component_initialization(self):
        """Test component initializes correctly."""
        with TemporaryDirectory() as tmp_dir:
            config = SkillConfiguration(
                skill_directories=[Path(tmp_dir)],
                max_loaded_skills=3,
            )
            component = SkillComponent(config)
            assert component._available_skills == {}
            assert component._loaded_skills == {}

    def test_list_skills_command(self):
        """Test the list_skills command."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "test-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                """---
name: test-skill
description: A test skill
author: Test Author
---
Content"""
            )

            config = SkillConfiguration(skill_directories=[Path(tmp_dir)])
            component = SkillComponent(config)

            result = component.list_skills()
            assert "test-skill" in result
            assert "A test skill" in result
            assert "Test Author" in result

    def test_load_skill_command(self):
        """Test the load_skill command."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "loadable"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                """---
name: loadable
description: A loadable skill
---
# Instructions

Use this skill for testing."""
            )

            config = SkillConfiguration(skill_directories=[Path(tmp_dir)])
            component = SkillComponent(config)

            result = component.load_skill("loadable")
            assert "loaded successfully" in result
            assert "loadable" in component._loaded_skills

    def test_load_skill_max_limit(self):
        """Test that max_loaded_skills limit is enforced."""
        with TemporaryDirectory() as tmp_dir:
            for i in range(3):
                skill_dir = Path(tmp_dir) / f"skill-{i}"
                skill_dir.mkdir()
                (skill_dir / "SKILL.md").write_text(
                    f"""---
name: skill-{i}
description: Skill {i}
---
Content"""
                )

            config = SkillConfiguration(
                skill_directories=[Path(tmp_dir)],
                max_loaded_skills=2,
            )
            component = SkillComponent(config)

            component.load_skill("skill-0")
            component.load_skill("skill-1")
            result = component.load_skill("skill-2")

            assert "Cannot load skill" in result
            assert "maximum" in result
            assert len(component._loaded_skills) == 2

    def test_unload_skill_command(self):
        """Test the unload_skill command."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "unloadable"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                """---
name: unloadable
description: Test
---
Content"""
            )

            config = SkillConfiguration(skill_directories=[Path(tmp_dir)])
            component = SkillComponent(config)

            component.load_skill("unloadable")
            assert "unloadable" in component._loaded_skills

            result = component.unload_skill("unloadable")
            assert "unloaded" in result
            assert "unloadable" not in component._loaded_skills

    def test_read_skill_file_command(self):
        """Test the read_skill_file command."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "with-files"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                """---
name: with-files
description: Has extra files
---
Main content"""
            )
            (skill_dir / "extra.md").write_text("Extra file content")

            config = SkillConfiguration(skill_directories=[Path(tmp_dir)])
            component = SkillComponent(config)

            # Must load skill first
            result = component.read_skill_file("with-files", "extra.md")
            assert "must be loaded first" in result

            component.load_skill("with-files")
            result = component.read_skill_file("with-files", "extra.md")
            assert "Extra file content" in result

    def test_get_messages_includes_catalog(self):
        """Test that get_messages includes skill catalog."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "catalog-test"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                """---
name: catalog-test
description: Test skill
---
Content"""
            )

            config = SkillConfiguration(skill_directories=[Path(tmp_dir)])
            component = SkillComponent(config)

            messages = list(component.get_messages())
            assert len(messages) >= 1
            assert "Available Skills" in messages[0].content
            assert "catalog-test" in messages[0].content

    def test_get_messages_includes_loaded_content(self):
        """Test that get_messages includes loaded skill content."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "content-test"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                """---
name: content-test
description: Test skill
---
# Skill Instructions

Use this for testing purposes."""
            )

            config = SkillConfiguration(skill_directories=[Path(tmp_dir)])
            component = SkillComponent(config)
            component.load_skill("content-test")

            messages = list(component.get_messages())
            # First message is catalog, second should be loaded skill content
            assert len(messages) >= 2
            full_content = "\n".join(m.content for m in messages)
            assert "Skill Instructions" in full_content

    def test_get_commands(self):
        """Test that get_commands returns appropriate commands."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "cmd-test"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                """---
name: cmd-test
description: Test
---
Content"""
            )

            config = SkillConfiguration(skill_directories=[Path(tmp_dir)])
            component = SkillComponent(config)

            commands = list(component.get_commands())
            cmd_names = [c.names[0] for c in commands]

            assert "list_skills" in cmd_names
            assert "load_skill" in cmd_names
            # unload_skill and read_skill_file only when skills are loaded
            assert "unload_skill" not in cmd_names

            component.load_skill("cmd-test")
            commands = list(component.get_commands())
            cmd_names = [c.names[0] for c in commands]
            assert "unload_skill" in cmd_names
            assert "read_skill_file" in cmd_names

    def test_get_resources(self):
        """Test that get_resources returns skill resource info."""
        with TemporaryDirectory() as tmp_dir:
            skill_dir = Path(tmp_dir) / "resource-test"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                """---
name: resource-test
description: Test
---
Content"""
            )

            config = SkillConfiguration(skill_directories=[Path(tmp_dir)])
            component = SkillComponent(config)

            resources = list(component.get_resources())
            assert len(resources) >= 1
            assert any("skill" in r.lower() for r in resources)

    def test_refresh_skills(self):
        """Test the refresh_skills method."""
        with TemporaryDirectory() as tmp_dir:
            config = SkillConfiguration(skill_directories=[Path(tmp_dir)])
            component = SkillComponent(config)

            # Initially empty
            assert len(component._available_skills) == 0

            # Add a skill
            skill_dir = Path(tmp_dir) / "new-skill"
            skill_dir.mkdir()
            (skill_dir / "SKILL.md").write_text(
                """---
name: new-skill
description: New
---
Content"""
            )

            # Refresh
            component.refresh_skills()
            assert len(component._available_skills) == 1
            assert "new-skill" in component._available_skills
