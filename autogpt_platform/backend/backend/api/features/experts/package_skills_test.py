"""Tests for the one installer both ways an expert arrives from a package use."""

from unittest.mock import AsyncMock

import pytest
import pytest_mock

from backend.api.features.experts import experts_db, package_import, package_skills
from backend.api.features.experts.package_model import (
    ExpertManifest,
    ExpertPackage,
    PackagedIdentity,
    PackagedSkill,
)
from backend.copilot.tools.skills import ParsedSkill, SkillFile, SkillPackage

SKILL_MD = (
    "---\n"
    "name: research\n"
    "description: Digs things up.\n"
    "version: 2.1.0\n"
    "license: MIT\n"
    "compatibility: needs network\n"
    "allowed-tools: web_fetch bash\n"
    "triggers:\n"
    "  - research\n"
    "---\n"
    "\n"
    "# Research\n"
)


def _package() -> ExpertPackage:
    return ExpertPackage(
        manifest=ExpertManifest(
            identity=PackagedIdentity(name="Maria Ops"),
            skills=[PackagedSkill(slug="research", name="Research")],
        ),
        skills={
            "research": SkillPackage(
                skill_md=SKILL_MD,
                files=[SkillFile(relative_path="refs/API.md", content=b"# API\n")],
            )
        },
    )


@pytest.mark.asyncio
async def test_a_packaged_skill_keeps_its_whole_frontmatter(
    mocker: pytest_mock.MockerFixture,
):
    """Version, license, compatibility and allowed-tools are supported
    frontmatter that the upload and copy paths keep; arriving inside an
    expert must not strip them."""
    store = mocker.patch.object(
        package_skills,
        "store_user_skill",
        new_callable=AsyncMock,
        return_value=ParsedSkill(name="research", description="", body=""),
    )
    package = _package()

    failed = await package_skills.install_package_skills(
        "user-1", "expert-1", package, package.manifest.skills
    )

    assert failed == []
    kwargs = store.await_args.kwargs
    assert kwargs["expert_id"] == "expert-1"
    assert kwargs["name"] == "research"
    assert kwargs["description"] == "Digs things up."
    assert kwargs["triggers"] == ["research"]
    assert kwargs["version"] == "2.1.0"
    assert kwargs["extra"] == {
        "license": "MIT",
        "compatibility": "needs network",
        "allowed-tools": "web_fetch bash",
    }
    assert [f.relative_path for f in kwargs["files"]] == ["refs/API.md"]


def test_importing_and_hiring_share_the_installer():
    """Both paths write packaged skills through the same function, so a
    field kept here is kept for an import and for a hire alike."""
    assert (
        package_import.install_package_skills is package_skills.install_package_skills
    )
    assert experts_db.install_package_skills is package_skills.install_package_skills
