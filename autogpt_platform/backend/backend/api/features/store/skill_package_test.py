"""The marketplace round trip for a whole skill package.

Publishing snapshots the creator's tree into the version, installing writes it
back, and the copilot then finds the files where its own read tool looks. The
fixture is a real published package rather than a hand-made one, because the
shapes that break this are the ones nobody would invent: a nested directory, a
licence file at the root, and a script that has to stay executable.
"""

import hashlib
import os
from pathlib import Path
from unittest.mock import AsyncMock, patch

import prisma.models
import pytest

from backend.copilot.context import set_execution_context
from backend.copilot.model import ChatSession
from backend.copilot.tools.skills import (
    ReadSkillResponse,
    ReadSkillTool,
    SkillFile,
    list_user_skill_files,
    parse_skill_markdown,
    store_user_skill,
)
from backend.copilot.tools.workspace_files import (
    ReadWorkspaceFileTool,
    WorkspaceFileContentResponse,
)
from backend.util.test import SpinTestServer

from . import skill_db, skill_model, skill_submission_db

FIXTURE_DIR = (
    Path(__file__).parents[4] / "test" / "fixtures" / "skills" / "webapp-testing"
)
SLUG = "webapp-testing"


def _fixture_files() -> list[SkillFile]:
    """The vendored package's siblings, with the executable bit git preserved.

    ``README.md`` is ours, naming the upstream commit; it is not part of the
    package.
    """
    return [
        SkillFile(
            relative_path=str(path.relative_to(FIXTURE_DIR)),
            content=path.read_bytes(),
            is_executable=bool(path.stat().st_mode & 0o111),
        )
        for path in sorted(FIXTURE_DIR.rglob("*"))
        if path.is_file() and path.name not in {"SKILL.md", "README.md"}
    ]


def test_the_fixture_still_carries_one_executable_script():
    """A checkout that lost the mode bit would leave every assertion below
    passing while proving nothing about executables."""
    executable = [f.relative_path for f in _fixture_files() if f.is_executable]

    assert executable == ["scripts/with_server.py"]


async def _store_library_package(user_id: str, files: list[SkillFile]) -> None:
    parsed = parse_skill_markdown(
        (FIXTURE_DIR / "SKILL.md").read_text(encoding="utf-8"), fallback_name=SLUG
    )
    assert parsed is not None
    await store_user_skill(
        user_id,
        name=SLUG,
        description=parsed.description,
        body=parsed.body,
        triggers=list(parsed.triggers),
        extra=parsed.extra,
        files=files,
    )


async def _publish(user_id: str, reviewer_id: str) -> skill_model.SkillSubmission:
    submission = await skill_submission_db.submit_skill(
        user_id,
        skill_model.SkillSubmissionRequest(skill_name=SLUG, categories=["content"]),
    )
    return await skill_submission_db.review_skill_submission(
        submission.skill_listing_version_id,
        is_approved=True,
        reviewer_id=reviewer_id,
        comments="ok",
    )


@pytest.fixture
async def creator(setup_test_user, server: SpinTestServer):
    await _drop_our_listings()
    await prisma.models.Profile.prisma().upsert(
        where={"userId": setup_test_user},
        data={
            "create": {
                "userId": setup_test_user,
                "username": "package-creator",
                "name": "Package Creator",
                "description": "",
                "links": [],
            },
            "update": {},
        },
    )
    yield setup_test_user
    # Left behind, these rows fail the emptiness guard in the suites that sort
    # after this one.
    await _drop_our_listings()


async def _drop_our_listings() -> None:
    """By slug, not wholesale: this table is shared with every other checkout
    here, where the starter listings are real rows."""
    await prisma.models.SkillListingVersion.prisma().delete_many(
        where={"SkillListing": {"is": {"slug": {"in": [SLUG]}}}}
    )
    await prisma.models.SkillListing.prisma().delete_many(
        where={"slug": {"in": [SLUG]}}
    )


@pytest.fixture
async def expert(creator: str) -> str:
    """A real row, because the workspace scope an expert session runs under is
    resolved from it and fails closed when it is missing."""
    row = await prisma.models.Expert.prisma().create(
        data={
            "ownerUserId": creator,
            "name": "Nova",
            "role": "",
            "identity": "I'm Nova, raised by you.",
        }
    )
    return row.id


@pytest.fixture(autouse=True)
def skills_enabled():
    with patch(
        "backend.copilot.tools.skills.is_skills_feature_enabled",
        new=AsyncMock(return_value=True),
    ):
        yield
    # The context var outlives the test and would scope the next one's reads.
    set_execution_context(None, None)


async def test_a_published_package_survives_install_into_an_expert(
    creator: str, expert: str, setup_admin_user: str
):
    files = _fixture_files()
    await _store_library_package(creator, files)

    await _publish(creator, setup_admin_user)
    await skill_db.install_marketplace_skill(creator, SLUG, expert_id=expert)

    session = ChatSession.new(creator, dry_run=False, expert_id=expert)
    # The workspace scope comes from the executing turn, never from a tool
    # argument, so the read tools see nothing without one.
    set_execution_context(creator, session)
    read = await ReadSkillTool()._execute(user_id=creator, session=session, name=SLUG)
    assert isinstance(read, ReadSkillResponse), read.message
    assert len(read.files) == len(files) == 5
    assert read.package_dir is not None

    # The path read_skill advertises is the one read_workspace_file must find.
    script = f"/experts/{expert}/skills/{SLUG}/scripts/with_server.py"
    assert script in {f.path for f in read.files}
    content = await ReadWorkspaceFileTool()._execute(
        user_id=creator, session=session, path=script
    )
    assert isinstance(content, WorkspaceFileContentResponse), content.message

    source = FIXTURE_DIR / "scripts" / "with_server.py"
    materialised = os.path.join(read.package_dir, "scripts", "with_server.py")
    with open(materialised, "rb") as handle:
        assert handle.read() == source.read_bytes()
    assert os.stat(materialised).st_mode & 0o111


async def test_the_version_keeps_its_own_bytes_when_the_library_copy_changes(
    creator: str, setup_admin_user: str
):
    await _store_library_package(creator, _fixture_files())
    published = await _publish(creator, setup_admin_user)

    await _store_library_package(
        creator,
        [SkillFile(relative_path="references/notes.md", content=b"rewritten")],
    )

    rows = await prisma.models.SkillListingFile.prisma().find_many(
        where={"skillListingVersionId": published.skill_listing_version_id}
    )
    # Sorted here rather than in the query: Postgres orders by its own
    # collation, which puts `examples/` before `LICENSE.txt`.
    assert sorted(row.relativePath for row in rows) == [
        "LICENSE.txt",
        "examples/console_logging.py",
        "examples/element_discovery.py",
        "examples/static_html_automation.py",
        "scripts/with_server.py",
    ]
    script = next(r for r in rows if r.relativePath == "scripts/with_server.py")
    source = (FIXTURE_DIR / "scripts" / "with_server.py").read_bytes()
    assert script.content.decode() == source
    assert script.sha256 == hashlib.sha256(source).hexdigest()
    assert script.sizeBytes == len(source)
    assert script.mimeType == "text/x-python"
    assert script.isExecutable is True
    assert [r.isExecutable for r in rows].count(True) == 1


# Both re-install tests install into an EXPERT, never the library: the library
# folder is where the creator's own skill lives, so re-storing it there would
# remove the sibling by itself and the assertion would hold with the install
# doing nothing.
async def test_installing_a_newer_version_drops_a_sibling_the_old_one_had(
    creator: str, expert: str, setup_admin_user: str
):
    await _store_library_package(creator, _fixture_files())
    await _publish(creator, setup_admin_user)
    await skill_db.install_marketplace_skill(creator, SLUG, expert_id=expert)

    await _store_library_package(
        creator, [f for f in _fixture_files() if f.relative_path != "LICENSE.txt"]
    )
    await _publish(creator, setup_admin_user)
    await skill_db.install_marketplace_skill(creator, SLUG, expert_id=expert)

    installed = {
        f.path for f in await list_user_skill_files(creator, SLUG, expert_id=expert)
    }
    folder = f"/experts/{expert}/skills/{SLUG}"
    assert f"{folder}/LICENSE.txt" not in installed
    assert f"{folder}/scripts/with_server.py" in installed


async def test_installing_a_single_file_listing_clears_a_package_left_behind(
    creator: str, expert: str, setup_admin_user: str
):
    """A version with no file rows is a single-file skill, so its install must
    pass an empty package rather than "leave the folder alone"."""
    await _store_library_package(creator, _fixture_files())
    await _publish(creator, setup_admin_user)
    await skill_db.install_marketplace_skill(creator, SLUG, expert_id=expert)

    await _store_library_package(creator, [])
    await _publish(creator, setup_admin_user)
    await skill_db.install_marketplace_skill(creator, SLUG, expert_id=expert)

    assert await list_user_skill_files(creator, SLUG, expert_id=expert) == []
