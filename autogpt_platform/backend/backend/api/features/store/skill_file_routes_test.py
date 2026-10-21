"""Reading one file of a published skill package.

The viewer exists so someone deciding to install a package — or an admin
deciding to approve one — can read its scripts before committing to them. It
serves text and refuses everything else, and a submission nobody has approved
is the submitter's and the reviewer's alone.
"""

from unittest.mock import AsyncMock, patch

import fastapi
import prisma.models
import pytest
import pytest_mock

from backend.api.features.admin import store_admin_routes
from backend.copilot.tools.skills import (
    SkillFile,
    parse_skill_markdown,
    store_user_skill,
)
from backend.util.exceptions import NotFoundError
from backend.util.test import SpinTestServer

from . import skill_db, skill_model, skill_routes, skill_submission_db

SLUG = "package-viewer"
SKILL_MD = (
    "---\nname: package-viewer\ndescription: A package worth reading\n---\n\nRead me.\n"
)
SCRIPT = SkillFile(
    relative_path="scripts/run.sh",
    content=b"#!/bin/sh\necho hi\n",
    is_executable=True,
)
# Enough of a header to be unmistakably not text, and short enough to publish.
PNG = SkillFile(
    relative_path="assets/logo.png", content=b"\x89PNG\r\n\x1a\n" + bytes(16)
)


async def test_the_live_version_serves_a_text_file_to_anyone(
    creator: str, setup_admin_user: str
):
    """Anonymous, because the listing this belongs to is public. Without it
    every refusal below would pass on a door that never opens."""
    await _publish(creator, [SCRIPT], approved_by=setup_admin_user)

    text = await skill_routes.read_marketplace_skill_file(
        slug=SLUG, path=SCRIPT.relative_path, version_id=None, user_id=None
    )

    assert text == "#!/bin/sh\necho hi\n"


async def test_a_file_whose_type_has_no_name_is_decided_on_its_bytes(
    creator: str, setup_admin_user: str
):
    """`mimetypes` cannot name a `Makefile`, and a reviewer still has to read
    one, so the decode decides rather than the missing name."""
    await _publish(
        creator,
        [SkillFile(relative_path="Makefile", content=b"all:\n\techo hi\n")],
        approved_by=setup_admin_user,
    )
    row = await prisma.models.SkillListingFile.prisma().find_first(
        where={"relativePath": "Makefile"}
    )
    assert row is not None and row.mimeType is None, "the null branch moved"

    text = await skill_routes.read_marketplace_skill_file(
        slug=SLUG, path="Makefile", version_id=None, user_id=None
    )

    assert text == "all:\n\techo hi\n"


async def test_a_file_over_the_cap_is_refused_without_reading_its_bytes(
    creator: str, setup_admin_user: str, mocker: pytest_mock.MockerFixture
):
    oversized = SkillFile(
        relative_path="references/big.md",
        content=b"x" * (skill_routes.MAX_VIEWABLE_FILE_BYTES + 1),
    )
    await _publish(creator, [oversized], approved_by=setup_admin_user)
    read_bytes = mocker.spy(skill_db, "read_package_file_bytes")

    with pytest.raises(fastapi.HTTPException) as refused:
        await skill_routes.read_marketplace_skill_file(
            slug=SLUG, path=oversized.relative_path, version_id=None, user_id=None
        )

    assert refused.value.status_code == 413
    # The cap is decided on the metadata row, so the bytes it refuses are
    # never loaded to refuse them.
    assert read_bytes.call_count == 0


async def test_a_binary_file_is_refused_by_its_type(
    creator: str, setup_admin_user: str, mocker: pytest_mock.MockerFixture
):
    await _publish(creator, [PNG], approved_by=setup_admin_user)
    read_bytes = mocker.spy(skill_db, "read_package_file_bytes")

    with pytest.raises(fastapi.HTTPException) as refused:
        await skill_routes.read_marketplace_skill_file(
            slug=SLUG, path=PNG.relative_path, version_id=None, user_id=None
        )

    assert refused.value.status_code == 415
    assert "image/png" in refused.value.detail
    assert read_bytes.call_count == 0


async def test_a_path_outside_the_package_is_not_found_and_reads_nothing(
    creator: str, setup_admin_user: str, mocker: pytest_mock.MockerFixture
):
    """The path is matched against the published rows, so it addresses no
    storage and a traversal attempt is simply a path the package lacks."""
    await _publish(creator, [SCRIPT], approved_by=setup_admin_user)
    read_bytes = mocker.spy(skill_db, "read_package_file_bytes")

    with pytest.raises(fastapi.HTTPException) as refused:
        await skill_routes.read_marketplace_skill_file(
            slug=SLUG, path="../../etc/passwd", version_id=None, user_id=None
        )

    assert refused.value.status_code == 404
    assert read_bytes.call_count == 0


async def test_a_pending_submission_is_readable_by_its_submitter_alone(
    creator: str, stranger: str
):
    """A first submission has no approved version, so the slug resolves to no
    live listing at all — which is why the version id exists."""
    version_id = await _publish(creator, [SCRIPT], approved_by=None)

    assert (
        await skill_routes.read_marketplace_skill_file(
            slug=SLUG,
            path=SCRIPT.relative_path,
            version_id=version_id,
            user_id=creator,
        )
        == "#!/bin/sh\necho hi\n"
    )

    # `rest_api.py` maps NotFoundError to 404, so a stranger cannot tell an
    # unapproved package from one that does not exist.
    for caller in (stranger, None):
        with pytest.raises(NotFoundError):
            await skill_routes.read_marketplace_skill_file(
                slug=SLUG,
                path=SCRIPT.relative_path,
                version_id=version_id,
                user_id=caller,
            )
    with pytest.raises(NotFoundError):
        await skill_routes.read_marketplace_skill_file(
            slug=SLUG, path=SCRIPT.relative_path, version_id=None, user_id=creator
        )


async def test_a_reviewer_reads_a_pending_submission_through_the_admin_route(
    creator: str,
):
    """The admin router's own security is what grants this; the handler is
    reached by version id because a first submission has no live slug."""
    version_id = await _publish(creator, [SCRIPT], approved_by=None)

    text = await store_admin_routes.read_skill_submission_file(
        version_id, SCRIPT.relative_path
    )

    assert text == "#!/bin/sh\necho hi\n"


async def test_a_pending_submission_lists_its_files_for_the_reviewer(creator: str):
    await _publish(creator, [SCRIPT, PNG], approved_by=None)

    pending = await skill_submission_db.list_pending_skill_submissions()
    [submission] = [s for s in pending if s.slug == SLUG]

    assert [(f.path, f.is_executable) for f in submission.files] == [
        (PNG.relative_path, False),
        (SCRIPT.relative_path, True),
    ]
    assert submission.files[0].mime_type == "image/png"
    assert submission.files[1].size_bytes == len(SCRIPT.content)
    # Not a literal: `.sh` is `application/x-sh` here and `text/x-sh` on CI.
    assert submission.files[1].mime_type is not None


async def test_the_detail_response_names_each_file_type_and_mode(
    creator: str, setup_admin_user: str
):
    """Against the stored rows, never a literal type: publish derives the type
    with ``mimetypes``, which reads the host's /etc/mime.types, so `.sh` is
    ``application/x-sh`` on this machine and ``text/x-sh`` on CI."""
    await _publish(creator, [SCRIPT, PNG], approved_by=setup_admin_user)

    details = await skill_db.get_marketplace_skill(SLUG)

    stored = {
        row.relativePath: row
        for row in await prisma.models.SkillListingFile.prisma().find_many(
            where={
                "SkillListingVersion": {"is": {"SkillListing": {"is": {"slug": SLUG}}}}
            }
        )
    }
    assert [(f.path, f.mime_type, f.is_executable) for f in details.files] == [
        (PNG.relative_path, stored[PNG.relative_path].mimeType, False),
        (SCRIPT.relative_path, stored[SCRIPT.relative_path].mimeType, True),
    ]
    assert stored[PNG.relative_path].mimeType == "image/png"


async def _publish(
    user_id: str, files: list[SkillFile], *, approved_by: str | None
) -> str:
    """Store *files* as the caller's library skill and submit it, returning the
    submitted version's id; approved when *approved_by* names a reviewer."""
    parsed = parse_skill_markdown(SKILL_MD, fallback_name=SLUG)
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
    submission = await skill_submission_db.submit_skill(
        user_id,
        skill_model.SkillSubmissionRequest(skill_name=SLUG, categories=["content"]),
    )
    if approved_by is not None:
        await skill_submission_db.review_skill_submission(
            submission.skill_listing_version_id,
            is_approved=True,
            reviewer_id=approved_by,
            comments="ok",
        )
    return submission.skill_listing_version_id


@pytest.fixture
async def creator(setup_test_user, server: SpinTestServer):
    await _drop_our_listing()
    await prisma.models.Profile.prisma().upsert(
        where={"userId": setup_test_user},
        data={
            "create": {
                "userId": setup_test_user,
                "username": "viewer-creator",
                "name": "Viewer Creator",
                "description": "",
                "links": [],
            },
            "update": {},
        },
    )
    yield setup_test_user
    # Left behind, these rows fail the emptiness guard in the store suites that
    # sort after this one.
    await _drop_our_listing()


async def _drop_our_listing() -> None:
    """By slug, not wholesale: this table is shared with every other checkout
    here, where the starter listings are real rows."""
    await prisma.models.SkillListingVersion.prisma().delete_many(
        where={"SkillListing": {"is": {"slug": SLUG}}}
    )
    await prisma.models.SkillListing.prisma().delete_many(where={"slug": SLUG})


@pytest.fixture
def stranger() -> str:
    """Never a row: the check compares the listing's owner to a caller id, so
    a string that owns nothing is exactly the case under test."""
    return "a-user-who-owns-nothing"


@pytest.fixture(autouse=True)
def skills_enabled():
    with patch(
        "backend.copilot.tools.skills.is_skills_feature_enabled",
        new=AsyncMock(return_value=True),
    ):
        yield
