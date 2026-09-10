import prisma.enums
import prisma.models
import pytest

from backend.copilot.tools.skills import ParsedSkill
from backend.util.exceptions import NotFoundError, PreconditionFailed
from backend.util.test import SpinTestServer

from . import skill_db, skill_model, skill_submission_db

LIBRARY_SKILL = ParsedSkill(
    name="brand-voice-guide",
    description="Write in a consistent brand voice.",
    body="# Brand voice\nLead with positioning.\n",
    triggers=("brand voice",),
)


def _request(**overrides) -> skill_model.SkillSubmissionRequest:
    data = {
        "skill_name": "brand-voice-guide",
        "categories": ["content"],
        "required_providers": ["google"],
    }
    data.update(overrides)
    return skill_model.SkillSubmissionRequest(**data)


@pytest.fixture(autouse=True)
async def library_skill(mocker, server: SpinTestServer):
    mocker.patch.object(
        skill_submission_db, "read_user_skill_with_body", return_value=LIBRARY_SKILL
    )
    await prisma.models.SkillListingVersion.prisma().delete_many()
    await prisma.models.SkillListing.prisma().delete_many()
    yield
    await prisma.models.SkillListingVersion.prisma().delete_many()
    await prisma.models.SkillListing.prisma().delete_many()


@pytest.fixture
async def reviewer(setup_admin_user) -> str:
    """A real admin row: SkillListingVersion.reviewerId is a User foreign key."""
    return setup_admin_user


@pytest.fixture
async def creator(setup_test_user) -> str:
    await prisma.models.Profile.prisma().upsert(
        where={"userId": setup_test_user},
        data={
            "create": {
                "userId": setup_test_user,
                "username": "creator",
                "name": "Creator",
                "description": "",
                "links": [],
            },
            "update": {},
        },
    )
    return setup_test_user


async def test_publishing_does_not_put_the_skill_on_the_shelf(creator):
    submission = await skill_submission_db.submit_skill(creator, _request())

    assert submission.status == prisma.enums.SubmissionStatus.PENDING
    assert submission.is_live is False
    browse = await skill_db.get_marketplace_skills()
    assert browse.skills == []


async def test_publishing_without_a_marketplace_profile_is_refused(setup_test_user):
    await prisma.models.Profile.prisma().delete_many(where={"userId": setup_test_user})

    with pytest.raises(PreconditionFailed, match="Marketplace Profile"):
        await skill_submission_db.submit_skill(setup_test_user, _request())

    assert await prisma.models.SkillListing.prisma().count() == 0


async def test_publishing_a_skill_that_is_not_in_the_library_is_refused(
    creator, mocker
):
    mocker.patch.object(
        skill_submission_db, "read_user_skill_with_body", return_value=None
    )

    with pytest.raises(NotFoundError):
        await skill_submission_db.submit_skill(creator, _request())


async def test_a_slug_another_creator_holds_is_refused(creator):
    await prisma.models.SkillListing.prisma().create(
        data={"slug": "brand-voice-guide", "owningUserId": None}
    )

    with pytest.raises(PreconditionFailed, match="taken"):
        await skill_submission_db.submit_skill(creator, _request())


async def test_approval_promotes_the_version_and_puts_it_on_the_shelf(
    creator, reviewer
):
    submission = await skill_submission_db.submit_skill(creator, _request())

    reviewed = await skill_submission_db.review_skill_submission(
        submission.skill_listing_version_id,
        is_approved=True,
        reviewer_id=reviewer,
        comments="Looks good",
    )

    assert reviewed.status == prisma.enums.SubmissionStatus.APPROVED
    assert reviewed.is_live is True
    browse = await skill_db.get_marketplace_skills()
    assert [s.slug for s in browse.skills] == ["brand-voice-guide"]


async def test_rejection_keeps_the_skill_off_the_shelf(creator, reviewer):
    submission = await skill_submission_db.submit_skill(creator, _request())

    reviewed = await skill_submission_db.review_skill_submission(
        submission.skill_listing_version_id,
        is_approved=False,
        reviewer_id=reviewer,
        comments="Needs work",
    )

    assert reviewed.status == prisma.enums.SubmissionStatus.REJECTED
    assert reviewed.is_live is False
    browse = await skill_db.get_marketplace_skills()
    assert browse.skills == []


async def test_republishing_leaves_the_live_version_serving_until_approved(
    creator, reviewer, mocker
):
    first = await skill_submission_db.submit_skill(creator, _request())
    await skill_submission_db.review_skill_submission(
        first.skill_listing_version_id,
        is_approved=True,
        reviewer_id=reviewer,
        comments="ok",
    )
    mocker.patch.object(
        skill_submission_db,
        "read_user_skill_with_body",
        return_value=ParsedSkill(
            name="brand-voice-guide",
            description="Rewritten.",
            body="# Rewritten\n",
            triggers=(),
        ),
    )

    second = await skill_submission_db.submit_skill(creator, _request())

    assert second.version == 2
    assert second.is_live is False
    live = await skill_db.get_marketplace_skill("brand-voice-guide")
    assert live.body == "# Brand voice\nLead with positioning.\n"


async def test_editing_re_snapshots_the_library_skill(creator, mocker):
    submission = await skill_submission_db.submit_skill(creator, _request())
    mocker.patch.object(
        skill_submission_db,
        "read_user_skill_with_body",
        return_value=ParsedSkill(
            name="brand-voice-guide",
            description="Tightened.",
            body="# Tightened\n",
            triggers=(),
        ),
    )

    edited = await skill_submission_db.edit_skill_submission(
        creator,
        submission.skill_listing_version_id,
        _request(categories=["marketing"]),
    )

    assert edited.description == "Tightened."
    assert edited.categories == ["marketing"]


async def test_an_approved_submission_cannot_be_edited(creator, reviewer):
    submission = await skill_submission_db.submit_skill(creator, _request())
    await skill_submission_db.review_skill_submission(
        submission.skill_listing_version_id,
        is_approved=True,
        reviewer_id=reviewer,
        comments="ok",
    )

    with pytest.raises(PreconditionFailed, match="pending"):
        await skill_submission_db.edit_skill_submission(
            creator, submission.skill_listing_version_id, _request()
        )


async def test_another_creators_submission_is_not_editable(creator):
    submission = await skill_submission_db.submit_skill(creator, _request())

    with pytest.raises(NotFoundError):
        await skill_submission_db.edit_skill_submission(
            "someone-else", submission.skill_listing_version_id, _request()
        )


async def test_the_review_queue_holds_only_pending_submissions(creator, reviewer):
    pending = await skill_submission_db.submit_skill(creator, _request())
    assert [
        s.skill_listing_version_id
        for s in await skill_submission_db.list_pending_skill_submissions()
    ] == [pending.skill_listing_version_id]

    await skill_submission_db.review_skill_submission(
        pending.skill_listing_version_id,
        is_approved=True,
        reviewer_id=reviewer,
        comments="ok",
    )

    assert await skill_submission_db.list_pending_skill_submissions() == []


async def test_editing_cannot_repoint_the_submission_at_another_skill(creator, mocker):
    submission = await skill_submission_db.submit_skill(creator, _request())
    mocker.patch.object(
        skill_submission_db,
        "read_user_skill_with_body",
        return_value=ParsedSkill(
            name="other-skill",
            description="Someone else's content.",
            body="# Other\n",
            triggers=(),
        ),
    )

    with pytest.raises(PreconditionFailed, match="brand-voice-guide"):
        await skill_submission_db.edit_skill_submission(
            creator,
            submission.skill_listing_version_id,
            _request(skill_name="other-skill"),
        )

    unchanged = await skill_submission_db.list_my_skill_submissions(creator)
    assert unchanged[0].description == LIBRARY_SKILL.description


async def test_a_rejected_submission_cannot_be_re_approved(creator, reviewer):
    submission = await skill_submission_db.submit_skill(creator, _request())
    await skill_submission_db.review_skill_submission(
        submission.skill_listing_version_id,
        is_approved=False,
        reviewer_id=reviewer,
        comments="Needs work",
    )

    with pytest.raises(PreconditionFailed, match="pending"):
        await skill_submission_db.review_skill_submission(
            submission.skill_listing_version_id,
            is_approved=True,
            reviewer_id=reviewer,
            comments="Changed my mind",
        )

    browse = await skill_db.get_marketplace_skills()
    assert browse.skills == []


async def test_a_deleted_listing_is_not_in_the_review_queue(creator):
    submission = await skill_submission_db.submit_skill(creator, _request())
    await prisma.models.SkillListing.prisma().update(
        where={"slug": "brand-voice-guide"}, data={"isDeleted": True}
    )

    assert await skill_submission_db.list_pending_skill_submissions() == []

    await prisma.models.SkillListing.prisma().update(
        where={"slug": "brand-voice-guide"}, data={"isDeleted": False}
    )
    await prisma.models.SkillListingVersion.prisma().update(
        where={"id": submission.skill_listing_version_id}, data={"isDeleted": True}
    )

    assert await skill_submission_db.list_pending_skill_submissions() == []
