import pytest

from backend.data.skill_content_approvals import parse_content_approvals

from . import content_approvals
from .content_checks import check_skill_bundle, collect_content_approvals

EXAMPLE = "Bearer " + "x" * 30


def _approved(files):
    return collect_content_approvals(
        files, approved_by_user_id="owner", allowed_pattern_classes=["bearer_token"]
    )


@pytest.mark.parametrize("change", ["new-value", "extra-copy", "other-file", "seeded"])
def test_approval_cannot_authorize_new_or_relocated_or_seeded_content(change):
    original = {"SKILL.md": "Example: " + EXAMPLE}
    approvals = _approved(original)
    assert len(approvals) == 1 and approvals[0].count == 1
    assert EXAMPLE not in approvals[0].model_dump_json()
    candidate = {"SKILL.md": "New instructions\n" + EXAMPLE}
    assert check_skill_bundle(candidate, approved_matches=approvals) is None
    seeds = []
    if change == "new-value":
        candidate["SKILL.md"] = EXAMPLE.replace("x", "y")
    elif change == "extra-copy":
        candidate["SKILL.md"] += "\n" + EXAMPLE
    elif change == "other-file":
        candidate = {"references/example.md": EXAMPLE}
    else:
        seeds = [EXAMPLE]
    assert check_skill_bundle(
        candidate, approved_matches=approvals, seeded_values=seeds
    )


def test_removed_occurrences_do_not_leave_allowance_for_future_copies():
    approvals = _approved({"SKILL.md": EXAMPLE + "\n" + EXAMPLE})
    assert approvals[0].count == 2
    reduced = collect_content_approvals(
        {"SKILL.md": EXAMPLE}, approved_by_user_id="owner", approved_matches=approvals
    )
    assert reduced[0].count == 1
    assert (
        collect_content_approvals(
            {"SKILL.md": "No examples"},
            approved_by_user_id="owner",
            approved_matches=reduced,
        )
        == []
    )
    assert (
        collect_content_approvals(
            {"SKILL.md": EXAMPLE}, approved_by_user_id="other", approved_matches=reduced
        )
        == []
    )
    assert parse_content_approvals([{"pattern_class": "bearer_token"}]) == []


@pytest.mark.asyncio
async def test_saved_approvals_are_scoped_to_the_current_owner_head(fake_store):
    head = await fake_store.ensure_head("owner", "expert-one", "example")
    version = await fake_store.create_version(
        "owner",
        head=head,
        content=EXAMPLE,
        description="Example",
        triggers=[],
        origin="edited",
        content_approvals=_approved({"SKILL.md": EXAMPLE}),
    )
    fake_store._put_head(
        head.model_copy(
            update={
                "current_version_id": version.id,
                "current_version": version.version,
                "content_hash": version.content_hash,
            }
        )
    )
    assert await content_approvals.current_content_approvals(
        "owner", "expert-one", "example"
    )
    assert (
        await content_approvals.current_content_approvals(
            "other", "expert-one", "example"
        )
        == []
    )
    assert (
        await content_approvals.current_content_approvals(
            "owner", "expert-two", "example"
        )
        == []
    )
    await fake_store.set_version_state("owner", version.id, "invalidated")
    assert (
        await content_approvals.current_content_approvals(
            "owner", "expert-one", "example"
        )
        == []
    )
