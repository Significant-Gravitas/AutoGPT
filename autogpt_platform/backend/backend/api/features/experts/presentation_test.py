from types import SimpleNamespace


def test_presentation_update_keeps_custom_fields_and_behavior():
    from backend.api.features.experts.presentation import presentation_changes

    previous = SimpleNamespace(
        avatarUrl="/old.svg",
        jobTitle="Writer",
        tagline="Old",
        bio="Old bio",
        categories=["content"],
    )
    current = SimpleNamespace(
        avatarUrl="https://example.com/custom.png",
        jobTitle="My title",
        tagline="Old",
        bio="My biography",
        categories=["research"],
    )
    replacement = SimpleNamespace(
        avatarUrl="/new.png",
        jobTitle="Content Writer",
        tagline="New",
        bio="New bio",
        categories=["marketing"],
    )
    assert presentation_changes(current, previous, replacement) == {"tagline": "New"}


def test_hired_identity_is_not_projected_from_template_name():
    from backend.api.features.experts.presentation import template_presentation

    row = SimpleNamespace(
        name="Maria",
        isTemplate=False,
        avatarUrl="/experts/maria.svg",
        bio="Custom bio",
        identity="Custom instructions",
        tagline="Custom tagline",
    )
    assert template_presentation(row)["avatarUrl"] == "/experts/maria.svg"


def test_template_projection_changes_only_recorded_defaults():
    from backend.api.features.experts.presentation import template_presentation

    row = SimpleNamespace(
        name="Maria",
        isTemplate=True,
        avatarUrl="/experts/maria.svg",
        bio="Custom bio",
        identity="Custom instructions",
        tagline="Custom tagline",
    )
    result = template_presentation(row)
    assert (
        result["avatarUrl"] == "/autogpt-characters/v1.1/expert-maria/neutral/128.webp"
    )
    assert result["bio"] == "Custom bio"
    assert result["identity"] == "Custom instructions"


async def test_backfill_keeps_custom_avatar_and_uses_concurrency_guard():
    from unittest.mock import AsyncMock, patch

    from backend.api.features.experts.seed import _backfill_hired_copies

    previous = SimpleNamespace(
        id="template",
        name="Test",
        avatarUrl="/old.svg",
        jobTitle="Writer",
        tagline="Old",
        bio="Old bio",
        categories=["content"],
        role="Writing",
        identity="Instructions",
    )
    replacement = SimpleNamespace(
        **{**vars(previous), "avatarUrl": "/new.webp", "tagline": "New"}
    )
    hired = SimpleNamespace(
        **{
            **vars(previous),
            "id": "hire",
            "avatarUrl": "https://custom.example/image.png",
            "updatedAt": "version-read",
        }
    )
    db = AsyncMock()
    db.find_many.return_value = [hired]
    db.update_many.return_value = 1
    with patch(
        "backend.api.features.experts.seed.prisma.models.Expert.prisma", return_value=db
    ):
        assert await _backfill_hired_copies(replacement, previous) == 1
    kwargs = db.update_many.call_args.kwargs
    assert kwargs["data"] == {"tagline": "New"}
    assert kwargs["where"]["updatedAt"] == "version-read"
    assert kwargs["where"]["sourceTemplateId"] == "template"
    assert kwargs["where"]["isTemplate"] is False
