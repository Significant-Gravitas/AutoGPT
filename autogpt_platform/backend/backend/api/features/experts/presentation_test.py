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
        avatarUrl="/old.svg",
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
    assert result["avatarUrl"] == "/experts/clay/v5/maria-marketing.png"
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


async def test_rescope_retry_defers_until_legacy_baseline_is_available():
    from unittest.mock import AsyncMock, patch

    from backend.api.features.experts.seed import _backfill_hired_copies

    previous = SimpleNamespace(
        id="template",
        name="Maria",
        avatarUrl="/old.svg",
        jobTitle="Marketing Specialist",
        tagline="Marketing tasks",
        bio="Marketing bio",
        categories=["marketing"],
        role="Marketing",
        identity="Legacy marketing identity",
    )
    replacement = SimpleNamespace(
        **{
            **vars(previous),
            "role": "SEO & Content",
            "identity": "SEO identity",
            "jobTitle": "SEO Content Strategist",
            "tagline": "SEO tasks",
            "bio": "SEO bio",
        }
    )
    hired = SimpleNamespace(
        **{**vars(previous), "id": "hire", "updatedAt": "version-read"}
    )
    db = AsyncMock()
    db.find_many.return_value = [hired]
    db.update_many.return_value = 1
    with (
        patch(
            "backend.api.features.experts.seed.prisma.models.Expert.prisma",
            return_value=db,
        ),
        patch(
            "backend.api.features.experts.seed.RESCOPED_TEMPLATES",
            [
                {
                    "name": "Maria",
                    "old_role": previous.role,
                    "old_identity": previous.identity,
                }
            ],
        ),
    ):
        assert await _backfill_hired_copies(replacement, replacement) == 0
        db.update_many.assert_not_awaited()
        assert await _backfill_hired_copies(replacement, previous) == 1
    assert db.update_many.call_args.kwargs["data"] == {
        "role": "SEO & Content",
        "identity": "SEO identity",
        "jobTitle": "SEO Content Strategist",
        "tagline": "SEO tasks",
        "bio": "SEO bio",
    }


async def test_backfill_reads_bounded_batches_without_offset_skips():
    from unittest.mock import AsyncMock, patch

    from backend.api.features.experts import seed

    previous = SimpleNamespace(
        id="template",
        name="Test",
        avatarUrl=None,
        jobTitle="Writer",
        tagline="Old",
        bio=None,
        categories=[],
        role="Writing",
        identity="Instructions",
    )
    replacement = SimpleNamespace(**{**vars(previous), "tagline": "New"})
    hires = [
        SimpleNamespace(**{**vars(previous), "id": str(i), "updatedAt": str(i)})
        for i in range(3)
    ]
    db = AsyncMock()
    db.find_many.side_effect = [hires[:2], hires[2:]]
    db.update_many.return_value = 1
    with (
        patch.object(seed, "_PRESENTATION_BACKFILL_BATCH_SIZE", 2),
        patch.object(seed.prisma.models.Expert, "prisma", return_value=db),
    ):
        assert await seed._backfill_hired_copies(replacement, previous) == 3
    assert db.find_many.await_count == 2
    for call in db.find_many.await_args_list:
        assert call.kwargs["take"] == 2
        assert call.kwargs["order"] == {"id": "asc"}
    assert db.find_many.await_args_list[1].kwargs["where"]["id"] == {"gt": "1"}
    assert [
        call.kwargs["where"]["updatedAt"] for call in db.update_many.await_args_list
    ] == ["0", "1", "2"]


def test_template_projection_replaces_shared_draft_avatars_only():
    from backend.api.features.experts.presentation import template_presentation

    row = SimpleNamespace(
        name="Noor",
        isTemplate=True,
        avatarUrl="/experts/clay/v1/marketing.png",
        bio=None,
        identity="",
        tagline=None,
    )
    assert (
        template_presentation(row)["avatarUrl"] == "/experts/clay/v5/noor-marketing.png"
    )
    row.avatarUrl = "https://custom.example/image.png"
    assert template_presentation(row)["avatarUrl"] == row.avatarUrl
    row.avatarUrl = "/experts/clay/v1/finance.png"
    assert template_presentation(row)["avatarUrl"] == row.avatarUrl
    row.avatarUrl = "/experts/clay/v1/marketing.png"
    row.isTemplate = False
    assert template_presentation(row)["avatarUrl"] == row.avatarUrl


async def test_hired_avatar_refresh_is_scoped_to_its_template_and_known_default():
    from unittest.mock import AsyncMock, patch

    from backend.api.features.experts.seed import _backfill_hired_copies

    template = SimpleNamespace(
        id="template",
        name="Noor",
        avatarUrl="/experts/clay/v5/noor-marketing.png",
        jobTitle="Writer",
        tagline="Hi",
        bio=None,
        categories=[],
        role="Communications",
        identity="Instructions",
    )
    hires = [
        SimpleNamespace(
            **{**vars(template), "id": str(i), "updatedAt": str(i), "avatarUrl": url}
        )
        for i, url in enumerate(
            [
                "/experts/clay/v1/marketing.png",
                "https://custom.example/image.png",
                "/experts/clay/v1/finance.png",
            ]
        )
    ]
    db = AsyncMock()
    db.find_many.return_value = hires
    db.update_many.return_value = 1
    with patch(
        "backend.api.features.experts.seed.prisma.models.Expert.prisma", return_value=db
    ):
        assert await _backfill_hired_copies(template, template) == 1
    db.update_many.assert_awaited_once_with(
        where={
            "id": "0",
            "sourceTemplateId": "template",
            "isTemplate": False,
            "updatedAt": "0",
        },
        data={"avatarUrl": "/experts/clay/v5/noor-marketing.png"},
    )
