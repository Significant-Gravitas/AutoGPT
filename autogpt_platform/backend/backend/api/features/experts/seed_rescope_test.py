from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from backend.api.features.experts import seed


async def test_rescope_retry_uses_recorded_defaults_and_preserves_owner_edits():
    legacy = SimpleNamespace(
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
            **vars(legacy),
            "role": "SEO & Content",
            "identity": "SEO identity",
            "jobTitle": "SEO Content Strategist",
            "tagline": "SEO tasks",
            "bio": "SEO bio",
            "categories": ["marketing", "content"],
        }
    )
    hires = [
        SimpleNamespace(
            **{
                **vars(legacy),
                "id": "hire-1",
                "updatedAt": "version-1",
                "avatarUrl": "/my-avatar.png",
            }
        ),
        SimpleNamespace(
            **{
                **vars(legacy),
                "id": "hire-2",
                "updatedAt": "version-2",
                "bio": "My biography",
                "jobTitle": "My title",
            }
        ),
        SimpleNamespace(
            **{
                **vars(legacy),
                "id": "hire-3",
                "updatedAt": "version-3",
                "identity": "My instructions",
            }
        ),
    ]
    db = AsyncMock()
    db.find_many.side_effect = [hires[:2], hires[2:]]
    db.update_many.return_value = 1
    with (
        patch.object(seed.prisma.models.Expert, "prisma", return_value=db),
        patch.object(seed, "_PRESENTATION_BACKFILL_BATCH_SIZE", 2),
        patch.object(
            seed,
            "RESCOPED_TEMPLATES",
            [
                {
                    "name": "Maria",
                    "old_role": legacy.role,
                    "old_identity": legacy.identity,
                    "old_presentation": legacy,
                }
            ],
        ),
    ):
        assert await seed._backfill_hired_copies(replacement, replacement) == 2
    first, second = db.update_many.await_args_list
    assert first.kwargs["data"] == {
        "role": "SEO & Content",
        "identity": "SEO identity",
        "jobTitle": "SEO Content Strategist",
        "tagline": "SEO tasks",
        "bio": "SEO bio",
        "categories": ["marketing", "content"],
    }
    assert second.kwargs["data"] == {
        "role": "SEO & Content",
        "identity": "SEO identity",
        "tagline": "SEO tasks",
        "categories": ["marketing", "content"],
    }
    assert first.kwargs["where"]["updatedAt"] == "version-1"
    assert second.kwargs["where"]["updatedAt"] == "version-2"
    assert db.find_many.await_args_list[1].kwargs["where"]["id"] == {"gt": "hire-2"}


@pytest.mark.parametrize("name", ["Maria", "Max"])
async def test_recorded_rescopes_resume_after_template_has_advanced(name):
    rescope = next(item for item in seed.RESCOPED_TEMPLATES if item["name"] == name)
    entry = next(item for item in seed.ROSTER if item["name"] == name)
    template = SimpleNamespace(
        id="template",
        name=name,
        role=entry["role"],
        identity=entry["identity"],
        jobTitle=entry["job_title"],
        tagline=entry["tagline"],
        bio=entry["bio"],
        categories=entry["categories"],
    )
    hire = SimpleNamespace(
        **rescope["old_presentation"].model_dump(),
        id="hire",
        avatarUrl="/my-avatar.png",
        role=rescope["old_role"],
        identity=rescope["old_identity"],
        updatedAt="version-read",
    )
    db = AsyncMock()
    db.find_many.return_value = [hire]
    db.update_many.return_value = 1
    with patch.object(seed.prisma.models.Expert, "prisma", return_value=db):
        assert await seed._backfill_hired_copies(template, template) == 1
        changes = db.update_many.await_args.kwargs["data"]
        for field in ("role", "identity", "jobTitle", "tagline", "bio", "categories"):
            assert changes.get(field, vars(hire)[field]) == vars(template)[field]
        assert "avatarUrl" not in changes
        for field, value in changes.items():
            setattr(hire, field, value)
        db.update_many.reset_mock()
        assert await seed._backfill_hired_copies(template, template) == 0
        db.update_many.assert_not_awaited()
