"""Seed the platform-authored starter skill listings.

Run as ``python -m backend.api.features.store.skill_seed``, like the expert
roster seed. Idempotent: re-running updates the live version in place rather
than stacking a new one, so the catalogue can be edited by editing the
markdown.

Each listing's content is its ``SKILL.md`` under ``starter_skills/``, parsed
with the same :func:`parse_skill_markdown` the copilot and the upload endpoint
use, so a starter skill cannot drift from the format an installed skill has.
The seed adds only what a listing needs beyond the file: its categories and
the integrations its instructions assume.
"""

import asyncio
import logging
from pathlib import Path
from typing import TypedDict

import prisma.enums
import prisma.models

from backend.copilot.tools.skills import ParsedSkill, parse_skill_markdown
from backend.data import db as database

logger = logging.getLogger(__name__)

_CONTENT_DIR = Path(__file__).parent / "starter_skills"


class StarterSkill(TypedDict):
    slug: str
    categories: list[str]
    required_providers: list[str]


STARTER_SKILLS: list[StarterSkill] = [
    {
        "slug": "brand-voice-guide",
        "categories": ["content"],
        "required_providers": [],
    },
    {
        "slug": "outreach-playbook",
        "categories": ["sales"],
        "required_providers": ["google"],
    },
]


async def seed_starter_skills() -> list[str]:
    """Upsert every starter listing. Returns the listing ids."""
    listing_ids = []
    for entry in STARTER_SKILLS:
        parsed = _load(entry["slug"])
        listing = await _upsert_listing(entry, parsed)
        listing_ids.append(listing.id)
        logger.info(f"Seeded starter skill '{entry['slug']}' (#{listing.id})")
    return listing_ids


async def _upsert_listing(
    entry: StarterSkill, parsed: ParsedSkill
) -> prisma.models.SkillListing:
    listing = await prisma.models.SkillListing.prisma().upsert(
        where={"slug": entry["slug"]},
        data={
            "create": {"slug": entry["slug"], "hasApprovedVersion": True},
            "update": {"hasApprovedVersion": True, "isDeleted": False},
        },
        include={"ActiveVersion": True},
    )
    version = await _upsert_version(listing, entry, parsed)
    if listing.activeVersionId != version.id:
        listing = (
            await prisma.models.SkillListing.prisma().update(
                where={"id": listing.id},
                data={"activeVersionId": version.id},
                include={"ActiveVersion": True},
            )
            or listing
        )
    return listing


async def _upsert_version(
    listing: prisma.models.SkillListing,
    entry: StarterSkill,
    parsed: ParsedSkill,
) -> prisma.models.SkillListingVersion:
    """Rewrite the listing's live version in place.

    A starter skill is platform-authored, so there is no review to preserve and
    no creator waiting on a version history — editing the markdown should change
    what installers get, not add a row.
    """
    content: dict = {
        "name": parsed.name,
        "description": parsed.description,
        "body": parsed.body,
        "triggers": list(parsed.triggers),
        "categories": entry["categories"],
        "requiredProviders": entry["required_providers"],
        "sourceSkillSlug": entry["slug"],
        "isVerified": True,
        "isAvailable": True,
        "isDeleted": False,
        "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
    }
    existing = listing.ActiveVersion
    if existing is not None:
        updated = await prisma.models.SkillListingVersion.prisma().update(
            where={"id": existing.id}, data=content
        )
        if updated is not None:
            return updated
    return await prisma.models.SkillListingVersion.prisma().create(
        data={**content, "skillListingId": listing.id}
    )


def _load(slug: str) -> ParsedSkill:
    path = _CONTENT_DIR / f"{slug}.md"
    parsed = parse_skill_markdown(path.read_text(encoding="utf-8"))
    if parsed is None:
        raise ValueError(f"starter_skills/{slug}.md is not a valid SKILL.md")
    if parsed.name != slug:
        raise ValueError(
            f"starter_skills/{slug}.md declares name '{parsed.name}'; the "
            "frontmatter name is the installed skill's name and must match "
            "the listing slug"
        )
    return parsed


async def main() -> None:
    await database.connect()
    try:
        await seed_starter_skills()
    finally:
        await database.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
