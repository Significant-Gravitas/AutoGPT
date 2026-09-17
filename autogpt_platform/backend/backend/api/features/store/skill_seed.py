"""Seed the platform-authored starter skill listings.

Run as ``python -m backend.api.features.store.skill_seed``, like the expert
roster seed. Idempotent: re-running updates the live version in place rather
than stacking a new one, so the catalogue can be edited by editing the
markdown.

Each listing's content is its ``SKILL.md`` under ``starter_skills/`` — either a
flat ``<slug>.md`` or a ``<slug>/`` package directory — parsed with the same
:func:`parse_skill_markdown` the copilot and the upload endpoint use, so a
starter skill cannot drift from the format an installed skill has. The seed
adds only what a listing needs beyond the file: its categories and the
integrations its instructions assume.
"""

import asyncio
import logging
import os
from pathlib import Path
from typing import TypedDict

import prisma.enums
import prisma.models

from backend.copilot.tools.skills import (
    ParsedSkill,
    SkillFile,
    SkillPackage,
    parse_skill_markdown,
    validate_package,
)
from backend.data import db as database

from .skill_submission_db import snapshot_version_files

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
    {
        "slug": "seo-content-brief",
        "categories": ["marketing", "content"],
        "required_providers": [],
    },
    {
        "slug": "on-page-seo-audit",
        "categories": ["marketing"],
        "required_providers": [],
    },
    {
        "slug": "content-repurposing",
        "categories": ["marketing", "content"],
        "required_providers": [],
    },
    {
        "slug": "competitor-teardown",
        "categories": ["research", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "icp-and-positioning",
        "categories": ["marketing", "research"],
        "required_providers": [],
    },
    {
        "slug": "lifecycle-email-map",
        "categories": ["marketing"],
        "required_providers": [],
    },
    {
        "slug": "email-deliverability-guardrails",
        "categories": ["marketing"],
        "required_providers": [],
    },
    {
        "slug": "account-health-and-qbrs",
        "categories": ["support", "sales"],
        "required_providers": [],
    },
    {
        "slug": "draft-the-reply",
        "categories": ["support", "content"],
        "required_providers": ["google"],
    },
    {
        "slug": "getting-started",
        "categories": ["support", "operations"],
        "required_providers": ["google", "notion", "slack"],
    },
    {
        "slug": "logistics-shipment-and-customs",
        "categories": ["operations", "support"],
        "required_providers": [],
    },
    {
        "slug": "onboarding-and-adoption",
        "categories": ["support", "sales"],
        "required_providers": [],
    },
    {
        "slug": "sensitive-data-safe-handling",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "technical-diagnostics-with-tools",
        "categories": ["support", "development"],
        "required_providers": [],
    },
    {
        "slug": "trust-and-safety-escalations",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "voice-of-customer-and-feedback",
        "categories": ["support", "research"],
        "required_providers": [],
    },
    {
        "slug": "billing-refunds-and-exceptions",
        "categories": ["support", "finance"],
        "required_providers": [],
    },
    {
        "slug": "enterprise-identity-sso-support",
        "categories": ["support", "development"],
        "required_providers": [],
    },
    {
        "slug": "help-center-answers-and-kb",
        "categories": ["support", "content"],
        "required_providers": ["google", "notion"],
    },
    {
        "slug": "mass-recovery-and-bulk-comms",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "own-to-closure",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "social-and-community-support",
        "categories": ["support", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "triage-and-prioritize",
        "categories": ["support", "operations"],
        "required_providers": ["google", "slack"],
    },
    {
        "slug": "upsell-and-retention-offers",
        "categories": ["sales", "support"],
        "required_providers": [],
    },
    {
        "slug": "workforce-and-capacity-planning",
        "categories": ["operations", "support"],
        "required_providers": [],
    },
    {
        "slug": "bpo-vendor-quality-ops",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "escalations-and-incidents",
        "categories": ["support", "operations"],
        "required_providers": ["slack"],
    },
    {
        "slug": "knowledge-centered-service",
        "categories": ["support", "content"],
        "required_providers": [],
    },
    {
        "slug": "live-channel-queue-operations",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "marketplace-two-sided-mediation",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "support-ops-improvement-program",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "troubleshoot-and-resolve",
        "categories": ["support", "operations"],
        "required_providers": ["google", "slack"],
    },
    {
        "slug": "vip-and-white-glove-care",
        "categories": ["support", "sales"],
        "required_providers": [],
    },
    {
        "slug": "compliance-and-regulated-support",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "fraud-and-chargeback-defense",
        "categories": ["support", "finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "orders-returns-and-warranty",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "member-benefits-and-claims",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "quality-csat-and-coaching",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "service-recovery-and-goodwill",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "travel-disruption-and-rebooking",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "voice-and-phone-support",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "draft-a-first-touch",
        "categories": ["content", "sales"],
        "required_providers": [],
    },
    {
        "slug": "multithread-and-stakeholder-maps",
        "categories": ["sales"],
        "required_providers": [],
    },
    {
        "slug": "retail-jbp-trade-and-sellout",
        "categories": ["finance", "operations", "sales"],
        "required_providers": [],
    },
    {
        "slug": "compliance-gated-deal-execution",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "credit-term-sheet-structuring",
        "categories": ["sales", "finance"],
        "required_providers": [],
    },
    {
        "slug": "draft-a-follow-up",
        "categories": ["sales", "content"],
        "required_providers": [],
    },
    {
        "slug": "enablement-playbooks-certification",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "marketplace-partner-revenue-growth",
        "categories": ["sales", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "media-plan-measure-optimize",
        "categories": ["sales", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "pipeline-review-and-forecast",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "regional-category-gtm-strategy",
        "categories": ["sales", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "cloud-commit-and-marketplace-selling",
        "categories": ["sales", "finance"],
        "required_providers": [],
    },
    {
        "slug": "discovery-and-qualification",
        "categories": ["sales"],
        "required_providers": [],
    },
    {
        "slug": "enterprise-deal-desk-close-plans",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "exec-engagement-and-sponsorship",
        "categories": ["sales"],
        "required_providers": [],
    },
    {
        "slug": "field-call-route-discipline",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "find-the-decision-makers",
        "categories": ["sales", "research"],
        "required_providers": [],
    },
    {
        "slug": "partner-and-channel-co-sell",
        "categories": ["sales"],
        "required_providers": [],
    },
    {
        "slug": "quarterback-the-deal-team",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "renewal-expansion-and-qbr",
        "categories": ["sales"],
        "required_providers": [],
    },
    {
        "slug": "rfp-and-competitive-bid-response",
        "categories": ["sales", "content"],
        "required_providers": [],
    },
    {
        "slug": "sales-team-leadership",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "showroom-fi-and-internet-bdc",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "territory-and-account-planning",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "voice-of-customer-loop",
        "categories": ["sales", "research"],
        "required_providers": [],
    },
    {
        "slug": "alliance-co-commercialization",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "build-the-target-list",
        "categories": ["sales", "research"],
        "required_providers": [],
    },
    {
        "slug": "business-case-and-roi-selling",
        "categories": ["sales", "finance"],
        "required_providers": [],
    },
    {
        "slug": "handle-a-reply",
        "categories": ["sales", "content"],
        "required_providers": [],
    },
    {
        "slug": "industrial-pursuit-tender-handover",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "next-step-and-handoff",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "objection-and-negotiation",
        "categories": ["sales"],
        "required_providers": [],
    },
    {
        "slug": "regulated-access-and-clinical-selling",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "research-an-account",
        "categories": ["sales", "research"],
        "required_providers": [],
    },
    {
        "slug": "sales-ops-coverage-and-quota",
        "categories": ["sales", "operations", "finance"],
        "required_providers": [],
    },
    {
        "slug": "signature-to-launch-and-account-ops",
        "categories": ["sales", "operations", "support"],
        "required_providers": [],
    },
]


async def seed_starter_skills() -> list[str]:
    """Upsert every starter listing. Returns the listing ids."""
    listing_ids = []
    for entry in STARTER_SKILLS:
        parsed, files = _load(entry["slug"])
        listing = await _upsert_listing(entry, parsed, files)
        listing_ids.append(listing.id)
        logger.info(
            f"Seeded starter skill '{entry['slug']}' (#{listing.id})"
            + (f" with {len(files)} package files" if files else "")
        )
    return listing_ids


async def _upsert_listing(
    entry: StarterSkill, parsed: ParsedSkill, files: list[SkillFile]
) -> prisma.models.SkillListing:
    listing = await prisma.models.SkillListing.prisma().upsert(
        where={"slug": entry["slug"]},
        data={
            "create": {"slug": entry["slug"], "hasApprovedVersion": True},
            "update": {"hasApprovedVersion": True, "isDeleted": False},
        },
        include={"ActiveVersion": True},
    )
    version = await _upsert_version(listing, entry, parsed, files)
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
    files: list[SkillFile],
) -> prisma.models.SkillListingVersion:
    """Rewrite the listing's live version in place, package and all.

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
        "isAvailable": True,
        "isDeleted": False,
        "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
    }
    existing = listing.ActiveVersion
    # One transaction: an install reads a version's instructions and its package
    # together, so neither may become visible without the other.
    async with database.transaction() as tx:
        if existing is not None:
            updated = await prisma.models.SkillListingVersion.prisma(tx).update(
                where={"id": existing.id}, data=content
            )
            if updated is not None:
                await snapshot_version_files(updated.id, files, tx)
                return updated
        created = await prisma.models.SkillListingVersion.prisma(tx).create(
            data={**content, "skillListingId": listing.id}
        )
        await snapshot_version_files(created.id, files, tx)
        return created


def _load(slug: str) -> tuple[ParsedSkill, list[SkillFile]]:
    """A starter skill's root and its package files, from either layout: a
    flat ``<slug>.md``, or a ``<slug>/`` directory whose ``SKILL.md`` sits
    beside the resources it references."""
    directory = _CONTENT_DIR / slug
    is_package = directory.is_dir()
    root = directory / "SKILL.md" if is_package else _CONTENT_DIR / f"{slug}.md"
    named = f"starter_skills/{root.relative_to(_CONTENT_DIR)}"
    text = root.read_text(encoding="utf-8")
    parsed = parse_skill_markdown(text)
    if parsed is None:
        raise ValueError(f"{named} is not a valid SKILL.md")
    if parsed.name != slug:
        raise ValueError(
            f"{named} declares name '{parsed.name}'; the frontmatter name is "
            "the installed skill's name and must match the listing slug"
        )
    files = _package_files(directory) if is_package else []
    validate_package(SkillPackage(skill_md=text, files=files))
    return parsed, files


def _package_files(directory: Path) -> list[SkillFile]:
    """Every file beside the directory's ``SKILL.md``, by its relative path.
    The executable bit rides along so a seeded script stays runnable."""
    root = directory / "SKILL.md"
    return [
        SkillFile(
            relative_path=path.relative_to(directory).as_posix(),
            content=path.read_bytes(),
            is_executable=os.access(path, os.X_OK),
        )
        for path in sorted(directory.rglob("*"))
        if path.is_file() and path != root
    ]


async def main() -> None:
    await database.connect()
    try:
        await seed_starter_skills()
    finally:
        await database.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
