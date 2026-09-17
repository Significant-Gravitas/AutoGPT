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
        "slug": "bookkeeping-getting-started",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "expense-categorization",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "invoice-drafting-and-issue",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "accounts-receivable-follow-up",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "statement-reconciliation",
        "categories": ["finance"],
        "required_providers": [],
    },
    {
        "slug": "month-end-close-checklist",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "monthly-profit-and-loss-summary",
        "categories": ["finance", "research"],
        "required_providers": [],
    },
    {
        "slug": "bookkeeping-exception-escalation",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "investor-relations-getting-started",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "pitch-deck-review",
        "categories": ["finance", "content"],
        "required_providers": [],
    },
    {
        "slug": "fundraising-data-room-checklist",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "investor-targeting-and-research",
        "categories": ["finance", "research"],
        "required_providers": [],
    },
    {
        "slug": "fundraising-pipeline-review",
        "categories": ["finance", "sales"],
        "required_providers": [],
    },
    {
        "slug": "cap-table-hygiene",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "monthly-investor-update",
        "categories": ["finance", "content"],
        "required_providers": [],
    },
    {
        "slug": "board-and-investor-metrics-brief",
        "categories": ["finance", "research"],
        "required_providers": [],
    },
    {
        "slug": "kpi-analysis-getting-started",
        "categories": ["research"],
        "required_providers": [],
    },
    {
        "slug": "metric-definition-and-data-quality",
        "categories": ["research", "development"],
        "required_providers": [],
    },
    {
        "slug": "weekly-kpi-digest",
        "categories": ["research", "finance"],
        "required_providers": [],
    },
    {
        "slug": "metric-anomaly-detection",
        "categories": ["research", "development"],
        "required_providers": [],
    },
    {
        "slug": "metric-movement-analysis",
        "categories": ["research", "finance"],
        "required_providers": [],
    },
    {
        "slug": "cohort-and-retention-analysis",
        "categories": ["research", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "funnel-conversion-analysis",
        "categories": ["research", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "experiment-readout",
        "categories": ["research", "development"],
        "required_providers": [],
    },
    {
        "slug": "recruiting-getting-started",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "role-intake-and-job-description",
        "categories": ["operations", "content"],
        "required_providers": [],
    },
    {
        "slug": "hiring-rubric-design",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "resume-screening",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "interview-plan-and-scorecard",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "candidate-interview-debrief",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "candidate-rejection-email",
        "categories": ["operations", "content"],
        "required_providers": [],
    },
    {
        "slug": "candidate-offer-draft",
        "categories": ["operations", "content"],
        "required_providers": [],
    },
    {
        "slug": "procurement-getting-started",
        "categories": ["operations", "finance"],
        "required_providers": [],
    },
    {
        "slug": "vendor-requirements-brief",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "vendor-quote-comparison",
        "categories": ["operations", "finance"],
        "required_providers": [],
    },
    {
        "slug": "vendor-due-diligence",
        "categories": ["operations", "research"],
        "required_providers": [],
    },
    {
        "slug": "procurement-decision-memo",
        "categories": ["operations", "finance"],
        "required_providers": [],
    },
    {
        "slug": "contract-renewal-tracker",
        "categories": ["operations", "finance"],
        "required_providers": [],
    },
    {
        "slug": "vendor-performance-review",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "spend-anomaly-review",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "contract-ops-getting-started",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "nda-playbook-review",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "msa-playbook-review",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "contract-clause-comparison",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "contract-key-term-extraction",
        "categories": ["operations", "research"],
        "required_providers": [],
    },
    {
        "slug": "contract-deviation-triage",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "contract-obligation-tracker",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "counsel-escalation-brief",
        "categories": ["operations", "content"],
        "required_providers": [],
    },
    *[
        {"slug": slug, "categories": ["development"], "required_providers": []}
        for slug in (
            "dependency-security-getting-started",
            "dependency-inventory",
            "outdated-dependency-review",
            "vulnerability-triage",
            "cve-stack-relevance",
            "dependency-upgrade-plan",
            "dependency-upgrade-pr",
            "dependency-change-risk-review",
        )
    ],
    *[
        {"slug": slug, "categories": ["support"], "required_providers": []}
        for slug in (
            "customer-success-getting-started",
            "customer-onboarding-plan",
            "customer-health-score",
            "churn-risk-review",
            "renewal-readiness-review",
            "renewal-touchpoint-draft",
            "expansion-opportunity-brief",
            "customer-success-plan",
        )
    ],
    *[
        {"slug": slug, "categories": ["sales"], "required_providers": []}
        for slug in (
            "deal-desk-getting-started",
            "proposal-draft",
            "statement-of-work-draft",
            "pipeline-stage-aging-review",
            "deal-risk-review",
            "renewal-negotiation-brief",
            "pricing-and-terms-approval-brief",
            "proposal-quality-check",
        )
    ],
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
