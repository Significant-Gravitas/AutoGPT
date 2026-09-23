"""Seed the platform-authored skill listings from the skills catalog.

Run as ``python -m backend.api.features.store.skill_seed``, like the expert
roster seed. Idempotent: re-running rewrites each listing's live version in
place rather than stacking a new one, so the marketplace is edited by editing
the catalog and seeding again.

The catalog is the private ``Significant-Gravitas/skills-catalog`` repo. Its
``catalog.yml`` names every listing with its categories and the integrations
its instructions assume; ``skills/<slug>/`` holds the SKILL.md beside the
references it points at. Each SKILL.md is parsed with the same
:func:`parse_skill_markdown` the copilot and the upload endpoint use, so a
seeded skill cannot drift from the format an installed skill has, and the
package files go through the same :func:`validate_package` an upload does.

Environment:

``SKILLS_CATALOG_PATH``
    A local checkout to seed from instead of GitHub.
``SKILLS_CATALOG_REPO`` / ``SKILLS_CATALOG_REF``
    The repo (``owner/name``) and branch, tag or commit to download.
``SKILLS_CATALOG_TOKEN``
    A GitHub token that can read the repo; ``GITHUB_TOKEN`` is the fallback.
"""

import asyncio
import io
import logging
import os
import tarfile
import tempfile
from datetime import timedelta
from pathlib import Path

import httpx
import prisma
import prisma.enums
import prisma.models
import yaml

from backend.copilot.tools.skills import (
    ParsedSkill,
    SkillFile,
    SkillPackage,
    SkillPackageError,
    _validate_name,
    parse_skill_markdown,
    validate_package,
    validate_skill_content,
)
from backend.data import db as database

from .categories import validate_canonical_categories
from .skill_submission_db import snapshot_version_files
from .starter_skill_catalog_wave_three import WAVE_THREE_STARTER_SKILLS, StarterSkill

logger = logging.getLogger(__name__)

DEFAULT_CATALOG_REPO = "Significant-Gravitas/skills-catalog"
DEFAULT_CATALOG_REF = "main"
CATALOG_FILE = "catalog.yml"
SKILLS_DIR = "skills"
_CONTENT_DIR = Path(__file__).parent / "starter_skills"
# One transaction for the whole catalog, so a failure leaves the marketplace
# as it was. The default 30s covers a handful of listings; a full catalog is
# several queries per listing against a remote database.
SEED_TRANSACTION_TIMEOUT = timedelta(minutes=10)


CatalogEntry = StarterSkill

STARTER_SKILLS: list[CatalogEntry] = [
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
        "required_providers": ["reddit"],
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
        "slug": "robin-getting-started",
        "categories": ["support", "operations"],
        "required_providers": [
            "google",
            "mcp_gong",
            "mcp_granola",
            "mcp_linear",
            "notion",
            "slack",
        ],
    },
    {
        "slug": "max-getting-started",
        "categories": ["sales"],
        "required_providers": [
            "google",
            "hubspot",
            "mcp_gong",
            "mcp_granola",
            "notion",
            "slack",
        ],
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
        "required_providers": ["mcp_linear", "slack"],
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
    {
        "slug": "anika-getting-started",
        "categories": ["sales", "operations"],
        "required_providers": ["google", "hubspot", "mcp_granola", "notion", "slack"],
    },
    {
        "slug": "define-the-partner-icp",
        "categories": ["sales", "research"],
        "required_providers": [],
    },
    {
        "slug": "source-and-qualify-partners",
        "categories": ["sales", "research"],
        "required_providers": [],
    },
    {
        "slug": "partner-first-touch-outreach",
        "categories": ["sales", "content"],
        "required_providers": ["google"],
    },
    {
        "slug": "structure-the-partner-agreement",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "onboard-and-enable-partners",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "run-the-partner-co-sell-cadence",
        "categories": ["sales"],
        "required_providers": [],
    },
    {
        "slug": "track-partner-pipeline",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "prep-the-partner-qbr",
        "categories": ["sales", "operations"],
        "required_providers": ["google"],
    },
    {
        "slug": "handle-partner-conflict-and-churn",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "design-the-partner-program",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "map-the-partner-ecosystem",
        "categories": ["research", "sales"],
        "required_providers": [],
    },
    {
        "slug": "plan-the-multi-year-partnership",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "govern-the-strategic-alliance",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "model-the-partnership-commercials",
        "categories": ["sales", "finance"],
        "required_providers": [],
    },
    {
        "slug": "scope-the-tech-partnership",
        "categories": ["sales", "development"],
        "required_providers": [],
    },
    {
        "slug": "run-the-partner-marketing-engine",
        "categories": ["marketing", "sales"],
        "required_providers": [],
    },
    {
        "slug": "scale-the-partner-channel",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "run-hyperscaler-marketplace-co-sell",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "run-partner-strategy-and-operations",
        "categories": ["operations", "sales"],
        "required_providers": [],
    },
    {
        "slug": "assure-partner-led-delivery",
        "categories": ["operations", "support"],
        "required_providers": [],
    },
    {
        "slug": "manage-partner-renewals-and-exits",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "build-partner-academies-at-scale",
        "categories": ["operations", "content"],
        "required_providers": [],
    },
    {
        "slug": "orchestrate-multi-party-partner-bids",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "run-regulated-partnership-motions",
        "categories": ["operations", "sales"],
        "required_providers": [],
    },
    {
        "slug": "source-partners-via-investor-ecosystems",
        "categories": ["sales", "research"],
        "required_providers": [],
    },
    {
        "slug": "build-data-and-r-d-alliances",
        "categories": ["research", "operations"],
        "required_providers": [],
    },
    {
        "slug": "run-creator-and-affiliate-partner-programs",
        "categories": ["marketing", "operations"],
        "required_providers": [],
    },
    {
        "slug": "run-brand-oem-and-supply-partnerships",
        "categories": ["marketing", "operations"],
        "required_providers": [],
    },
    {
        "slug": "set-board-level-alliance-strategy",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "own-the-alliance-pnl",
        "categories": ["finance", "operations"],
        "required_providers": [],
    },
    {
        "slug": "run-global-partner-executive-councils",
        "categories": ["operations", "sales"],
        "required_providers": [],
    },
    {
        "slug": "drive-alliance-ma-and-strategic-investments",
        "categories": ["finance", "research"],
        "required_providers": [],
    },
    {
        "slug": "build-partner-led-category-creation",
        "categories": ["marketing", "sales"],
        "required_providers": [],
    },
    {
        "slug": "daniel-getting-started",
        "categories": ["finance", "operations"],
        "required_providers": ["google", "hubspot", "notion", "slack", "stripe"],
    },
    {
        "slug": "budget-vs-actuals-and-reforecast",
        "categories": ["finance"],
        "required_providers": ["google"],
    },
    {
        "slug": "variance-and-flux-analysis",
        "categories": ["finance", "research"],
        "required_providers": ["google"],
    },
    {
        "slug": "cash-treasury-and-fx",
        "categories": ["finance"],
        "required_providers": ["google"],
    },
    {
        "slug": "close-controls-and-accounting",
        "categories": ["finance", "operations"],
        "required_providers": ["google"],
    },
    {
        "slug": "unit-economics-and-roi",
        "categories": ["finance", "research"],
        "required_providers": [],
    },
    {
        "slug": "saas-gtm-finance",
        "categories": ["finance", "sales"],
        "required_providers": [],
    },
    {
        "slug": "deal-economics-and-pricing-guardrails",
        "categories": ["finance", "sales"],
        "required_providers": [],
    },
    {
        "slug": "finance-board-and-investor-reporting",
        "categories": ["finance", "content"],
        "required_providers": ["google"],
    },
    {
        "slug": "automate-finance-reporting",
        "categories": ["finance", "operations"],
        "required_providers": ["google"],
    },
    {
        "slug": "alex-getting-started",
        "categories": ["development", "research"],
        "required_providers": ["github", "google", "mcp_linear", "notion", "slack"],
    },
    {
        "slug": "product-roadmap-and-prioritization",
        "categories": ["development", "operations"],
        "required_providers": [],
    },
    {
        "slug": "product-prd-and-acceptance-criteria",
        "categories": ["development"],
        "required_providers": [],
    },
    {
        "slug": "product-discovery-and-user-research",
        "categories": ["research"],
        "required_providers": [],
    },
    {
        "slug": "product-metrics-and-instrumentation",
        "categories": ["research", "development"],
        "required_providers": [],
    },
    {
        "slug": "product-strategy-and-bets",
        "categories": ["research", "development"],
        "required_providers": [],
    },
    {
        "slug": "product-experiment-design",
        "categories": ["research", "development"],
        "required_providers": [],
    },
    {
        "slug": "product-ai-feature-scoping-and-evals",
        "categories": ["development"],
        "required_providers": [],
    },
    {
        "slug": "product-launch-plan",
        "categories": ["marketing", "operations"],
        "required_providers": [],
    },
    {
        "slug": "product-market-and-competitor-read",
        "categories": ["research", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "product-exec-briefing",
        "categories": ["operations", "development"],
        "required_providers": [],
    },
    {
        "slug": "sofia-getting-started",
        "categories": ["operations"],
        "required_providers": [
            "google",
            "mcp_granola",
            "mcp_linear",
            "notion",
            "slack",
        ],
    },
    {
        "slug": "role-intake-and-scorecard",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "job-description-drafting",
        "categories": ["operations", "content"],
        "required_providers": [],
    },
    {
        "slug": "candidate-sourcing-strategy",
        "categories": ["operations", "research"],
        "required_providers": ["github"],
    },
    {
        "slug": "passive-candidate-outreach",
        "categories": ["operations", "content"],
        "required_providers": ["google"],
    },
    {
        "slug": "interview-kit-design",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "interview-coordination",
        "categories": ["operations"],
        "required_providers": ["google"],
    },
    {
        "slug": "hiring-debrief-and-decision",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "job-offer-and-close-plan",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "hiring-pipeline-analytics",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "james-getting-started",
        "categories": ["operations"],
        "required_providers": ["google", "mcp_linear", "notion", "slack"],
    },
    {
        "slug": "ops-run-the-operating-rhythm",
        "categories": ["operations"],
        "required_providers": ["google"],
    },
    {
        "slug": "ops-scorecard-and-kpis",
        "categories": ["operations"],
        "required_providers": ["google"],
    },
    {
        "slug": "ops-write-an-sop",
        "categories": ["operations"],
        "required_providers": ["notion"],
    },
    {
        "slug": "ops-map-and-improve-a-process",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "ops-automate-a-workflow",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "ops-vendor-and-procurement",
        "categories": ["operations", "finance"],
        "required_providers": ["google"],
    },
    {
        "slug": "ops-capacity-and-headcount-plan",
        "categories": ["operations"],
        "required_providers": ["google"],
    },
    {
        "slug": "ops-controls-and-escalations",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "ops-govern-a-program",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "maya-getting-started",
        "categories": ["marketing", "content"],
        "required_providers": ["google", "notion", "slack", "hubspot"],
    },
    {
        "slug": "content-brief-writer-handoff",
        "categories": ["marketing", "content"],
        "required_providers": [],
    },
    {
        "slug": "editorial-calendar-ops",
        "categories": ["marketing", "content"],
        "required_providers": ["google", "notion"],
    },
    {
        "slug": "messaging-and-tone-matrix",
        "categories": ["marketing", "content"],
        "required_providers": [],
    },
    {
        "slug": "campaign-brief-and-asset-plan",
        "categories": ["marketing"],
        "required_providers": [],
    },
    {
        "slug": "channel-draft-shapes",
        "categories": ["content", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "nurture-sequence-build-and-readout",
        "categories": ["marketing"],
        "required_providers": [],
    },
    {
        "slug": "weekly-marketing-read",
        "categories": ["marketing", "research"],
        "required_providers": [],
    },
    {
        "slug": "zara-getting-started",
        "categories": ["marketing", "sales"],
        "required_providers": [
            "apollo",
            "google",
            "hubspot",
            "mcp_amplitude",
            "mcp_gong",
            "mcp_linear",
            "slack",
            "stripe",
        ],
    },
    {
        "slug": "positioning-and-messaging",
        "categories": ["marketing"],
        "required_providers": ["google", "mcp_gong"],
    },
    {
        "slug": "icp-and-segmentation",
        "categories": ["marketing", "sales"],
        "required_providers": ["apollo", "hubspot"],
    },
    {
        "slug": "pricing-and-packaging",
        "categories": ["marketing", "finance"],
        "required_providers": ["google", "stripe"],
    },
    {
        "slug": "commercial-launch-strategy",
        "categories": ["marketing"],
        "required_providers": ["google", "mcp_linear"],
    },
    {
        "slug": "competitive-intelligence",
        "categories": ["marketing", "research"],
        "required_providers": ["google", "mcp_gong"],
    },
    {
        "slug": "gtm-performance-diagnostics",
        "categories": ["marketing", "sales"],
        "required_providers": ["google", "hubspot", "mcp_amplitude"],
    },
    {
        "slug": "gtm-planning-and-market-entry",
        "categories": ["marketing", "sales"],
        "required_providers": ["google"],
    },
    {
        "slug": "opportunity-sizing-and-business-case",
        "categories": ["marketing", "finance"],
        "required_providers": ["apollo", "google"],
    },
    {
        "slug": "sales-marketing-alignment-and-sla",
        "categories": ["marketing", "sales"],
        "required_providers": ["hubspot"],
    },
    {
        "slug": "field-enablement-content",
        "categories": ["sales", "marketing"],
        "required_providers": ["google", "mcp_gong"],
    },
    {
        "slug": "vertical-industry-plays",
        "categories": ["marketing", "sales"],
        "required_providers": [],
    },
    {
        "slug": "developer-and-api-motion",
        "categories": ["marketing", "development"],
        "required_providers": ["mcp_amplitude"],
    },
    *WAVE_THREE_STARTER_SKILLS,
]


async def seed_catalog_skills(catalog_dir: Path | None = None) -> list[str]:
    """Upsert every catalog listing. Returns the listing ids.

    An explicit *catalog_dir* seeds only that tree. A normal run downloads the
    catalog, or reads ``SKILLS_CATALOG_PATH``, and also keeps checked-in skills
    that the expert roster needs until the catalog holds the same slug.
    """
    if catalog_dir is not None:
        return await _seed_catalog(catalog_dir, include_starters=False)

    local = os.environ.get("SKILLS_CATALOG_PATH")
    if local:
        return await _seed_catalog(Path(local), include_starters=True)
    with tempfile.TemporaryDirectory() as tmp:
        return await _seed_catalog(_download_catalog(Path(tmp)), include_starters=True)


async def _seed_catalog(root: Path, *, include_starters: bool) -> list[str]:
    entries = load_catalog(root)
    loaded = [(entry, *_load(root, entry)) for entry in entries]
    if not include_starters:
        return await _seed_loaded(loaded)

    catalog_slugs = {entry["slug"] for entry in entries}
    loaded += [
        (entry, *_load_starter(entry))
        for entry in STARTER_SKILLS
        if entry["slug"] not in catalog_slugs
    ]
    return await _seed_loaded(loaded)


async def seed_starter_skills() -> list[str]:
    """Upsert the checked-in skills needed by the expert roster."""
    loaded = [(entry, *_load_starter(entry)) for entry in STARTER_SKILLS]
    return await _seed_loaded(loaded)


# Starter slugs that shipped and were then renamed or folded away. The seed
# never deletes a listing, so a retired slug is delisted: hidden from the hub
# and unavailable to new installs, while copies already installed stay put.
RETIRED_STARTER_SLUGS: list[str] = [
    # Renamed to max-getting-started when the senior sales package was folded
    # into Max.
    "blake-getting-started",
]


async def _delist_retired_starters(tx: prisma.Prisma) -> int:
    listings = await prisma.models.SkillListing.prisma(tx).find_many(
        where={
            "slug": {"in": RETIRED_STARTER_SLUGS},
            "isDeleted": False,
            "owningUserId": None,
            "owningOrgId": None,
        }
    )
    for listing in listings:
        await prisma.models.SkillListing.prisma(tx).update(
            where={"id": listing.id}, data={"isDeleted": True}
        )
        if listing.activeVersionId is not None:
            await prisma.models.SkillListingVersion.prisma(tx).update(
                where={"id": listing.activeVersionId},
                data={"isAvailable": False},
            )
    return len(listings)


async def _seed_loaded(
    loaded: list[tuple[CatalogEntry, ParsedSkill, list[SkillFile]]],
) -> list[str]:
    """Write a set whose packages have all been loaded and checked."""
    listing_ids = []
    async with database.transaction(timeout=SEED_TRANSACTION_TIMEOUT) as tx:
        for entry, parsed, files in loaded:
            listing = await _upsert_listing(tx, entry, parsed, files)
            listing_ids.append(listing.id)
            logger.info(
                f"Seeded skill '{entry['slug']}' (#{listing.id})"
                + (f" with {len(files)} package files" if files else "")
            )
        delisted = await _delist_retired_starters(tx)
        if delisted:
            logger.info(
                f"Delisted {delisted} retired starter(s): {RETIRED_STARTER_SLUGS}"
            )
    return listing_ids


def load_catalog(root: Path) -> list[CatalogEntry]:
    """The catalog's entries, each with a canonical category set."""
    raw = yaml.safe_load((root / CATALOG_FILE).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"{CATALOG_FILE} must contain a mapping")
    skills = raw.get("skills") or []
    if not isinstance(skills, list):
        raise ValueError(f"{CATALOG_FILE}: skills must be a list")
    entries: list[CatalogEntry] = []
    seen: set[str] = set()
    for item in skills:
        if not isinstance(item, dict):
            raise ValueError(f"{CATALOG_FILE}: every skill must be a mapping")
        slug = str(item.get("slug") or "").strip()
        if not slug:
            raise ValueError(f"{CATALOG_FILE}: an entry has no slug")
        if error := _validate_name(slug):
            raise ValueError(f"{CATALOG_FILE}: '{slug}' {error}")
        if slug in seen:
            raise ValueError(f"{CATALOG_FILE}: '{slug}' is listed twice")
        seen.add(slug)
        raw_categories = item.get("categories")
        categories = [] if raw_categories is None else raw_categories
        if not isinstance(categories, list) or not all(
            isinstance(category, str) for category in categories
        ):
            raise ValueError(
                f"{CATALOG_FILE}: '{slug}' categories must be a list of strings"
            )
        raw_required_providers = item.get("required_providers")
        required_providers = (
            [] if raw_required_providers is None else raw_required_providers
        )
        if not isinstance(required_providers, list) or not all(
            isinstance(provider, str) for provider in required_providers
        ):
            raise ValueError(
                f"{CATALOG_FILE}: '{slug}' required_providers must be a list of strings"
            )
        entries.append(
            CatalogEntry(
                slug=slug,
                categories=validate_canonical_categories(categories),
                required_providers=required_providers,
            )
        )
    if not entries:
        raise ValueError(f"{CATALOG_FILE} lists no skills")
    return entries


async def _upsert_listing(
    tx: prisma.Prisma,
    entry: CatalogEntry,
    parsed: ParsedSkill,
    files: list[SkillFile],
) -> prisma.models.SkillListing:
    listing = await prisma.models.SkillListing.prisma(tx).find_unique(
        where={"slug": entry["slug"]}, include={"ActiveVersion": True}
    )
    if listing is None:
        listing = await prisma.models.SkillListing.prisma(tx).create(
            data={"slug": entry["slug"], "hasApprovedVersion": True},
            include={"ActiveVersion": True},
        )
    else:
        if listing.owningUserId is not None or listing.owningOrgId is not None:
            raise ValueError(
                f"catalog skill '{entry['slug']}' conflicts with an owned listing"
            )
        listing = (
            await prisma.models.SkillListing.prisma(tx).update(
                where={"id": listing.id},
                data={"hasApprovedVersion": True, "isDeleted": False},
                include={"ActiveVersion": True},
            )
            or listing
        )
    version = await _upsert_version(tx, listing, entry, parsed, files)
    if listing.activeVersionId != version.id:
        listing = (
            await prisma.models.SkillListing.prisma(tx).update(
                where={"id": listing.id},
                data={"activeVersionId": version.id},
                include={"ActiveVersion": True},
            )
            or listing
        )
    return listing


async def _upsert_version(
    tx: prisma.Prisma,
    listing: prisma.models.SkillListing,
    entry: CatalogEntry,
    parsed: ParsedSkill,
    files: list[SkillFile],
) -> prisma.models.SkillListingVersion:
    """Rewrite the listing's live version in place, package and all.

    A catalog skill is platform-authored, so there is no review to preserve
    and no creator waiting on a version history — editing the catalog should
    change what installers get, not add a row.
    """
    metadata = parsed.extra.get("metadata") or {}
    if not isinstance(metadata, dict):
        metadata = {}
    license_value = parsed.extra.get("license")
    content: dict = {
        "name": parsed.name,
        "description": parsed.description,
        "body": parsed.body,
        "triggers": list(parsed.triggers),
        "categories": entry["categories"],
        "requiredProviders": entry["required_providers"],
        "sourceSkillSlug": entry["slug"],
        "sourceRepo": _attribution_value(parsed, metadata, "source"),
        "sourceUrl": _attribution_value(parsed, metadata, "source_url"),
        "license": _optional_str(license_value),
        "isAvailable": True,
        "isDeleted": False,
        "submissionStatus": prisma.enums.SubmissionStatus.APPROVED,
    }
    existing = listing.ActiveVersion
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


def _optional_str(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _attribution_value(
    parsed: ParsedSkill, metadata: dict[object, object], key: str
) -> str | None:
    return _optional_str(metadata.get(key)) or _optional_str(parsed.extra.get(key))


def _load(root: Path, entry: CatalogEntry) -> tuple[ParsedSkill, list[SkillFile]]:
    """A catalog skill's SKILL.md and the files beside it, validated as the
    package an installer will receive."""
    slug = entry["slug"]
    directory = root / SKILLS_DIR / slug
    skill_md = directory / "SKILL.md"
    named = f"{SKILLS_DIR}/{slug}/SKILL.md"
    if not skill_md.is_file():
        raise ValueError(f"{named} is missing")
    text = skill_md.read_text(encoding="utf-8")
    parsed = parse_skill_markdown(text)
    if parsed is None:
        raise ValueError(f"{named} is not a valid SKILL.md")
    if parsed.name != slug:
        raise ValueError(
            f"{named} declares name '{parsed.name}'; the frontmatter name is "
            "the installed skill's name and must match the catalog slug"
        )
    validate_skill_content(parsed.description, parsed.body, parsed.triggers)
    files = _package_files(directory)
    validate_package(SkillPackage(skill_md=text, files=files))
    return parsed, files


def _load_starter(entry: CatalogEntry) -> tuple[ParsedSkill, list[SkillFile]]:
    slug = entry["slug"]
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
    validate_skill_content(parsed.description, parsed.body, parsed.triggers)
    files = _package_files(directory) if is_package else []
    validate_package(SkillPackage(skill_md=text, files=files))
    return parsed, files


def _package_files(directory: Path) -> list[SkillFile]:
    """Every file beside the directory's ``SKILL.md``, by its relative path.
    The executable bit rides along so a seeded script stays runnable."""
    root = directory / "SKILL.md"
    files = []
    for path in sorted(directory.rglob("*")):
        relative_path = path.relative_to(directory).as_posix()
        if path.is_symlink():
            raise SkillPackageError(f"file '{relative_path}' may not be a symlink")
        if path.is_file() and path != root:
            files.append(
                SkillFile(
                    relative_path=relative_path,
                    content=path.read_bytes(),
                    is_executable=os.access(path, os.X_OK),
                )
            )
    return files


def _download_catalog(into: Path) -> Path:
    """Fetch the catalog repo's tarball from GitHub and unpack it under *into*,
    returning the checkout root."""
    repo = os.environ.get("SKILLS_CATALOG_REPO") or DEFAULT_CATALOG_REPO
    ref = os.environ.get("SKILLS_CATALOG_REF") or DEFAULT_CATALOG_REF
    token = os.environ.get("SKILLS_CATALOG_TOKEN") or os.environ.get("GITHUB_TOKEN")
    if not token:
        raise RuntimeError(
            "SKILLS_CATALOG_TOKEN (or GITHUB_TOKEN) is required to download "
            f"{repo}; set SKILLS_CATALOG_PATH to seed from a local checkout"
        )
    logger.info(f"Downloading {repo}@{ref}")
    response = httpx.get(
        f"https://api.github.com/repos/{repo}/tarball/{ref}",
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "User-Agent": "autogpt-platform-skill-seed",
        },
        follow_redirects=True,
        timeout=60,
    )
    response.raise_for_status()
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        _extract_catalog_archive(tar, into)
    # GitHub wraps the tree in one "<owner>-<repo>-<sha>" directory.
    roots = [p for p in into.iterdir() if p.is_dir()]
    if len(roots) != 1:
        raise RuntimeError(f"unexpected tarball layout for {repo}: {roots}")
    return roots[0]


def _extract_catalog_archive(archive: tarfile.TarFile, into: Path) -> None:
    """Extract with the data filter, including on Python versions before its backport."""
    if hasattr(tarfile, "data_filter"):
        archive.extractall(into, filter="data")
        return

    root = into.resolve()
    members = archive.getmembers()
    for member in members:
        target = (into / member.name).resolve()
        if not (member.isfile() or member.isdir()) or (
            target != root and root not in target.parents
        ):
            raise RuntimeError(f"unsafe catalog archive member: {member.name}")
    archive.extractall(into, members=members)


async def main() -> None:
    await database.connect()
    try:
        await seed_catalog_skills()
    finally:
        await database.disconnect()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())
