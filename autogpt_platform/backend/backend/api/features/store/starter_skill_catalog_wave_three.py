"""Starter skills bundled by the third roster wave."""

from typing import TypedDict


class StarterSkill(TypedDict):
    slug: str
    categories: list[str]
    required_providers: list[str]


WAVE_THREE_STARTER_SKILLS: list[StarterSkill] = [
    {
        "slug": "support-getting-started",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "ticket-triage",
        "categories": ["support"],
        "required_providers": [],
    },
    {
        "slug": "support-reply-draft",
        "categories": ["support", "content"],
        "required_providers": [],
    },
    {
        "slug": "support-macro-library",
        "categories": ["support", "content"],
        "required_providers": [],
    },
    {
        "slug": "help-article-from-tickets",
        "categories": ["support", "content"],
        "required_providers": [],
    },
    {
        "slug": "bug-report-handoff",
        "categories": ["support", "development"],
        "required_providers": [],
    },
    {
        "slug": "refund-and-exception-brief",
        "categories": ["support", "finance"],
        "required_providers": [],
    },
    {
        "slug": "weekly-ticket-themes",
        "categories": ["support", "research"],
        "required_providers": [],
    },
    {
        "slug": "product-getting-started",
        "categories": ["research", "operations"],
        "required_providers": [],
    },
    {
        "slug": "feedback-synthesis",
        "categories": ["research"],
        "required_providers": [],
    },
    {
        "slug": "feature-request-triage",
        "categories": ["research", "operations"],
        "required_providers": [],
    },
    {
        "slug": "user-interview-guide",
        "categories": ["research"],
        "required_providers": [],
    },
    {
        "slug": "opportunity-brief",
        "categories": ["research", "content"],
        "required_providers": [],
    },
    {
        "slug": "product-requirements-draft",
        "categories": ["content", "development"],
        "required_providers": [],
    },
    {
        "slug": "roadmap-prioritisation",
        "categories": ["operations", "research"],
        "required_providers": [],
    },
    {
        "slug": "release-notes-draft",
        "categories": ["content"],
        "required_providers": [],
    },
    {
        "slug": "paid-ads-getting-started",
        "categories": ["marketing", "operations"],
        "required_providers": [],
    },
    {
        "slug": "campaign-structure-plan",
        "categories": ["marketing"],
        "required_providers": [],
    },
    {
        "slug": "ad-copy-variants",
        "categories": ["marketing", "content"],
        "required_providers": [],
    },
    {
        "slug": "landing-page-message-match",
        "categories": ["marketing", "content"],
        "required_providers": [],
    },
    {
        "slug": "wasted-spend-audit",
        "categories": ["marketing", "finance"],
        "required_providers": [],
    },
    {
        "slug": "budget-pacing-review",
        "categories": ["marketing", "finance"],
        "required_providers": [],
    },
    {
        "slug": "creative-test-readout",
        "categories": ["marketing", "research"],
        "required_providers": [],
    },
    {
        "slug": "paid-performance-report",
        "categories": ["marketing", "research"],
        "required_providers": [],
    },
    {
        "slug": "communications-getting-started",
        "categories": ["marketing", "operations"],
        "required_providers": [],
    },
    {
        "slug": "news-angle-and-key-messages",
        "categories": ["marketing", "content"],
        "required_providers": [],
    },
    {
        "slug": "press-release-draft",
        "categories": ["content", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "media-list-research",
        "categories": ["research", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "media-pitch-email",
        "categories": ["content", "marketing"],
        "required_providers": [],
    },
    {
        "slug": "launch-communications-plan",
        "categories": ["marketing", "operations"],
        "required_providers": [],
    },
    {
        "slug": "holding-statement-draft",
        "categories": ["content", "operations"],
        "required_providers": [],
    },
    {
        "slug": "spokesperson-briefing",
        "categories": ["content", "operations"],
        "required_providers": [],
    },
    {
        "slug": "code-quality-getting-started",
        "categories": ["development", "operations"],
        "required_providers": [],
    },
    {
        "slug": "pull-request-review",
        "categories": ["development"],
        "required_providers": [],
    },
    {
        "slug": "test-plan-draft",
        "categories": ["development"],
        "required_providers": [],
    },
    {
        "slug": "bug-reproduction-report",
        "categories": ["development", "support"],
        "required_providers": [],
    },
    {
        "slug": "flaky-test-triage",
        "categories": ["development"],
        "required_providers": [],
    },
    {
        "slug": "regression-risk-review",
        "categories": ["development", "research"],
        "required_providers": [],
    },
    {
        "slug": "release-readiness-checklist",
        "categories": ["development", "operations"],
        "required_providers": [],
    },
    {
        "slug": "incident-postmortem-draft",
        "categories": ["development", "operations"],
        "required_providers": [],
    },
    {
        "slug": "people-ops-getting-started",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "new-hire-onboarding-plan",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "handbook-policy-draft",
        "categories": ["operations", "content"],
        "required_providers": [],
    },
    {
        "slug": "one-to-one-agenda",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "performance-review-prep",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "engagement-survey-readout",
        "categories": ["operations", "research"],
        "required_providers": [],
    },
    {
        "slug": "offboarding-checklist",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "hr-escalation-brief",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "revops-getting-started",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "crm-field-audit",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "crm-duplicate-review",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "pipeline-stage-definitions",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "lead-routing-rules",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "sales-forecast-rollup",
        "categories": ["sales", "finance"],
        "required_providers": [],
    },
    {
        "slug": "lost-deal-analysis",
        "categories": ["sales", "research"],
        "required_providers": [],
    },
    {
        "slug": "crm-hygiene-report",
        "categories": ["sales", "operations"],
        "required_providers": [],
    },
    {
        "slug": "compliance-ops-getting-started",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "security-questionnaire-answers",
        "categories": ["operations", "sales"],
        "required_providers": [],
    },
    {
        "slug": "personal-data-map",
        "categories": ["operations", "research"],
        "required_providers": [],
    },
    {
        "slug": "subprocessor-register",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "dpa-checklist-review",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "policy-gap-review",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "data-subject-request-draft",
        "categories": ["operations", "support"],
        "required_providers": [],
    },
    {
        "slug": "compliance-escalation-brief",
        "categories": ["operations"],
        "required_providers": [],
    },
    {
        "slug": "executive-assistant-getting-started",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "inbox-triage",
        "categories": ["support", "operations"],
        "required_providers": ["google"],
    },
    {
        "slug": "reply-draft-in-your-voice",
        "categories": ["support", "content"],
        "required_providers": ["google"],
    },
    {
        "slug": "meeting-prep-brief",
        "categories": ["support", "operations"],
        "required_providers": ["google"],
    },
    {
        "slug": "meeting-follow-up-draft",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "calendar-conflict-review",
        "categories": ["support", "operations"],
        "required_providers": ["google"],
    },
    {
        "slug": "travel-plan",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
    {
        "slug": "weekly-priorities-review",
        "categories": ["support", "operations"],
        "required_providers": [],
    },
]
