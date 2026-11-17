"""Search types, categories and per-type request rules for Exa ``/search``."""

from datetime import datetime
from enum import Enum
from typing import Any

# exa_py used to add this when a request asked for no contents; the blocks keep
# it so saved workflows with default content settings still get page text.
LEGACY_DEFAULT_CONTENTS: dict[str, Any] = {"text": {"maxCharacters": 10000}}


class ExaSearchTypes(Enum):
    AUTO = "auto"
    LITE = "lite"
    DEEP_LITE = "deep-lite"
    DEEP = "deep"
    FAST = "fast"
    INSTANT = "instant"
    DEEP_REASONING = "deep-reasoning"
    NEURAL = "neural"
    KEYWORD = "keyword"


SEARCH_TYPE_DESCRIPTION = (
    "How Exa searches. "
    "'auto' (default): balanced quality and speed, ~1 s, $7 per 1k requests. "
    "'lite': cheapest and fastest, ~450 ms, $4 per 1k; cached content only and "
    "accepts only the query, result count, text/highlights, snapshot, category, "
    "domain and published-date filters, location, moderation, objective and user "
    "identifier. "
    "'deep-lite': lightweight research with synthesis, ~4 s, $12 per 1k. "
    "'deep': multi-step research with synthesis, 4-15 s, $12 per 1k; best with "
    "Output Schema. "
    "Legacy: 'fast' and 'instant' (use 'lite'), 'deep-reasoning' (use 'deep'; "
    "12-40 s, $15 per 1k), 'neural' and 'keyword' (no longer documented by Exa; "
    "use 'auto')."
)

DEEP_SEARCH_TYPES = frozenset(
    {ExaSearchTypes.DEEP_LITE, ExaSearchTypes.DEEP, ExaSearchTypes.DEEP_REASONING}
)


class ExaSearchCategories(Enum):
    COMPANY = "company"
    PEOPLE = "people"
    NEWS = "news"
    PUBLICATION = "publication"
    PERSONAL_SITE = "personal site"
    FINANCIAL_REPORT = "financial report"
    RESEARCH_PAPER = "research paper"
    LINKEDIN_PROFILE = "linkedin profile"
    PDF = "pdf"
    GITHUB = "github"
    TWEET = "tweet"


CATEGORY_DESCRIPTION = (
    "Focus the search on one kind of page: company, people, news, publication "
    "(scholarly papers, preprints and journal articles), personal site or "
    "financial report. 'company' and 'people' do not support published-date "
    "filters, and 'people' does not support Exclude Domains. "
    "Legacy: 'research paper' is sent as 'publication' and 'linkedin profile' as "
    "'people', the replacements Exa documents; 'pdf', 'github' and 'tweet' are "
    "being retired by Exa."
)

# Categories Exa replaced; the old value is translated at request time.
LEGACY_CATEGORY_REPLACEMENTS: dict[ExaSearchCategories, ExaSearchCategories] = {
    ExaSearchCategories.RESEARCH_PAPER: ExaSearchCategories.PUBLICATION,
    ExaSearchCategories.LINKEDIN_PROFILE: ExaSearchCategories.PEOPLE,
}


def category_value(category: ExaSearchCategories | None) -> str | None:
    """The category string to send, with legacy values replaced."""
    if category is None:
        return None
    return LEGACY_CATEGORY_REPLACEMENTS.get(category, category).value


# What 'lite' accepts (Exa rejects any other parameter that is set).
LITE_TOP_LEVEL_KEYS = frozenset(
    {
        "query",
        "type",
        "objective",
        "userIdentifier",
        "metadata",
        "category",
        "numResults",
        "contents",
        "userLocation",
        "includeDomains",
        "excludeDomains",
        "startPublishedDate",
        "endPublishedDate",
        "moderation",
        "compliance",
    }
)
LITE_CONTENTS_KEYS = frozenset({"highlights", "text", "snapshotAsOf"})

# Payload key -> block input label, for error messages.
PAYLOAD_KEY_LABELS = {
    "additionalQueries": "Additional Queries",
    "outputSchema": "Output Schema",
    "systemPrompt": "System Prompt",
    "includeText": "Include Text",
    "excludeText": "Exclude Text",
    "contents.summary": "Contents > Summary",
    "contents.extras": "Contents > Extras",
    "contents.subpages": "Contents > Subpages",
    "contents.subpageTarget": "Contents > Subpage Target",
    "contents.maxAgeHours": "Contents > Max Age Hours",
    "contents.livecrawl": "Contents > Livecrawl",
    "contents.livecrawlTimeout": "Contents > Livecrawl Timeout",
    "contents.context": "Contents > Context",
}


def check_search_type_rules(payload: dict[str, Any]) -> None:
    """Reject option combinations Exa would refuse for the chosen type."""
    search_type = payload.get("type")
    if payload.get("additionalQueries") and search_type not in {
        t.value for t in DEEP_SEARCH_TYPES
    }:
        raise ValueError(
            "Additional Queries only works with the deep search types "
            "('deep-lite', 'deep', 'deep-reasoning'). Change the type or clear it."
        )
    if search_type == ExaSearchTypes.LITE.value:
        check_lite_payload(payload)


def check_lite_payload(payload: dict[str, Any]) -> None:
    unsupported = [key for key in payload if key not in LITE_TOP_LEVEL_KEYS]
    contents = payload.get("contents") or {}
    unsupported += [
        f"contents.{key}" for key in contents if key not in LITE_CONTENTS_KEYS
    ]
    if not unsupported:
        return
    labels = ", ".join(PAYLOAD_KEY_LABELS.get(key, key) for key in unsupported)
    raise ValueError(
        f"The 'lite' search type does not support: {labels}. Clear these options "
        "or use the 'auto' search type."
    )


def iso_or_none(value: datetime | None) -> str | None:
    return value.isoformat() if value else None
