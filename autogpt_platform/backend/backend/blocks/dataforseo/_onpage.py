"""
Parsing helpers for DataForSEO On-Page crawl results.
"""

from typing import Any
from urllib.parse import urlparse

from backend.sdk import BaseModel

# On-Page checks where a true value (or a non-zero page count) is a problem.
# The rest, such as is_https or seo_friendly_url, are true on a healthy page.
ISSUE_CHECKS = frozenset(
    {
        "no_title", "title_too_long", "title_too_short", "duplicate_title_tag",
        "irrelevant_title", "no_description", "irrelevant_description",
        "duplicate_meta_tags", "no_h1_tag", "is_4xx_code", "is_5xx_code",
        "is_broken", "is_redirect", "redirect_chain", "has_meta_refresh_redirect",
        "has_links_to_redirects", "canonical_to_broken", "canonical_to_redirect",
        "canonical_chain", "recursive_canonical", "is_orphan_page", "is_http",
        "https_to_http_links", "is_link_relation_conflict", "no_image_alt",
        "low_content_rate", "low_character_count", "low_readability_rate",
        "has_micromarkup_errors", "high_loading_time", "high_waiting_time",
        "large_page_size", "size_greater_than_3mb", "has_render_blocking_resources",
        "no_doctype", "no_encoding_meta_tag", "no_favicon", "deprecated_html_tags",
        "lorem_ipsum", "flash", "frame",
    }
)  # fmt: skip
# Page-level flags that sit beside the checks rather than inside them.
PAGE_FLAGS = ("duplicate_title", "duplicate_description", "duplicate_content", "broken_links", "broken_resources")  # fmt: skip


class AuditedPage(BaseModel):
    url: str
    status_code: int | None
    title: str
    description: str
    h1: list[str]
    canonical: str
    click_depth: int | None
    onpage_score: float | None
    issues: list[str]


def normalise_domain(target: str) -> str:
    """'https://www.example.com/shop' -> 'example.com', as On-Page expects."""
    target = target.strip()
    host = urlparse(target if "://" in target else f"//{target}").hostname or ""
    return host.removeprefix("www.")


def count_issues(metrics: dict[str, Any]) -> dict[str, int]:
    counts = {
        name: count
        for name, count in (metrics.get("checks") or {}).items()
        if name in ISSUE_CHECKS and count
    }
    counts |= {flag: metrics[flag] for flag in PAGE_FLAGS if metrics.get(flag)}
    return dict(sorted(counts.items(), key=lambda kv: kv[1], reverse=True))


def parse_page(page: dict[str, Any]) -> AuditedPage:
    meta = page.get("meta") or {}
    checks = page.get("checks") or {}
    issues = sorted(
        name for name, bad in checks.items() if name in ISSUE_CHECKS and bad
    )
    issues += [flag for flag in PAGE_FLAGS if page.get(flag)]
    return AuditedPage(
        url=page.get("url") or "",
        status_code=page.get("status_code"),
        title=meta.get("title") or "",
        description=meta.get("description") or "",
        h1=(meta.get("htags") or {}).get("h1") or [],
        canonical=meta.get("canonical") or "",
        click_depth=page.get("click_depth"),
        onpage_score=page.get("onpage_score"),
        issues=issues,
    )
