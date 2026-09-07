"""The canonical marketplace category set, and how legacy free-text values fold into it."""

from enum import Enum

from backend.util.settings import Settings

settings = Settings()


class StoreCategory(str, Enum):
    MARKETING = "marketing"
    SALES = "sales"
    FINANCE = "finance"
    SUPPORT = "support"
    OPERATIONS = "operations"
    RESEARCH = "research"
    CONTENT = "content"
    DEVELOPMENT = "development"


CATEGORY_LABELS: dict[StoreCategory, str] = {
    StoreCategory.MARKETING: "Marketing",
    StoreCategory.SALES: "Sales",
    StoreCategory.FINANCE: "Finance",
    StoreCategory.SUPPORT: "Support",
    StoreCategory.OPERATIONS: "Operations",
    StoreCategory.RESEARCH: "Research",
    StoreCategory.CONTENT: "Content",
    StoreCategory.DEVELOPMENT: "Development",
}

CATEGORY_DESCRIPTIONS: dict[StoreCategory, str] = {
    StoreCategory.MARKETING: "Campaigns, SEO, social media, brand and audience growth",
    StoreCategory.SALES: "Lead generation, prospecting, outreach and CRM work",
    StoreCategory.FINANCE: "Accounting, invoicing, budgeting, reporting and analysis",
    StoreCategory.SUPPORT: "Customer service, ticket handling and personal assistance",
    StoreCategory.OPERATIONS: "Internal process automation, scheduling, HR and admin",
    StoreCategory.RESEARCH: "Gathering, extracting and analysing information or data",
    StoreCategory.CONTENT: "Writing, editing, design and media production",
    StoreCategory.DEVELOPMENT: "Software engineering, DevOps, testing and integrations",
}

# Only mappings with one defensible target live here; anything ambiguous is left
# for the classifier. "business" spanned Finance, Sales and Support in the
# existing data, so folding it by rule would have mislabelled most of it.
CATEGORY_ALIASES: dict[str, StoreCategory] = {
    "writing": StoreCategory.CONTENT,
    "creative": StoreCategory.CONTENT,
    "content & writing": StoreCategory.CONTENT,
    "writing & content": StoreCategory.CONTENT,
    "creative & design": StoreCategory.CONTENT,
    "design": StoreCategory.CONTENT,
    "marketing & seo": StoreCategory.MARKETING,
    "seo": StoreCategory.MARKETING,
    "research & learning": StoreCategory.RESEARCH,
    "development": StoreCategory.DEVELOPMENT,
    "engineering": StoreCategory.DEVELOPMENT,
    "devops": StoreCategory.DEVELOPMENT,
    "productivity": StoreCategory.OPERATIONS,
    "automation": StoreCategory.OPERATIONS,
    "customer support": StoreCategory.SUPPORT,
    "customer service": StoreCategory.SUPPORT,
    "sales & crm": StoreCategory.SALES,
    "business & finance": StoreCategory.FINANCE,
    "accounting": StoreCategory.FINANCE,
}


def normalize_category(value: str) -> StoreCategory | None:
    """Fold one free-text category onto the canonical set, or None if it doesn't fit."""
    key = value.strip().lower()
    if not key:
        return None
    try:
        return StoreCategory(key)
    except ValueError:
        return CATEGORY_ALIASES.get(key)


def normalize_categories(values: list[str]) -> list[StoreCategory]:
    """Fold a listing's categories, dropping unmappable ones and preserving order."""
    seen: list[StoreCategory] = []
    for value in values:
        category = normalize_category(value)
        if category and category not in seen:
            seen.append(category)
    return seen


def category_filter_values(category: str | None) -> list[str] | None:
    """Stored category strings a listing query should match, or None for no filter.

    With no category asked for, the default view still narrows to the canonical
    set once the backfill has run and the setting is turned on.
    """
    if category:
        return category_match_values(category)
    if settings.config.marketplace_require_canonical_category:
        return all_category_match_values()
    return None


def category_match_values(category: str) -> list[str]:
    """The stored category strings a filter for ``category`` should match.

    Includes the legacy aliases so a filter works both before and after the
    backfill has folded a listing onto the canonical set.
    """
    canonical = normalize_category(category)
    if canonical is None:
        return [category]
    return [canonical.value] + [
        alias for alias, target in CATEGORY_ALIASES.items() if target is canonical
    ]


def all_category_match_values() -> list[str]:
    """Every stored string that counts as "has a canonical category"."""
    return [category.value for category in StoreCategory] + list(CATEGORY_ALIASES)


def validate_canonical_categories(values: list[str]) -> list[str]:
    """Reusable field validator: a listing must carry at least one canonical category."""
    normalized = normalize_categories(values)
    if not normalized:
        raise ValueError(
            "at least one category is required, from: "
            + ", ".join(category.value for category in StoreCategory)
        )
    return [category.value for category in normalized]
