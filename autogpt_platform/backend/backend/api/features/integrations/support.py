"""Whether the platform can connect a service someone names.

The greeting promises automations, Otto answers "can you use X?", and the
Home checklist labels tasks, all from one answer: a service is supported
when a user can actually connect it here, through a block provider whose
sign-in a user here can finish or an official MCP catalog server. A provider
whose only sign-in is OAuth, with no app configured, has blocks but no way
in, so it does not count.

A provider's sign-ins come from every place one is declared: its
ProviderBuilder, and the credentials fields of the blocks that can run.
Older connectors (Slack's bot token) declare theirs only on their blocks.
"""

import logging
import re
from collections.abc import Iterable

from pydantic import BaseModel

from backend.api.features.integrations import block_coverage
from backend.api.features.integrations.block_coverage import ProviderBlocks
from backend.api.features.integrations.models import (
    get_all_provider_names,
    get_provider_description,
    get_supported_auth_types,
)
from backend.integrations.mcp_catalog import get_mcp_catalog
from backend.integrations.oauth import DEVICE_HANDLERS_BY_NAME
from backend.integrations.oauth.app_credentials import oauth_client_credentials
from backend.integrations.providers import provider_key
from backend.integrations.service_identity import catalog_slug
from backend.util.cache import cached
from backend.util.text import singular

logger = logging.getLogger(__name__)

# Providers that are plumbing rather than a service a person names.
_NOT_SERVICES = frozenset({"database", "generic_webhook", "http", "mcp", "smtp"})

# Tools small businesses run on that we have no connector for. Only used to
# read free text written before tasks recorded what they connect to: a new
# task cannot be stored without its integrations (intro._parse_prompts), so
# this covers older rows until they are backfilled. Whether each is
# supported is still decided against the live registry, so one that gains
# a connector drops out on its own.
KNOWN_GAPS = (
    "Shopify",
    "Printful",
    "Gelato",
    "WooCommerce",
    "Squarespace",
    "BigCommerce",
    "Redbubble",
    "QuickBooks",
    "Xero",
    "Mailchimp",
    "Klaviyo",
    "Printify",
    "Etsy",
)

# Capitalised words after a vendor's name that do not name its product: the
# small words of a Title Case title ("Add To Notion And Google Sheets") and
# days and months ("post in Slack Monday morning").
_NOT_PRODUCT_WORDS = frozenset(
    {
        *("a", "an", "and", "as", "at", "by", "each", "every", "for", "from"),
        *("in", "into", "of", "on", "or", "the", "then", "to", "via", "with"),
        *("my", "our", "your", "is", "are", "if", "when", "daily", "weekly"),
        *("monday", "tuesday", "wednesday", "thursday", "friday"),
        *("saturday", "sunday", "today", "tomorrow"),
        *("january", "february", "march", "april", "may", "june", "july"),
        *("august", "september", "october", "november", "december"),
    }
)
# A sentence ends at a line break or at .!? before a space, not inside a name
# such as "monday.com".
_SENTENCE_BREAK = re.compile(r"\n+|(?<=[.!?])\s+")
# Words for how a vendor is reached rather than what it does: "Printify API"
# is Printify.
_ACCESS_WORDS = frozenset(
    {"api", "app", "account", "integration", "connector", "platform", "workspace"}
)
_DOMAIN_SUFFIX = re.compile(r"\.(com|io|ai|dev|app|so|co)\b")
_PARENTHETICAL = re.compile(r"\([^)]*\)")


class ConnectableService(BaseModel):
    key: str
    name: str
    description: str | None = None
    products: frozenset[str] = frozenset()
    """What its blocks that can run cover (``block_coverage``)."""
    has_blocks: bool = False
    """False for a service with no blocks, such as an MCP catalog server,
    which is the vendor's whole surface."""
    product_names: frozenset[str] = frozenset()
    """Products its description names ("Gmail, Drive, Calendar, Sheets"),
    normalized: each is the provider on its own."""


def connectable_services() -> list[ConnectableService]:
    """Every service a user can connect in this deployment, providers first."""
    try:
        from_blocks = block_coverage.blocks_by_provider()
    except Exception as e:  # static providers and the catalog still answer
        logger.warning(f"Service support: block load failed: {e}")
        from_blocks = {}
    services = []
    for name in sorted({*get_all_provider_names(), *from_blocks}):
        blocks = from_blocks.get(name, ProviderBlocks())
        if name in _NOT_SERVICES or not _can_sign_in(name, blocks.auth_types):
            continue
        description = get_provider_description(name)
        services.append(
            ConnectableService(
                key=name,
                name=name.replace("_", " ").title(),
                description=description,
                products=blocks.products,
                has_blocks=blocks.has_blocks,
                product_names=described_products(description),
            )
        )
    keys = {service.key for service in services}
    for entry in get_mcp_catalog():
        slug = catalog_slug(entry)
        if slug not in keys:
            keys.add(slug)
            services.append(
                ConnectableService(
                    key=slug, name=entry.display_name, description=entry.description
                )
            )
    return services


def _can_sign_in(provider: str, from_blocks: frozenset[str]) -> bool:
    """Whether a user here can finish at least one of ``provider``'s sign-ins.

    A key, password or website header is the user's own to paste. An OAuth
    account comes from the provider's app configured in this deployment
    (Google and Reddit declare OAuth either way) or from its device-code
    handler, whose sign-in yields the same OAuth credentials (Codex).
    """
    has_device_handler = provider_key(provider) in DEVICE_HANDLERS_BY_NAME
    for auth_type in {*get_supported_auth_types(provider), *from_blocks}:
        if auth_type == "oauth2":
            if has_device_handler or oauth_client_credentials(provider) is not None:
                return True
        elif auth_type == "device_code":
            if has_device_handler:
                return True
        else:
            return True
    return False


def find_connectable(name: str) -> ConnectableService | None:
    """The service ``name`` refers to, by id, display name or product."""
    return _match(name, connectable_services())


def unsupported_services(
    names: Iterable[str], services: list[ConnectableService] | None = None
) -> list[str]:
    """The names in ``names`` we cannot connect, de-duplicated, in order.

    Pass ``services`` when labelling many names at once, so the registry is
    read once rather than per call.
    """
    services = connectable_services() if services is None else services
    missing: dict[str, str] = {}
    for name in names:
        label = name.strip()
        key = _normalize(label)
        if key and key not in missing and _match(label, services) is None:
            missing[key] = label
    return list(missing.values())


def unsupported_in_task(
    integrations: list[str] | None,
    text: str,
    services: list[ConnectableService] | None = None,
) -> list[str]:
    """The tools a task needs that we cannot connect: the ones it declares,
    and the ones its text names that we know of, our known gaps and every
    vendor on the platform (connectable here or not). The text is read
    whatever the declaration says, since the model that wrote it can leave
    a tool out (``[]`` on a Shopify task)."""
    services = connectable_services() if services is None else services
    declared = [canonical_name(name) for name in integrations or []]
    named = [
        *_known_gaps_in(text, capitalised=True),
        *_unconnectable_vendors_in(text, services),
    ]
    return unsupported_services([*declared, *named], services)


def described_products(description: str | None) -> frozenset[str]:
    """The products a provider's description names, when it names them:
    "Gmail, Drive, Calendar, Sheets" does, every item capitalised;
    "Subreddits, posts, and comments" lists kinds of things."""
    items = [
        item.strip()
        for item in re.split(r",|\band\b", description or "")
        if item.strip()
    ]
    if len(items) < 2 or not all(item[0].isupper() for item in items):
        return frozenset()
    return frozenset(_normalize(item) for item in items)


def _unconnectable_vendors_in(
    text: str, services: list[ConnectableService]
) -> list[str]:
    """Each vendor the platform knows that ``text`` names, with its product,
    as written, when that cannot be connected here. Only a capitalised
    mention counts, so "the notion of" is not Notion.

    Read a sentence at a time. In ordinary prose a capitalised word after
    the vendor is its product ("Google Analytics"), judged by the first such
    word; later ones are kept in the name but not judged. In a Title Case
    sentence every word is capitalised, which says nothing ("Ask Notion
    Questions"), so only the vendor is judged there."""
    found = []
    for sentence in _SENTENCE_BREAK.split(text):
        matches = list(_vendor_pattern().finditer(sentence))
        title_case = _is_title_case(sentence, [m.span() for m in matches])
        for match in matches:
            vendor = match.group("vendor")
            mention = vendor if title_case else match.group(0)
            head = mention.split()[: len(vendor.split()) + 1]
            if _match(" ".join(head), services) is None:
                found.append(mention)
    return found


def _is_title_case(sentence: str, mentions: list[tuple[int, int]]) -> bool:
    """Whether most of the other words of ``sentence`` start with a capital,
    so a capital marks nothing. The vendor mentions themselves, the first
    word (capitalised anyway) and small joining words are not counted."""
    words = [
        word.group(0)
        for index, word in enumerate(re.finditer(r"[A-Za-z][A-Za-z']*", sentence))
        if index > 0
        and word.group(0).lower() not in _NOT_PRODUCT_WORDS
        and not any(start <= word.start() < end for start, end in mentions)
    ]
    capitalised = sum(word[0].isupper() for word in words)
    return bool(words) and capitalised >= 0.6 * len(words)


@cached(ttl_seconds=3600)
def _vendor_pattern() -> re.Pattern[str]:
    """Every provider (connectable here or not) and MCP catalog server, by
    the names people write: "google maps", "googlemaps", "Customer.io"."""
    try:
        keys = {*get_all_provider_names(), *block_coverage.blocks_by_provider()}
    except Exception:  # static providers still answer
        keys = set(get_all_provider_names())
    names = {
        form
        for key in keys - _NOT_SERVICES
        for form in (key.replace("_", " "), key.replace("_", ""))
    }
    names |= {
        _PARENTHETICAL.sub("", entry.display_name).strip()
        for entry in get_mcp_catalog()
    }
    alternatives = "|".join(
        re.escape(name) for name in sorted(names - {""}, key=len, reverse=True)
    )
    joiners = "|".join(sorted(_NOT_PRODUCT_WORDS))
    # The vendor, then up to two capitalised words naming its product
    # ("Google Analytics", "Google Search Console"), so the whole name is
    # checked: "Google" alone would pass as supported.
    product = rf"(?:[ \t]+(?!(?i:{joiners})\b)[A-Z][A-Za-z0-9]*){{0,2}}"
    return re.compile(rf"\b(?=[A-Z])(?P<vendor>(?i:{alternatives}))\b{product}")


def canonical_name(name: str) -> str:
    """One spelling per known tool ("Shopify store" is Shopify), so asks for
    the same tool land together; anything else is kept as written."""
    known = _known_gaps_in(name)
    return known[0] if known else " ".join(name.split())


def _known_gaps_in(text: str, capitalised: bool = False) -> list[str]:
    """The known gaps ``text`` names as whole words. In free text only a
    capitalised mention counts, as for vendors: "a gelato flavour" is not
    Gelato. A name given as a service ("shopify store") counts in any case."""
    head = "(?=[A-Z])" if capitalised else ""
    return [
        name
        for name in KNOWN_GAPS
        if re.search(rf"\b{head}(?i:{re.escape(name)})\b", text)
    ]


def _match(name: str, services: list[ConnectableService]) -> ConnectableService | None:
    """Match on any spelling; on the vendor's own name followed by products
    its blocks cover ("Google Sheets" is Google's, "Google Analytics" is
    not); or on a product its description names (``described_products``:
    "Gmail" is in Google's "Gmail, Drive, Calendar, Sheets", so editing that
    description changes what counts as Google's). Only the vendor's own id
    or full name counts as the vendor, so "Customer support" is not
    Customer.io."""
    wanted = _forms(name)
    if not wanted:
        return None
    words = _words(name)
    for service in services:
        if wanted & (_forms(service.key) | _forms(service.name)):
            return service
        vendor = (_normalize(service.key), _normalize(service.name))
        if words[0] in vendor and _covers(service, words[1:]):
            return service
        if wanted & service.product_names:
            return service
    return None


def _covers(service: ConnectableService, words: list[str]) -> bool:
    """Whether the words after a vendor's name are products it covers."""
    if not service.has_blocks:
        return True
    return all(
        word in _ACCESS_WORDS or singular(word) in service.products for word in words
    )


def _forms(name: str) -> set[str]:
    """The spellings a name is matched on: as written and without a domain."""
    bare = _PARENTHETICAL.sub("", name.lower())
    forms = {_normalize(bare), _normalize(_DOMAIN_SUFFIX.sub("", bare))}
    return {form for form in forms if form}


def _words(name: str) -> list[str]:
    words = re.split(r"[^a-z0-9]+", _PARENTHETICAL.sub("", name.lower()))
    return [word for word in words if word] or [""]


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())
