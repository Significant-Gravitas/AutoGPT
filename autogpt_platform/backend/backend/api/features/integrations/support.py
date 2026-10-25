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
from backend.api.features.integrations.block_coverage import ProviderBlocks, singular
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
    """What its blocks cover (``block_coverage``); empty for a service with
    no blocks, such as an MCP catalog server, which is the vendor's whole
    surface."""
    leads: frozenset[str] = frozenset()


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
        services.append(
            ConnectableService(
                key=name,
                name=name.replace("_", " ").title(),
                description=get_provider_description(name),
                products=blocks.products,
                leads=blocks.leads,
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
    and the known ones its text names. The text is read whatever the
    declaration says, since the model that wrote it can leave a tool out
    (``[]`` on a Shopify task)."""
    declared = [canonical_name(name) for name in integrations or []]
    return unsupported_services([*declared, *_known_gaps_in(text)], services)


def canonical_name(name: str) -> str:
    """One spelling per known tool ("Shopify store" is Shopify), so asks for
    the same tool land together; anything else is kept as written."""
    known = _known_gaps_in(name)
    return known[0] if known else " ".join(name.split())


def _known_gaps_in(text: str) -> list[str]:
    return [
        name
        for name in KNOWN_GAPS
        if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE)
    ]


def _match(name: str, services: list[ConnectableService]) -> ConnectableService | None:
    """Match on any spelling; on the vendor's own name followed by products
    its blocks cover ("Google Sheets" is Google's, "Google Analytics" is
    not); or on one word naming a product only one provider's blocks lead
    with ("Gmail"). Only the vendor's own id or full name counts as the
    vendor, so "Customer support" is not Customer.io."""
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
        if len(words) == 1 and singular(words[0]) in service.leads:
            return service
    return None


def _covers(service: ConnectableService, words: list[str]) -> bool:
    """Whether the words after a vendor's name are products it covers."""
    if not service.products:
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
