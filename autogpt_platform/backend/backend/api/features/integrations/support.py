"""Whether the platform can connect a service someone names.

The greeting promises automations, Otto answers "can you use X?", and the
Home checklist labels tasks, all from one answer: a service is supported
when a user can actually connect it here, through a block provider that
offers a sign-in in this deployment or an official MCP catalog server. A
provider whose OAuth app is not configured has blocks but no way in, so it
does not count.
"""

import logging
import re
from collections.abc import Iterable

from pydantic import BaseModel

from backend.api.features.integrations.models import (
    get_all_provider_names,
    get_provider_description,
    get_supported_auth_types,
)
from backend.blocks import load_all_blocks
from backend.integrations.mcp_catalog import get_mcp_catalog
from backend.integrations.service_identity import catalog_slug

logger = logging.getLogger(__name__)

# Providers that are plumbing rather than a service a person names.
_NOT_SERVICES = frozenset({"database", "generic_webhook", "http", "mcp", "smtp"})

# Tools small businesses run on that we have no connector for. Only used to
# read free text written before tasks recorded what they connect to; whether
# each is supported is still decided against the live registry, so one that
# gains a connector drops out on its own.
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

_DOMAIN_SUFFIX = re.compile(r"\.(com|io|ai|dev|app|so|co)\b")
_PARENTHETICAL = re.compile(r"\([^)]*\)")


class ConnectableService(BaseModel):
    key: str
    name: str
    description: str | None = None


def connectable_services() -> list[ConnectableService]:
    """Every service a user can connect in this deployment, providers first."""
    try:
        load_all_blocks()
    except Exception as e:  # static providers and the catalog still answer
        logger.warning(f"Service support: block load failed: {e}")
    services = [
        ConnectableService(
            key=name,
            name=name.replace("_", " ").title(),
            description=get_provider_description(name),
        )
        for name in get_all_provider_names()
        if name not in _NOT_SERVICES and get_supported_auth_types(name)
    ]
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


def find_connectable(name: str) -> ConnectableService | None:
    """The service ``name`` refers to, by id, display name or product."""
    return _match(name, connectable_services())


def unsupported_services(names: Iterable[str]) -> list[str]:
    """The names in ``names`` we cannot connect, de-duplicated, in order."""
    services = connectable_services()
    missing: dict[str, str] = {}
    for name in names:
        label = name.strip()
        key = _normalize(label)
        if key and key not in missing and _match(label, services) is None:
            missing[key] = label
    return list(missing.values())


def mentioned_gaps(text: str) -> list[str]:
    """Known unsupported tools that ``text`` names as whole words."""
    mentioned = [
        name
        for name in KNOWN_GAPS
        if re.search(rf"\b{re.escape(name)}\b", text, re.IGNORECASE)
    ]
    return unsupported_services(mentioned)


def _match(name: str, services: list[ConnectableService]) -> ConnectableService | None:
    """Match on any spelling, or on the first word naming the vendor itself:
    "Google Sheets" is Google's. Only the vendor's own id or full name counts
    there, so "Customer support" is not Customer.io."""
    wanted = _forms(name)
    if not wanted:
        return None
    first_word = _first_word(name)
    for service in services:
        if wanted & (_forms(service.key) | _forms(service.name)):
            return service
        if first_word in (_normalize(service.key), _normalize(service.name)):
            return service
    return None


def _forms(name: str) -> set[str]:
    """The spellings a name is matched on: as written and without a domain."""
    bare = _PARENTHETICAL.sub("", name.lower())
    forms = {_normalize(bare), _normalize(_DOMAIN_SUFFIX.sub("", bare))}
    return {form for form in forms if form}


def _first_word(name: str) -> str:
    words = re.split(r"[^a-z0-9]+", _PARENTHETICAL.sub("", name.lower()))
    return next((word for word in words if word), "")


def _normalize(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", name.lower())
