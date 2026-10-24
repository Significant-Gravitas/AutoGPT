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
from collections import defaultdict
from collections.abc import Iterable

from pydantic import BaseModel

from backend.api.features.integrations.models import (
    get_all_provider_names,
    get_provider_description,
    get_supported_auth_types,
)
from backend.blocks import load_all_blocks
from backend.data.model import CredentialsFieldInfo
from backend.integrations.mcp_catalog import get_mcp_catalog
from backend.integrations.oauth import DEVICE_HANDLERS_BY_NAME
from backend.integrations.oauth.app_credentials import oauth_client_credentials
from backend.integrations.providers import provider_key
from backend.integrations.service_identity import catalog_slug
from backend.util.cache import cached

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

_DOMAIN_SUFFIX = re.compile(r"\.(com|io|ai|dev|app|so|co)\b")
_PARENTHETICAL = re.compile(r"\([^)]*\)")


class ConnectableService(BaseModel):
    key: str
    name: str
    description: str | None = None


def connectable_services() -> list[ConnectableService]:
    """Every service a user can connect in this deployment, providers first."""
    try:
        from_blocks = block_credential_types()
    except Exception as e:  # static providers and the catalog still answer
        logger.warning(f"Service support: block load failed: {e}")
        from_blocks = {}
    services = [
        ConnectableService(
            key=name,
            name=name.replace("_", " ").title(),
            description=get_provider_description(name),
        )
        for name in sorted({*get_all_provider_names(), *from_blocks})
        if name not in _NOT_SERVICES
        and _can_sign_in(name, from_blocks.get(name, frozenset()))
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


@cached(ttl_seconds=3600)
def block_credential_types() -> dict[str, frozenset[str]]:
    """Each provider a block that can run takes credentials for, with the
    credential types its fields accept. Loads the blocks."""
    types: dict[str, set[str]] = defaultdict(set)
    for block_id, block_cls in load_all_blocks().items():
        try:
            block = block_cls()
            if block.disabled:
                continue
            infos = block.input_schema.get_credentials_fields_info().values()
        except Exception:  # one bad block costs its own providers only
            logger.debug("Service support: skipping block %s", block_id)
            continue
        for info in infos:
            for provider, accepted in _types_by_provider(info):
                types[provider_key(provider)].update(accepted)
    return {provider: frozenset(found) for provider, found in types.items()}


def _types_by_provider(info: CredentialsFieldInfo) -> list[tuple[str, frozenset]]:
    """What each provider of one credentials field accepts. A field that
    switches provider by another input (OpenAI key or a ChatGPT sign-in)
    gives each provider only its own types."""
    if not (info.discriminator and info.discriminator_mapping):
        return [(provider, info.supported_types) for provider in info.provider]
    pairs = []
    for value in info.discriminator_mapping:
        try:
            narrowed = info.discriminate(value)
        except ValueError:  # a value with no types configured offers none
            continue
        pairs += [
            (provider, narrowed.supported_types) for provider in narrowed.provider
        ]
    return pairs


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
