"""What the blocks say about each provider they take credentials for.

Two things the support check needs and no provider metadata holds for every
connector: the sign-ins a provider's blocks accept (older connectors such as
Slack declare theirs only on their blocks' credentials fields), and the words
of their names, which say what they cover (``GoogleSheetsAppendRowBlock``
covers sheets). The words include verbs, so they qualify a vendor's name
("Google Sheets"), never stand for a product on their own.
"""

import logging
import re
from collections import defaultdict

from pydantic import BaseModel

from backend.blocks import load_all_blocks
from backend.data.model import CredentialsFieldInfo
from backend.integrations.providers import provider_key
from backend.util.cache import cached

logger = logging.getLogger(__name__)

_CAMEL_WORD = re.compile(r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+")


class ProviderBlocks(BaseModel):
    auth_types: frozenset[str] = frozenset()
    """Credential types accepted by the provider's blocks that can run."""
    products: frozenset[str] = frozenset()
    """Every word of its blocks' names, singular: what the blocks cover."""


@cached(ttl_seconds=3600)
def blocks_by_provider() -> dict[str, ProviderBlocks]:
    """Each provider a block takes credentials for. Loads the blocks."""
    types: dict[str, set[str]] = defaultdict(set)
    words: dict[str, set[str]] = defaultdict(set)
    for block_id, block_cls in load_all_blocks().items():
        try:
            block = block_cls()
            infos = block.input_schema.get_credentials_fields_info().values()
        except Exception:  # one bad block costs its own providers only
            logger.debug("Block coverage: skipping block %s", block_id)
            continue
        name_words = name_words_of(block.name)
        for info in infos:
            for provider, accepted in _types_by_provider(info):
                key = provider_key(provider)
                words[key].update(name_words)
                if not block.disabled:  # a block that cannot run takes nothing
                    types[key].update(accepted)
    return {
        key: ProviderBlocks(
            auth_types=frozenset(types[key]), products=frozenset(words[key])
        )
        for key in words
    }


def name_words_of(block_name: str) -> list[str]:
    """``GoogleSheetsAppendRowBlock`` as ["google", "sheet", "append", "row"]."""
    found = [singular(word.lower()) for word in _CAMEL_WORD.findall(block_name)]
    return [word for word in found if word != "block"]


def singular(word: str) -> str:
    """One plural "s" off ("sheets" is sheet); "glass" and "boss" stay."""
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


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
