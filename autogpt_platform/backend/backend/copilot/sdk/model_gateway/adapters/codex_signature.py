"""Codex's encrypted reasoning, carried in a thinking block's ``signature``.

With ``store: false`` nothing persists on OpenAI's side, so each request must
hand back the reasoning items the model produced earlier in the turn. The CLI
returns a thinking block's signature byte for byte (verified against CLI
2.1.284), so the encrypted content rides there.

The signature is sealed with a tag for the ChatGPT account and model that
produced it, and only an exact match is unsealed. A thinking block from any
other source -- a Claude turn earlier in the chat (Anthropic signs its own),
a local model (empty), another account or another model -- is dropped
rather than sent to ChatGPT as reasoning it cannot decrypt.
"""

import hashlib

_PREFIX = "agptcx1"


def reasoning_tag(account: str, model: str) -> str:
    """A short, non-reversible tag for the (account, model) pair."""
    return hashlib.sha256(f"{account}\n{model}".encode("utf-8")).hexdigest()[:16]


def seal(encrypted_content: str, tag: str) -> str:
    return f"{_PREFIX}.{tag}.{encrypted_content}"


def unseal(signature: str | None, tag: str) -> str | None:
    """The encrypted content, if this signature was sealed under ``tag``."""
    if not signature:
        return None
    prefix, _, rest = signature.partition(".")
    sealed_tag, _, encrypted = rest.partition(".")
    if prefix != _PREFIX or sealed_tag != tag or not encrypted:
        return None
    return encrypted
