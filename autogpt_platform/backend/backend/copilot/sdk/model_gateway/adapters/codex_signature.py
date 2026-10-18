"""Account-bound reasoning integrity, independent of access-token refresh."""

import hashlib
import hmac
import json

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.hkdf import HKDF
from pydantic import BaseModel, ConfigDict

from backend.util import encryption

_PREFIX = "agptcx2"
_LABEL = b"autogpt/copilot/codex-reasoning-seal/v1"


class ReasoningIdentity(BaseModel):
    model_config = ConfigDict(frozen=True)

    account: str
    model: str


def reasoning_tag(account: str, model: str) -> ReasoningIdentity:
    return ReasoningIdentity(account=account, model=model)


def seal(encrypted_content: str, tag: ReasoningIdentity) -> str:
    return f"{_PREFIX}.{_mac(encrypted_content, tag)}.{encrypted_content}"


def unseal(signature: str | None, tag: ReasoningIdentity) -> str | None:
    if not signature:
        return None
    prefix, _, rest = signature.partition(".")
    supplied, _, encrypted = rest.partition(".")
    if prefix != _PREFIX or not encrypted:
        return None
    expected = _mac(encrypted, tag)
    if not hmac.compare_digest(supplied.encode("utf-8"), expected.encode("ascii")):
        return None
    return encrypted


def _mac(encrypted: str, tag: ReasoningIdentity) -> str:
    key = HKDF(algorithm=hashes.SHA256(), length=32, salt=None, info=_LABEL).derive(
        encryption.ENCRYPTION_KEY.encode("utf-8")
    )
    canonical = json.dumps(
        [_PREFIX, tag.account, tag.model, encrypted],
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()
