"""Whose connection this is: the tenant boundary.

The backend mints a proxy credential for each running stretch of a box and
records it in Redis (``backend/util/e2b_network.py``): the username is the
key, the record names the owner and the user, and holds a SHA-256 digest of
the secret, never the secret.  E2B's host presents the pair in the SOCKS5
handshake; nothing inside the box ever holds it.

So a connection is some owner's only if its username resolves *and* its
secret hashes to the recorded digest.  Anything else is refused before a
single byte is relayed.  A box cannot claim another owner: it would need that
owner's current 256-bit secret.
"""

import hashlib
import hmac
import json
import logging
import re
from collections.abc import Awaitable
from dataclasses import dataclass
from typing import Any, Optional, Protocol

logger = logging.getLogger(__name__)

CREDENTIAL_KEY_PREFIX = "e2b:egress:cred:"
# What the backend mints.  Checked before the lookup so that arbitrary bytes
# from the internet never become a Redis key.
_USERNAME_RE = re.compile(r"box-[0-9a-f]{16}")


@dataclass(frozen=True)
class Owner:
    """*user_id* is who the box runs for, and is always audited.  Their
    credentials are swapped in only if the backend said so (*swaps*): it does
    for CoPilot boxes, not for a block running a graph someone else wrote."""

    label: str
    user_id: Optional[str]
    sandbox_id: Optional[str]
    swaps: bool = False

    @property
    def swap_user_id(self) -> Optional[str]:
        return self.user_id if self.swaps else None


class RedisReader(Protocol):
    def get(self, name: str) -> Awaitable[Any]: ...


class OwnerDirectory:
    def __init__(self, redis: RedisReader):
        self._redis = redis

    async def authenticate(self, username: str, secret: str) -> Optional[Owner]:
        # fullmatch: ``$`` would let a trailing newline through.
        if not _USERNAME_RE.fullmatch(username) or not secret:
            return None
        try:
            raw = await self._redis.get(CREDENTIAL_KEY_PREFIX + username)
        except Exception:
            # Fail closed: a connection nobody can vouch for is not relayed.
            logger.exception("Owner lookup failed; refusing the connection")
            return None
        if not raw:
            return None
        try:
            record = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
            expected = str(record["secret_sha256"])
            label = str(record["owner"])
        except (ValueError, KeyError, TypeError):
            logger.warning("Malformed credential record for %s", username)
            return None
        presented = hashlib.sha256(secret.encode()).hexdigest()
        if not hmac.compare_digest(presented, expected):
            return None
        return Owner(
            label=label,
            user_id=record.get("user_id") or None,
            sandbox_id=record.get("sandbox_id") or None,
            # Fail closed: only an explicit true.
            swaps=record.get("swaps") is True,
        )
