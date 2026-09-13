"""Refuse to boot on secrets that were once shipped in a public file.

`backend/.env.default` carried working values for these settings until
SECRT-2611. The file is public, so those values are public: a deployment still
using one encrypts stored integration credentials with a key anybody can read,
and signs Redis cache entries with a forgeable HMAC key (`util/cache.py`
derives that key from `ENCRYPTION_KEY`).

The retired values are matched by SHA-256 digest, so detecting them does not
require putting the literals back into a tracked file.
"""

import hashlib
import logging

from backend.util.settings import Settings

logger = logging.getLogger(__name__)

BOOTSTRAP_HINT = (
    "Run `make init-env` in autogpt_platform/ to generate per-developer secrets "
    "into backend/.env (it never overwrites values you already set)."
)

# SHA-256 of each value published in backend/.env.default before SECRT-2611.
# Treat all of them as permanently burned.
_RETIRED_DIGESTS: dict[str, set[str]] = {
    "ENCRYPTION_KEY": {
        "731473d9f10755f30ae97baafe65bf0aba95f83100ed418a7c3bc257a2a1ef7c"  # pragma: allowlist secret
    },
    "UNSUBSCRIBE_SECRET_KEY": {
        "0a2f4a5ce1e9e072b5773bccab2839f62d46e61cc81fdf6aba590f7983694517"  # pragma: allowlist secret
    },
}


def check_secrets(settings: Settings | None = None) -> None:
    """Raise if a required secret is missing or is a known-published value."""
    secrets = (settings or Settings()).secrets
    configured = {
        "ENCRYPTION_KEY": secrets.encryption_key,
        "UNSUBSCRIBE_SECRET_KEY": secrets.unsubscribe_secret_key,
    }

    if not configured["ENCRYPTION_KEY"]:
        raise ValueError(
            "ENCRYPTION_KEY is not set. It encrypts stored integration "
            "credentials and signs cached values, so the backend will not "
            f"start without it. {BOOTSTRAP_HINT}"
        )

    for name, value in configured.items():
        if not value or hashlib.sha256(value.encode()).hexdigest() not in (
            _RETIRED_DIGESTS[name]
        ):
            continue
        rotation_note = (
            " Credentials stored under the old key become unreadable, so those "
            "integrations need reconnecting."
            if name == "ENCRYPTION_KEY"
            else ""
        )
        raise ValueError(
            f"{name} is set to a value that was published in this repository's "
            "public .env.default and must be treated as compromised. Replace it "
            f"with a fresh secret. {BOOTSTRAP_HINT}{rotation_note}"
        )

    if not configured["UNSUBSCRIBE_SECRET_KEY"]:
        logger.warning(
            "[SECURITY] UNSUBSCRIBE_SECRET_KEY is not set: email unsubscribe "
            f"links cannot be signed. {BOOTSTRAP_HINT}"
        )
