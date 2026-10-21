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

# `make init-env` only fills `NAME=` lines that are present and blank, so each
# hint spells out the one edit that makes it apply.
MISSING_HINT = (
    "Add a line `ENCRYPTION_KEY=` to autogpt_platform/backend/.env (create the "
    "file if needed) and run `make init-env` in autogpt_platform/ to generate a "
    "value for it. Values you already set are never overwritten."
)
UPGRADE_HINT = (
    " Upgrading an install that already has connected integrations? They can be "
    "moved to the new key instead of reconnected: see 'Upgrading: secrets are "
    "generated per install' in docs/platform/getting-started.md."
)
RETIRED_HINT = (
    "Clear the value in autogpt_platform/backend/.env (leave `{name}=`) and run "
    "`make init-env` in autogpt_platform/ to generate a fresh one."
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
            f"start without it. {MISSING_HINT}{UPGRADE_HINT}"
        )

    for name, value in configured.items():
        if not value or hashlib.sha256(value.encode()).hexdigest() not in (
            _RETIRED_DIGESTS[name]
        ):
            continue
        rotation_note = UPGRADE_HINT if name == "ENCRYPTION_KEY" else ""
        raise ValueError(
            f"{name} is set to a value that was published in this repository's "
            "public .env.default and must be treated as compromised. "
            f"{RETIRED_HINT.format(name=name)}{rotation_note}"
        )

    if not configured["UNSUBSCRIBE_SECRET_KEY"]:
        logger.warning(
            "[SECURITY] UNSUBSCRIBE_SECRET_KEY is not set: email unsubscribe "
            "links cannot be signed. "
            + MISSING_HINT.replace("ENCRYPTION_KEY", "UNSUBSCRIBE_SECRET_KEY")
        )
