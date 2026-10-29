"""Refuse to boot on secrets that were once shipped in a public file.

`backend/.env.default` carried working values for these settings until
SECRT-2611. The file is public, so those values are public: a deployment still
using one encrypts stored integration credentials with a key anybody can read,
and signs Redis cache entries with a forgeable HMAC key (`util/cache.py`
derives that key from `ENCRYPTION_KEY`).

The .env.default files also gave the database, message broker and caches
their passwords. Those are only warned about, since the stack keeps running on
them and `make init-env` replaces them.

The retired values are matched by SHA-256 digest, so detecting them does not
require putting the literals back into a tracked file.
"""

import hashlib
import logging
import os
from urllib.parse import unquote, urlsplit

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
CREDENTIAL_HINT = (
    "Follow 'Upgrading: service passwords are generated per install' in "
    "docs/platform/getting-started.md to move to passwords generated for this "
    "install: it empties RabbitMQ's queues before `make init-env` and "
    "`docker compose up -d` restart it, which would lose what they hold."
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

# SHA-256 of every password the public .env.default and .env.example files have
# given the stack's own database, message broker and caches. Replacing one of
# these loses nothing, unlike the keys above, so the backend only warns and
# `make init-env` replaces them (single-container/runtime_config.py keeps the
# same set for that).
_PUBLISHED_CREDENTIAL_DIGESTS = {
    "53d3eb3d9500cdcdbbe39c030eb2959438bcc459d5de9380db2a3c85b5de5360",  # pragma: allowlist secret
    "9b8769a4a742959a2d0298c36fb70623f2dfacda8436237df08d8dfd5b37374c",  # pragma: allowlist secret
    "e4b879cfddda5d56e0d4c5b1ec46e605159469234adbabc5c5bcf2ebce5755cd",  # pragma: allowlist secret
    "5e884898da28047151d0e56f8dc6292773603d0d6aabbdd62a11ef721d1542d8",  # pragma: allowlist secret
    "d417f13ced355a54a9444aed43c7ba4f47a81c0c07b8fedf3207cc876bcb8db3",  # pragma: allowlist secret
}


def check_secrets(settings: Settings | None = None) -> None:
    """Raise if a required secret is missing or is a known-published value."""
    settings = settings or Settings()
    _warn_on_published_credentials(settings)
    _warn_on_missing_passwords(settings)
    secrets = settings.secrets
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


def _warn_on_published_credentials(settings: Settings) -> None:
    """Warn about each service password that is a published value. The service
    keeps working on it; anyone who can reach that service can log in too."""
    credentials = {
        f"The database password in {name}": _url_password(os.environ.get(name, ""))
        for name in ("DATABASE_URL", "DIRECT_URL")
    }
    credentials |= {
        "RABBITMQ_DEFAULT_PASS": settings.secrets.rabbitmq_default_pass,
        "REDIS_PASSWORD": settings.config.redis_password,
        "GRAPHITI_FALKORDB_PASSWORD": os.environ.get("GRAPHITI_FALKORDB_PASSWORD", ""),
    }
    for name, value in credentials.items():
        digest = hashlib.sha256(value.encode()).hexdigest()
        if value and digest in _PUBLISHED_CREDENTIAL_DIGESTS:
            logger.warning(
                f"[SECURITY] {name} is a password published in this repository's "
                f"public .env.default files, so anyone can know it. {CREDENTIAL_HINT}"
            )


def _warn_on_missing_passwords(settings: Settings) -> None:
    """Warn about each Compose service the backend uses that has no password:
    Redis then lets anyone on its networks in, and the others don't work."""
    # `redis-0` is the Compose cluster, whose shards take REDIS_PASSWORD from
    # backend/.env; one from before they had a password runs them without one.
    if settings.config.redis_host == "redis-0" and not settings.config.redis_password:
        logger.warning(
            "[SECURITY] The Docker Compose Redis cluster has no password, so any "
            f"container on its networks can read and change it. {CREDENTIAL_HINT}"
        )
    # A backend/.env from before these two had lines took them from
    # .env.default, which now leaves them blank for `make init-env` to fill.
    # Compose's FalkorDB stops without one, and RabbitMQ refuses a blank one.
    if os.environ.get("GRAPHITI_FALKORDB_HOST") == "falkordb" and not os.environ.get(
        "GRAPHITI_FALKORDB_PASSWORD"
    ):
        logger.warning(
            "GRAPHITI_FALKORDB_PASSWORD is not set, so the Docker Compose FalkorDB "
            f"doesn't start and memory is unavailable. {CREDENTIAL_HINT}"
        )
    if (
        settings.config.rabbitmq_host == "rabbitmq"
        and not settings.secrets.rabbitmq_default_pass
    ):
        logger.warning(
            "RABBITMQ_DEFAULT_PASS is not set, so the backend can't log in to the "
            f"Docker Compose RabbitMQ. {CREDENTIAL_HINT}"
        )


def _url_password(url: str) -> str:
    """A connection string's password, decoded, or nothing for one urllib can't
    take apart (a `[` or `]` it reads as part of an IPv6 address): the database
    client reports that one, and a warning must not stop the process first."""
    try:
        return unquote(urlsplit(url).password or "")
    except ValueError:
        return ""
