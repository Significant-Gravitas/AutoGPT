"""Configuration, all from the environment.

Redis uses the backend's own variable names so that a deployment hands both
the same values.
"""

import os
from dataclasses import dataclass, field


def _flag(name: str) -> bool:
    return os.getenv(name, "").lower() in ("1", "true", "yes")


def _list(name: str) -> list[str]:
    return [e.strip() for e in os.getenv(name, "").split(",") if e.strip()]


@dataclass(frozen=True)
class Settings:
    listen_host: str = field(
        default_factory=lambda: os.getenv("SWAP_PROXY_LISTEN_HOST", "0.0.0.0")
    )
    listen_port: int = field(
        default_factory=lambda: int(os.getenv("SWAP_PROXY_LISTEN_PORT", "1080"))
    )
    # The backend's swap service (backend/copilot/swap_service.py), which
    # answers the two calls in source.py and nothing else.
    backend_url: str = field(
        default_factory=lambda: os.getenv(
            "SWAP_PROXY_BACKEND_URL", "http://localhost:8012"
        )
    )
    # Where the CA the proxy signs with is mounted (``mitmproxy-ca.pem``: key
    # and certificate).  The boxes' image trusts its certificate, so it is one
    # CA for every replica and every restart: provisioned, never generated.
    confdir: str = field(
        default_factory=lambda: os.getenv("SWAP_PROXY_CONFDIR", "~/.mitmproxy")
    )
    # Local runs only: let mitmproxy make up a CA when none is there.
    generate_ca: bool = field(default_factory=lambda: _flag("SWAP_PROXY_GENERATE_CA"))
    # Private hosts or CIDRs boxes may reach anyway.  Empty: default deny.
    egress_allow: list[str] = field(
        default_factory=lambda: _list("SWAP_PROXY_EGRESS_ALLOW")
    )
    # Credentialed requests (ones that get a value swapped in) per box and per
    # user in each window; 0 turns a limit off.  Past either, the request is
    # not sent and the box is told why (``quota.py``).
    quota_per_box: int = field(
        default_factory=lambda: int(os.getenv("SWAP_PROXY_QUOTA_PER_BOX", "1000"))
    )
    quota_per_user: int = field(
        default_factory=lambda: int(os.getenv("SWAP_PROXY_QUOTA_PER_USER", "3000"))
    )
    quota_window_seconds: int = field(
        default_factory=lambda: int(
            os.getenv("SWAP_PROXY_QUOTA_WINDOW_SECONDS", "3600")
        )
    )
    redis_host: str = field(
        default_factory=lambda: os.getenv("REDIS_CLUSTER_HOST")
        or os.getenv("REDIS_HOST", "localhost")
    )
    redis_port: int = field(
        default_factory=lambda: int(
            os.getenv("REDIS_CLUSTER_PORT") or os.getenv("REDIS_PORT", "6379")
        )
    )
    redis_password: str | None = field(
        default_factory=lambda: os.getenv("REDIS_PASSWORD") or None
    )
    redis_use_announced_address: bool = field(
        default_factory=lambda: _flag("REDIS_USE_ANNOUNCED_ADDRESS")
    )
