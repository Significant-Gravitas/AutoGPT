"""Optional fastapi-guard security middleware wiring.

fastapi-guard (https://github.com/Guard-Core/fastapi-guard) provides IP
block/allow lists, rate limiting with auto-ban, user-agent blocking,
penetration-attempt detection, security headers, optional Redis-backed
shared state, and optional IPInfo geo/cloud-provider lookups.

Everything here is opt-in: with no AUTOGPT_GUARD_* environment variables
set, attach_guard() is a no-op and the app behaves exactly as before. Set
AUTOGPT_GUARD_ENABLED=1 to activate the middleware on a service.
"""

import os
from typing import Any

from fastapi import FastAPI, WebSocket
from guard import SecurityConfig, SecurityDecorator, SecurityMiddleware, guard_websocket

DEFAULT_EXCLUDED_PATHS = "/docs,/redoc,/openapi.json,/health_check"


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("", "0", "false", "no", "off"):
        return False
    msg = f"{name} must be a boolean (1/true/yes/on or 0/false/no/off), " f"got {raw!r}"
    raise ValueError(msg)


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    return raw if raw not in (None, "") else default


def _env_csv(name: str, default: str | None = None) -> tuple[str, ...]:
    # Unset uses the default; a set-but-empty variable is taken literally.
    raw = os.environ.get(name, default) or ""
    return tuple(item.strip() for item in raw.split(",") if item.strip())


def build_guard_config() -> SecurityConfig:
    """Build the SecurityConfig from AUTOGPT_GUARD_* environment variables."""
    kwargs: dict[str, Any] = {
        "enable_rate_limiting": True,
        "rate_limit": _env_int("AUTOGPT_GUARD_RATE_LIMIT", 100),
        "rate_limit_window": _env_int("AUTOGPT_GUARD_RATE_LIMIT_WINDOW", 60),
        "enable_ip_banning": _env_bool("AUTOGPT_GUARD_IP_BANNING", default=True),
        "auto_ban_threshold": _env_int("AUTOGPT_GUARD_AUTO_BAN_THRESHOLD", 10),
        "auto_ban_duration": _env_int("AUTOGPT_GUARD_AUTO_BAN_DURATION", 300),
        "enable_rate_limit_auto_ban": _env_bool("AUTOGPT_GUARD_RATE_LIMIT_AUTO_BAN"),
        "enable_penetration_detection": _env_bool(
            "AUTOGPT_GUARD_PENETRATION_DETECTION", default=True
        ),
        # Graphs, block inputs and copilot messages carry SQL, HTML and code as
        # data, so signature-scanning bodies flags ordinary content. Detection
        # keeps screening the path, query params and headers; body scanning is
        # opt-in.
        "detection_scan_body": _env_bool("AUTOGPT_GUARD_SCAN_BODY"),
        # In-memory state unless a Redis URL is configured: never implicitly
        # depend on a Redis server being reachable at localhost.
        "enable_redis": False,
        "passive_mode": _env_bool("AUTOGPT_GUARD_PASSIVE_MODE"),
        "blacklist": _env_csv("AUTOGPT_GUARD_BLOCKED_IPS"),
        "blocked_user_agents": list(_env_csv("AUTOGPT_GUARD_BLOCKED_USER_AGENTS")),
        # No proxy is trusted unless configured, so X-Forwarded-For can't be
        # spoofed by any client that reaches the service over a private network.
        "trusted_proxies": _env_csv("AUTOGPT_GUARD_TRUSTED_PROXIES"),
        "trusted_proxy_depth": _env_int("AUTOGPT_GUARD_TRUSTED_PROXY_DEPTH", 1),
        "exclude_paths": list(
            _env_csv("AUTOGPT_GUARD_EXCLUDED_PATHS", DEFAULT_EXCLUDED_PATHS)
        ),
        "custom_log_file": _env_str("AUTOGPT_GUARD_LOG_FILE"),
        "log_format": _env_str("AUTOGPT_GUARD_LOG_FORMAT", "text"),
        "security_headers": (
            {
                "enabled": True,
                "hsts": {"max_age": 31536000, "include_subdomains": True},
                "frame_options": "SAMEORIGIN",
                "content_type_options": "nosniff",
                "referrer_policy": "strict-origin-when-cross-origin",
            }
            if _env_bool("AUTOGPT_GUARD_SECURITY_HEADERS")
            else None
        ),
        "enforce_https": _env_bool("AUTOGPT_GUARD_ENFORCE_HTTPS"),
    }

    # Behind a TLS-terminating proxy, enforce_https must read the forwarded
    # scheme or every request looks like plain HTTP.
    explicit_xfp = _env_str("AUTOGPT_GUARD_TRUST_X_FORWARDED_PROTO")
    if explicit_xfp is not None:
        kwargs["trust_x_forwarded_proto"] = _env_bool(
            "AUTOGPT_GUARD_TRUST_X_FORWARDED_PROTO"
        )
    else:
        kwargs["trust_x_forwarded_proto"] = _env_bool("AUTOGPT_GUARD_ENFORCE_HTTPS")

    if allowed_ips := _env_csv("AUTOGPT_GUARD_ALLOWED_IPS"):
        kwargs["whitelist"] = allowed_ips
    if blocked_countries := _env_csv("AUTOGPT_GUARD_BLOCKED_COUNTRIES"):
        kwargs["blocked_countries"] = frozenset(blocked_countries)
    if allowed_countries := _env_csv("AUTOGPT_GUARD_ALLOWED_COUNTRIES"):
        kwargs["whitelist_countries"] = frozenset(allowed_countries)
    if cloud_providers := _env_csv("AUTOGPT_GUARD_BLOCK_CLOUD_PROVIDERS"):
        kwargs["block_cloud_providers"] = frozenset(cloud_providers)

    if redis_url := _env_str("AUTOGPT_GUARD_REDIS_URL"):
        kwargs["enable_redis"] = True
        kwargs["redis_url"] = redis_url
        kwargs["redis_prefix"] = "autogpt_guard:"

    if ipinfo_token := _env_str("AUTOGPT_GUARD_IPINFO_TOKEN"):
        kwargs["ipinfo_token"] = ipinfo_token

    return SecurityConfig(**kwargs)


def attach_guard(app: FastAPI, *, internal_service: bool = False) -> None:
    """Attach the fastapi-guard middleware when AUTOGPT_GUARD_ENABLED is set.

    No-op unless the env flag is on. Also exposes the shared SecurityDecorator
    on app.state so per-route guard decorators adopt the same config as the
    middleware.

    Internal RPC services share the backend env with the public ones, and
    their callers don't retry a 4xx, so they stay unguarded unless
    AUTOGPT_GUARD_INTERNAL_SERVICES is set as well.
    """
    if not _env_bool("AUTOGPT_GUARD_ENABLED"):
        return
    if internal_service and not _env_bool("AUTOGPT_GUARD_INTERNAL_SERVICES"):
        return
    config = build_guard_config()
    app.add_middleware(SecurityMiddleware, config=config)
    app.state.guard_decorator = SecurityDecorator(config)


async def guard_websocket_connection(websocket: WebSocket) -> None:
    """Run the guard's checks on a websocket handshake when the guard is attached.

    SecurityMiddleware only sees HTTP requests, so websocket routes add this as
    a dependency.
    """
    if hasattr(websocket.app.state, "guard_decorator"):
        await guard_websocket(websocket)
