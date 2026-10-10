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
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from fastapi import FastAPI
    from guard import SecurityConfig

DEFAULT_EXCLUDED_PATHS = "/docs,/redoc,/openapi.json,/health_check"
DEFAULT_TRUSTED_PROXIES = "10.0.0.0/8,172.16.0.0/12,192.168.0.0/16"


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "False").strip().lower() in ("1", "true", "yes")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw not in (None, "") else default


def _env_str(name: str, default: str | None = None) -> str | None:
    raw = os.environ.get(name)
    return raw if raw not in (None, "") else default


def _env_csv(name: str, default: str | None = None) -> tuple[str, ...]:
    return tuple(
        item.strip()
        for item in (_env_str(name, default) or "").split(",")
        if item.strip()
    )


def build_guard_config() -> "SecurityConfig":
    """Build the SecurityConfig from AUTOGPT_GUARD_* environment variables."""
    from guard import SecurityConfig

    kwargs: dict[str, Any] = {
        "enable_rate_limiting": True,
        "rate_limit": _env_int("AUTOGPT_GUARD_RATE_LIMIT", 100),
        "rate_limit_window": _env_int("AUTOGPT_GUARD_RATE_LIMIT_WINDOW", 60),
        "enable_ip_banning": True,
        "auto_ban_threshold": _env_int("AUTOGPT_GUARD_AUTO_BAN_THRESHOLD", 10),
        "auto_ban_duration": _env_int("AUTOGPT_GUARD_AUTO_BAN_DURATION", 300),
        "enable_penetration_detection": True,
        # In-memory state unless a Redis URL is configured: never implicitly
        # depend on a Redis server being reachable at localhost.
        "enable_redis": False,
        "passive_mode": _env_bool("AUTOGPT_GUARD_PASSIVE_MODE"),
        "blacklist": _env_csv("AUTOGPT_GUARD_BLOCKED_IPS"),
        "blocked_user_agents": list(_env_csv("AUTOGPT_GUARD_BLOCKED_USER_AGENTS")),
        "trusted_proxies": _env_csv(
            "AUTOGPT_GUARD_TRUSTED_PROXIES", DEFAULT_TRUSTED_PROXIES
        )
        or (),
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


def attach_guard(app: "FastAPI") -> None:
    """Attach the fastapi-guard middleware when AUTOGPT_GUARD_ENABLED is set.

    No-op unless the env flag is on. Fails loudly when the flag is on but
    the package is missing: a misconfiguration must never silently disable
    security. Also exposes the shared SecurityDecorator on app.state so
    per-route guard decorators adopt the same config as the middleware.
    """
    if not _env_bool("AUTOGPT_GUARD_ENABLED"):
        return
    try:
        from guard import SecurityDecorator, SecurityMiddleware
    except ImportError as exc:
        raise RuntimeError(
            "AUTOGPT_GUARD_ENABLED=1 requires the fastapi-guard package. "
            "Install it with: pip install fastapi-guard"
        ) from exc

    config = build_guard_config()
    app.add_middleware(SecurityMiddleware, config=config)
    app.state.guard_decorator = SecurityDecorator(config)
