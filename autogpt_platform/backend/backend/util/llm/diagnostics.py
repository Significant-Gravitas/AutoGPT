"""Backend diagnostics for the configured chat provider."""

from __future__ import annotations

import time
import os
from collections.abc import Mapping
from typing import cast
from urllib.parse import urlparse

from pydantic import BaseModel

from backend.util.llm.config import resolve_chat_config
from backend.util.llm.providers import ProviderLiteral, call_provider


class ProviderDiagnosticResult(BaseModel):
    provider: str
    model: str
    base_url: str
    latency_ms: int
    success: bool
    response: str | None = None
    error: str | None = None


def _safe_base_url(value: str) -> str:
    parsed_url = urlparse(value)
    if not parsed_url.hostname:
        return ""
    try:
        parsed_port = parsed_url.port
    except ValueError:
        parsed_port = None
    port = f":{parsed_port}" if parsed_port else ""
    return f"{parsed_url.scheme}://{parsed_url.hostname}{port}"


async def diagnose_chat_provider(
    environment: Mapping[str, str] | None = None,
) -> ProviderDiagnosticResult:
    started_at = time.monotonic()
    try:
        config = resolve_chat_config(environment)
    except Exception as exc:
        env = environment if environment is not None else os.environ
        return ProviderDiagnosticResult(
            provider=env.get("CHAT_PROVIDER", "unknown") or "unknown",
            model=env.get("CHAT_MODEL", "unknown") or "unknown",
            base_url=_safe_base_url(env.get("CHAT_BASE_URL", "")),
            latency_ms=int((time.monotonic() - started_at) * 1000),
            success=False,
            error=f"{type(exc).__name__}: {exc}",
        )
    safe_base_url = _safe_base_url(config.base_url)
    try:
        if config.api_key is None and config.provider != "local":
            raise ValueError(f"No API key configured from {config.api_key_source}")
        result = await call_provider(
            provider=cast(ProviderLiteral, config.dispatch_provider),
            model=config.model,
            api_key=(
                config.api_key.get_secret_value() if config.api_key is not None else ""
            ),
            base_url=config.base_url,
            messages=[{"role": "user", "content": "Reply with ok."}],
            max_tokens=8,
            ollama_host=config.base_url,
            timeout_seconds=config.request_timeout_s,
            max_retries=config.max_retries,
        )
        return ProviderDiagnosticResult(
            provider=config.provider,
            model=config.model,
            base_url=safe_base_url,
            latency_ms=int((time.monotonic() - started_at) * 1000),
            success=True,
            response=result.content,
        )
    except Exception as exc:
        error_message = str(exc)
        if config.api_key is not None:
            error_message = error_message.replace(
                config.api_key.get_secret_value(), "***"
            )
        return ProviderDiagnosticResult(
            provider=config.provider,
            model=config.model,
            base_url=safe_base_url,
            latency_ms=int((time.monotonic() - started_at) * 1000),
            success=False,
            error=f"{type(exc).__name__}: {error_message}",
        )
