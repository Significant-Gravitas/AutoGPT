"""Backend diagnostics for the configured chat provider."""

from __future__ import annotations

import time
import os
from collections.abc import Mapping
from typing import cast
from urllib.parse import urlparse

from pydantic import BaseModel

from backend.util.llm.providers import ProviderLiteral, ProviderResponse, call_provider
from backend.util.llm.runtime_config import (
    EffectiveLlmConfig,
    LlmRuntimeOverrides,
    resolve_effective_llm_config,
)


class ProviderDiagnosticResult(BaseModel):
    provider: str
    model: str
    base_url: str
    base_url_host: str
    config_source: str
    latency_ms: int
    success: bool
    response: str | None = None
    error_type: str | None = None
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
    *,
    config: EffectiveLlmConfig | None = None,
    overrides: LlmRuntimeOverrides | None = None,
    use_persisted: bool | None = None,
) -> ProviderDiagnosticResult:
    started_at = time.monotonic()
    try:
        effective = config or await resolve_effective_llm_config(
            environment,
            use_persisted=(
                environment is None if use_persisted is None else use_persisted
            ),
            overrides=overrides,
        )
    except Exception as exc:
        env = environment if environment is not None else os.environ
        safe_base_url = _safe_base_url(env.get("CHAT_BASE_URL", ""))
        return ProviderDiagnosticResult(
            provider=env.get("CHAT_PROVIDER", "unknown") or "unknown",
            model=env.get("CHAT_MODEL", "unknown") or "unknown",
            base_url=safe_base_url,
            base_url_host=urlparse(safe_base_url).hostname or "",
            config_source="env",
            latency_ms=int((time.monotonic() - started_at) * 1000),
            success=False,
            error_type=type(exc).__name__,
            error=f"{type(exc).__name__}: {exc}",
        )
    safe_base_url = _safe_base_url(effective.base_url)
    try:
        if effective.api_key is None and effective.provider != "local":
            raise ValueError(f"No API key configured from {effective.api_key_source}")
        result = await call_provider(
            provider=cast(ProviderLiteral, effective.dispatch_provider),
            model=effective.model,
            api_key=(
                effective.api_key.get_secret_value()
                if effective.api_key is not None
                else ""
            ),
            base_url=effective.base_url,
            messages=[{"role": "user", "content": "Reply with ok."}],
            max_tokens=32,
            ollama_host=effective.base_url,
            timeout_seconds=effective.request_timeout_s,
            max_retries=effective.max_retries,
        )
        if not isinstance(result, ProviderResponse):
            raise RuntimeError("Provider diagnostic requires a synchronous response")
        return ProviderDiagnosticResult(
            provider=effective.provider,
            model=effective.model,
            base_url=safe_base_url,
            base_url_host=effective.base_url_host,
            config_source=effective.source,
            latency_ms=int((time.monotonic() - started_at) * 1000),
            success=True,
            response=(result.content or result.reasoning or "").strip()[:200],
        )
    except Exception as exc:
        error_message = str(exc)
        if effective.api_key is not None:
            error_message = error_message.replace(
                effective.api_key.get_secret_value(), "***"
            )
        return ProviderDiagnosticResult(
            provider=effective.provider,
            model=effective.model,
            base_url=safe_base_url,
            base_url_host=effective.base_url_host,
            config_source=effective.source,
            latency_ms=int((time.monotonic() - started_at) * 1000),
            success=False,
            error_type=type(exc).__name__,
            error=f"{type(exc).__name__}: {error_message}",
        )
