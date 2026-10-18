"""Codex's quota-specific retry hints on the common error restating path."""

import math
import re
import time
from collections.abc import Mapping

from backend.copilot.sdk.model_gateway.errors import (
    GatewayError,
    UpstreamError,
    retry_headers,
)

_TRY_AGAIN = re.compile(r"(?i)try again in (\d+(?:\.\d+)?)\s*(ms|s)\b")


def rate_limited(
    error: UpstreamError, message: str, headers: Mapping[str, str]
) -> GatewayError:
    hints = retry_headers(headers)
    if "retry-after" not in hints and (seconds := _reset_seconds(error)) is not None:
        hints["retry-after"] = str(seconds)
    if (error.code or error.error_type) in (
        "usage_limit_reached",
        "usage_not_included",
    ):
        hints["x-should-retry"] = "false"
    return GatewayError(429, "rate_limit_error", message, headers=hints)


def _reset_seconds(error: UpstreamError) -> int | None:
    seconds = error.fields.get("resets_in_seconds")
    if isinstance(seconds, int) and not isinstance(seconds, bool) and seconds >= 0:
        return seconds
    resets_at = error.fields.get("resets_at")
    if isinstance(resets_at, int) and not isinstance(resets_at, bool):
        return max(0, resets_at - int(time.time()))
    if match := _TRY_AGAIN.search(error.message):
        scale = 0.001 if match.group(2).lower() == "ms" else 1.0
        return math.ceil(float(match.group(1)) * scale)
    return None
