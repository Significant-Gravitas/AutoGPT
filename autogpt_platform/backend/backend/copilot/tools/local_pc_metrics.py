"""Prometheus metrics for the LocalPC shim integration.

Surfaces just enough for an operator dashboard answering: how many shims
are connected, what platforms are they, how often do they handshake,
what's the error landscape. Per-op latency + per-error-code histograms
are layered on later via ``LocalPCShim._rpc`` once the parallel
adapter-wiring work settles.

## Metric names (consumed by infra/grafana dashboards)

- ``copilot_localpc_shim_connections_total`` (Counter) — total HELLO
  handshakes that registered a shim. Labels: ``platform``, ``arch``,
  ``shim_version`` (normalized to major.minor).
- ``copilot_localpc_shim_handshake_failures_total`` (Counter) — total
  handshakes that failed before ``register()``. Labels: ``stage``
  (``"missing_token"`` | ``"invalid_token"`` | ``"auth_error"`` |
  ``"expected_hello"`` | ``"handshake_error"``).
- ``copilot_localpc_shim_active`` (Gauge) — currently-connected shims
  on this worker. Labels: ``platform``, ``arch``. Reset to 0 on worker
  restart.
- ``copilot_localpc_shim_rpc_duration_seconds`` (Histogram) — wall
  time per wire op roundtrip. Labels: ``op`` (wire message type),
  ``outcome`` (``"ok"`` | ``"error"`` | ``"timeout"``). Wired in
  ``LocalPCShim._rpc`` in a follow-up commit; the metric definition
  lives here so the infra dashboard JSON can reference it ahead of
  the instrumentation landing.
- ``copilot_localpc_shim_rpc_errors_total`` (Counter) — error-code
  distribution. Labels: ``op``, ``code`` (the shim's wire error code
  enum value).

Per-worker accuracy is fine for these — Prometheus scrapes each pod
and aggregation at the Grafana side gives the global view. The
``platform``/``arch`` labels are bounded enums (4 × 2 = 8 series), so
cardinality is safe. ``shim_version`` is normalized against an explicit set of
supported major.minor release lines so user-controlled versions cannot create
an unbounded number of time series. Wire ops, outcomes, handshake stages, and
error codes are also allowlisted before use as labels.
"""

from __future__ import annotations

import re

from prometheus_client import Counter, Gauge, Histogram

from .local_pc_relay_protocol import PLATFORM_REQUEST_TYPES

_PLATFORMS = frozenset({"darwin", "linux", "windows", "wsl2"})
_ARCHITECTURES = frozenset({"x86_64", "arm64"})
_SUPPORTED_SHIM_RELEASE_LINES = frozenset({"0.0", "0.1"})
_HANDSHAKE_STAGES = frozenset(
    {
        "missing_token",
        "invalid_token",
        "session_access_denied",
        "feature_disabled",
        "auth_error",
        "hello_too_large",
        "expected_hello",
        "invalid_hello_id",
        "invalid_hello",
        "protocol_version_mismatch",
        "hello_timeout",
        "handshake_error",
    }
)
_RPC_OUTCOMES = frozenset({"ok", "error", "timeout"})
_SHIM_ERROR_CODES = frozenset(
    {
        "PATH_OUTSIDE_ALLOWED_ROOT",
        "PATH_RESERVED_NAME",
        "PATH_INVALID_CHARS",
        "PATH_NOT_FOUND",
        "PATH_NOT_EMPTY",
        "PATH_EXISTS",
        "COMMAND_TIMEOUT",
        "SHELL_NOT_AVAILABLE",
        "UNSUPPORTED_ARCH",
        "CAPABILITY_NOT_GRANTED",
        "AUTH_FAILED",
        "SHIM_OVERLOADED",
        "INTERNAL_ERROR",
        "FILE_TOO_LARGE",
        "DEPENDENCY_MISSING",
        "WINDOW_STALE",
        "PERMISSION_PENDING",
        "FEATURE_NOT_SUPPORTED",
        "CLIPBOARD_CONCEALED",
        "INPUT_OUT_OF_BOUNDS",
        "MODEL_NOT_AVAILABLE",
        "LOCAL_LLM_BUSY",
        "LOCAL_LLM_FAILED",
        "RECORDING_NOT_FOUND",
        "RECORDING_CHANNEL_UNAVAILABLE",
        "RECORDING_ALREADY_ACTIVE",
        "RECORDING_REVIEW_INVALID",
        "CONSENT_REQUIRED",
        "INTERPRETATION_UNAVAILABLE",
    }
)

LOCAL_PC_SHIM_CONNECTIONS = Counter(
    "copilot_localpc_shim_connections_total",
    "Total HELLO handshakes that successfully registered a LocalPC shim.",
    labelnames=("platform", "arch", "shim_version"),
)

LOCAL_PC_SHIM_HANDSHAKE_FAILURES = Counter(
    "copilot_localpc_shim_handshake_failures_total",
    "Handshakes that failed before register(). Use `stage` to slice the funnel.",
    labelnames=("stage",),
)

LOCAL_PC_SHIM_ACTIVE = Gauge(
    "copilot_localpc_shim_active",
    "Currently-connected LocalPC shims on this worker.",
    labelnames=("platform", "arch"),
)

LOCAL_PC_SHIM_RPC_DURATION = Histogram(
    "copilot_localpc_shim_rpc_duration_seconds",
    "LocalPC shim wire-op roundtrip latency.",
    labelnames=("op", "outcome"),
    # Buckets sized for sub-ms (cursor_position) up to multi-second
    # (large file reads, screenshot capture on big displays).
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0),
)

LOCAL_PC_SHIM_RPC_ERRORS = Counter(
    "copilot_localpc_shim_rpc_errors_total",
    "Errors returned by the shim per wire op + error code.",
    labelnames=("op", "code"),
)

LOCAL_PC_SHIM_RPC_RETRIES = Counter(
    "copilot_localpc_shim_rpc_retries_total",
    "Idempotent ops that hit the in-flight-disconnect path and got auto-retried "
    "by the adapter. Slice by `outcome` to see how often the retry recovered.",
    labelnames=(
        "op",
        "outcome",
    ),  # outcome: "success_after_retry" | "failed_after_retry"
)


def _normalize_enum_label(
    value: str | None,
    allowed: frozenset[str],
    *,
    fallback: str = "unknown",
    uppercase: bool = False,
) -> str:
    """Map untrusted metric labels into a finite allowlisted vocabulary."""
    normalized = (value or "").strip()
    normalized = normalized.upper() if uppercase else normalized.lower()
    return normalized if normalized in allowed else fallback


def _normalize_version_label(value: str | None) -> str:
    """Bucket versions to explicitly supported release lines."""
    match = re.match(r"^v?(\d+)\.(\d+)(?:\D|$)", (value or "").strip())
    if match is None:
        return "unknown"
    release_line = f"{match.group(1)}.{match.group(2)}"
    return release_line if release_line in _SUPPORTED_SHIM_RELEASE_LINES else "unknown"


def record_shim_connected(
    *, platform: str | None, arch: str | None, shim_version: str | None
) -> None:
    """Record a successful HELLO/HELLO_ACK roundtrip + register."""
    platform_label = _normalize_enum_label(platform, _PLATFORMS)
    arch_label = _normalize_enum_label(arch, _ARCHITECTURES)
    LOCAL_PC_SHIM_CONNECTIONS.labels(
        platform=platform_label,
        arch=arch_label,
        shim_version=_normalize_version_label(shim_version),
    ).inc()
    LOCAL_PC_SHIM_ACTIVE.labels(
        platform=platform_label,
        arch=arch_label,
    ).inc()


def record_shim_disconnected(*, platform: str | None, arch: str | None) -> None:
    """Record a shim WS close after a successful registration."""
    LOCAL_PC_SHIM_ACTIVE.labels(
        platform=_normalize_enum_label(platform, _PLATFORMS),
        arch=_normalize_enum_label(arch, _ARCHITECTURES),
    ).dec()


def record_handshake_failure(stage: str) -> None:
    """Bump the handshake-funnel failure counter at a specific stage."""
    LOCAL_PC_SHIM_HANDSHAKE_FAILURES.labels(
        stage=_normalize_enum_label(stage, _HANDSHAKE_STAGES)
    ).inc()


def record_rpc(op: str, *, outcome: str, duration_seconds: float) -> None:
    """Record a wire-op roundtrip. Called from ``LocalPCShim._rpc``."""
    LOCAL_PC_SHIM_RPC_DURATION.labels(
        op=_normalize_enum_label(
            op, PLATFORM_REQUEST_TYPES, fallback="UNKNOWN", uppercase=True
        ),
        outcome=_normalize_enum_label(outcome, _RPC_OUTCOMES),
    ).observe(duration_seconds)


def record_rpc_error(op: str, code: str) -> None:
    """Bump the per-(op, code) error counter."""
    LOCAL_PC_SHIM_RPC_ERRORS.labels(
        op=_normalize_enum_label(
            op, PLATFORM_REQUEST_TYPES, fallback="UNKNOWN", uppercase=True
        ),
        code=_normalize_enum_label(
            code, _SHIM_ERROR_CODES, fallback="UNKNOWN", uppercase=True
        ),
    ).inc()


def record_rpc_retry(op: str, *, recovered: bool) -> None:
    """Bump the retry counter after an idempotent op's auto-retry resolves.

    Distinguishes recovered (the retry succeeded; LLM never saw an error)
    from failed (retry also failed; the caller raised). High
    `success_after_retry` rate ÷ low `failed_after_retry` is the healthy
    shape — high failed_after_retry means the shim's having a bad time.
    """
    LOCAL_PC_SHIM_RPC_RETRIES.labels(
        op=_normalize_enum_label(
            op, PLATFORM_REQUEST_TYPES, fallback="UNKNOWN", uppercase=True
        ),
        outcome="success_after_retry" if recovered else "failed_after_retry",
    ).inc()
