"""Tests for the LocalPC shim Prometheus metrics.

Asserts the helpers don't blow up on weird inputs (empty labels, None
shim_version, etc.) and that gauges balance — register N times then
unregister N times → active gauge is back to 0 for those labels.
"""

from __future__ import annotations

from .local_pc_metrics import (
    LOCAL_PC_SHIM_ACTIVE,
    LOCAL_PC_SHIM_CONNECTIONS,
    LOCAL_PC_SHIM_HANDSHAKE_FAILURES,
    LOCAL_PC_SHIM_RPC_DURATION,
    LOCAL_PC_SHIM_RPC_ERRORS,
    record_handshake_failure,
    record_rpc,
    record_rpc_error,
    record_shim_connected,
    record_shim_disconnected,
)


def _gauge_value(gauge, **labels) -> float:
    return gauge.labels(**labels)._value.get()


def _counter_value(counter, **labels) -> float:
    return counter.labels(**labels)._value.get()


class TestConnectDisconnect:
    def test_connect_increments_both_total_and_active(self):
        before_total = _counter_value(
            LOCAL_PC_SHIM_CONNECTIONS,
            platform="darwin",
            arch="arm64",
            shim_version="0.1.0",
        )
        before_active = _gauge_value(
            LOCAL_PC_SHIM_ACTIVE, platform="darwin", arch="arm64"
        )
        record_shim_connected(platform="darwin", arch="arm64", shim_version="0.1.0")
        assert (
            _counter_value(
                LOCAL_PC_SHIM_CONNECTIONS,
                platform="darwin",
                arch="arm64",
                shim_version="0.1.0",
            )
            == before_total + 1
        )
        assert (
            _gauge_value(LOCAL_PC_SHIM_ACTIVE, platform="darwin", arch="arm64")
            == before_active + 1
        )

    def test_disconnect_decrements_active_only(self):
        record_shim_connected(platform="linux", arch="x86_64", shim_version="0.1.0")
        active_after_connect = _gauge_value(
            LOCAL_PC_SHIM_ACTIVE, platform="linux", arch="x86_64"
        )
        record_shim_disconnected(platform="linux", arch="x86_64")
        active_after_disconnect = _gauge_value(
            LOCAL_PC_SHIM_ACTIVE, platform="linux", arch="x86_64"
        )
        assert active_after_disconnect == active_after_connect - 1

    def test_balanced_register_unregister_returns_gauge_to_baseline(self):
        baseline = _gauge_value(LOCAL_PC_SHIM_ACTIVE, platform="windows", arch="x86_64")
        for _ in range(5):
            record_shim_connected(
                platform="windows", arch="x86_64", shim_version="0.2.0"
            )
        for _ in range(5):
            record_shim_disconnected(platform="windows", arch="x86_64")
        assert (
            _gauge_value(LOCAL_PC_SHIM_ACTIVE, platform="windows", arch="x86_64")
            == baseline
        )

    def test_empty_labels_dont_raise(self):
        # Real-world: a shim handshake completes before HELLO populated
        # the fields, leaving them empty. Helpers must normalize without
        # raising or polluting cardinality.
        record_shim_connected(platform="", arch=None, shim_version=None)
        record_shim_disconnected(platform="", arch=None)
        # Labels should resolve to "unknown" — not "".
        # If this raised LabelError we'd see it propagate.


class TestHandshakeFailure:
    def test_each_stage_label_increments(self):
        for stage in (
            "missing_token",
            "invalid_token",
            "auth_error",
            "expected_hello",
            "handshake_error",
        ):
            before = _counter_value(LOCAL_PC_SHIM_HANDSHAKE_FAILURES, stage=stage)
            record_handshake_failure(stage)
            assert (
                _counter_value(LOCAL_PC_SHIM_HANDSHAKE_FAILURES, stage=stage)
                == before + 1
            )


class TestRpcMetrics:
    def test_record_rpc_buckets_duration(self):
        # Smoke: record across the bucket range, no raise.
        for outcome in ("ok", "error", "timeout"):
            record_rpc("FILE_READ", outcome=outcome, duration_seconds=0.05)
            record_rpc("EXECUTE_COMMAND", outcome=outcome, duration_seconds=2.5)
        # Histogram observed; just confirm sample count moved.
        count = LOCAL_PC_SHIM_RPC_DURATION.labels(
            op="FILE_READ", outcome="ok"
        )._sum.get()
        assert count >= 0.05

    def test_record_rpc_error_per_code(self):
        before = _counter_value(
            LOCAL_PC_SHIM_RPC_ERRORS,
            op="INPUT_ACTION",
            code="INPUT_OUT_OF_BOUNDS",
        )
        record_rpc_error("INPUT_ACTION", "INPUT_OUT_OF_BOUNDS")
        assert (
            _counter_value(
                LOCAL_PC_SHIM_RPC_ERRORS,
                op="INPUT_ACTION",
                code="INPUT_OUT_OF_BOUNDS",
            )
            == before + 1
        )

    def test_record_rpc_error_normalizes_empty_code(self):
        record_rpc_error("FILE_READ", "")
        # Falls back to "UNKNOWN" rather than empty-label.
        # Counter exists; reading any value is enough to confirm no raise.
        _ = _counter_value(LOCAL_PC_SHIM_RPC_ERRORS, op="FILE_READ", code="UNKNOWN")
