"""Tests for Prometheus instrumentation helpers."""


def test_record_graph_run_completion_increments_by_status():
    from prometheus_client import REGISTRY

    from backend.monitoring.instrumentation import record_graph_run_completion

    def value(status: str) -> float:
        return (
            REGISTRY.get_sample_value(
                "autogpt_graph_run_completions_total", {"status": status}
            )
            or 0.0
        )

    before_ok, before_fail = value("COMPLETED"), value("FAILED")
    record_graph_run_completion("COMPLETED")
    record_graph_run_completion("FAILED")
    record_graph_run_completion("FAILED")
    assert value("COMPLETED") == before_ok + 1
    assert value("FAILED") == before_fail + 2
