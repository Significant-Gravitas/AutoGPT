"""Tests for run-health helpers in execution_utils."""

from datetime import datetime, timezone

from backend.data.execution import ExecutionStatus, NodeExecutionResult

from .execution_utils import build_run_health_warning, summarize_node_failures

STORE_VALUE_BLOCK_ID = "1ff065e9-88e8-4358-9d82-8dc91f622ba9"


def make_node_execution(
    status: ExecutionStatus,
    output_data: dict | None = None,
    block_id: str = STORE_VALUE_BLOCK_ID,
) -> NodeExecutionResult:
    now = datetime.now(timezone.utc)
    return NodeExecutionResult(
        user_id="user-1",
        graph_id="graph-1",
        graph_version=1,
        graph_exec_id="exec-1",
        node_exec_id="node-exec-1",
        node_id="node-1",
        block_id=block_id,
        status=status,
        input_data={},
        output_data=output_data or {},
        add_time=now,
        queue_time=None,
        start_time=now,
        end_time=now,
    )


def test_summarize_node_failures_extracts_failed_nodes():
    failures = summarize_node_failures(
        [
            make_node_execution(ExecutionStatus.COMPLETED),
            make_node_execution(
                ExecutionStatus.FAILED,
                output_data={"error": ["NameError: name 'variables' is not defined"]},
            ),
        ]
    )

    assert len(failures) == 1
    assert failures[0]["node_id"] == "node-1"
    assert failures[0]["block_name"] == "StoreValueBlock"
    assert failures[0]["error"] == "NameError: name 'variables' is not defined"


def test_summarize_node_failures_handles_missing_error_and_unknown_block():
    failures = summarize_node_failures(
        [make_node_execution(ExecutionStatus.FAILED, block_id="not-a-block-id")]
    )

    assert failures == [
        {"node_id": "node-1", "block_name": "not-a-block-id", "error": None}
    ]


def test_summarize_node_failures_truncates_long_errors():
    failures = summarize_node_failures(
        [
            make_node_execution(
                ExecutionStatus.FAILED, output_data={"error": ["x" * 900]}
            )
        ]
    )

    assert failures[0]["error"] is not None
    assert len(failures[0]["error"]) == 500


def test_build_run_health_warning_reports_failed_nodes():
    warning = build_run_health_warning(
        outputs={"result": ["ok"]},
        node_failures=[
            {"node_id": "n1", "block_name": "CodeExecutionBlock", "error": "boom"}
        ],
    )

    assert warning is not None
    assert "1 node(s) FAILED" in warning
    assert "CodeExecutionBlock: boom" in warning


def test_build_run_health_warning_flags_empty_outputs():
    warning = build_run_health_warning(outputs={}, node_failures=[])

    assert warning is not None
    assert "no outputs" in warning


def test_build_run_health_warning_none_when_healthy():
    assert build_run_health_warning({"result": ["ok"]}, []) is None


def test_incomplete_nodes_are_not_flagged_as_failures():
    """INCOMPLETE nodes are legal in a COMPLETED run — unexecuted branches
    (e.g. a ConditionBlock's untaken path) finish INCOMPLETE by design, so
    flagging them would false-positive every conditional agent. Only FAILED
    nodes indicate a broken run."""
    from unittest.mock import MagicMock

    from backend.data.execution import ExecutionStatus

    ne = MagicMock()
    ne.status = ExecutionStatus.INCOMPLETE
    ne.node_id = "n1"
    ne.block_id = "block-x"
    ne.output_data = {}

    assert summarize_node_failures([ne]) == []


def test_many_failures_capped_detail_but_full_count():
    """The warning lists at most 5 failure details but counts all of them."""
    from unittest.mock import MagicMock

    from backend.data.execution import ExecutionStatus

    nodes = []
    for i in range(8):
        ne = MagicMock()
        ne.status = ExecutionStatus.FAILED
        ne.node_id = f"n{i}"
        ne.block_id = f"block-{i}"
        ne.output_data = {"error": [f"boom {i}"]}
        nodes.append(ne)

    failures = summarize_node_failures(nodes)
    assert len(failures) == 8

    warning = build_run_health_warning({"out": ["x"]}, failures)
    assert warning is not None
    assert "8 node(s) FAILED" in warning
    assert warning.count("boom") == 5
