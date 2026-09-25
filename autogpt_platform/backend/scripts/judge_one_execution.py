"""Run the TypeSafe Jev run judge once, end to end, and print the wire path.

Uses the real ``_build_execution_summary`` and ``judge_execution`` with the
platform ``TYPESAFE_API_KEY``. By default it judges a fixture execution (no
database needed); ``--execution <graph_exec_id>`` loads a real one through the
database manager when the stack is running.

    poetry run python scripts/judge_one_execution.py
    poetry run python scripts/judge_one_execution.py --failed
    poetry run python scripts/judge_one_execution.py --execution <id>

Bounded: one Jev call, then exit.
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from types import SimpleNamespace

from backend.data.execution import ExecutionStatus, NodeExecutionResult
from backend.data.model import GraphExecutionStats
from backend.executor.activity_status_generator import _build_execution_summary
from backend.executor.run_judge import JudgeResult, judge_execution
from backend.util.clients import get_database_manager_async_client
from backend.util.settings import Settings

INPUT_BLOCK_ID = "c0a8e994-ebf1-4a9c-a4d8-89d09c86741b"  # AgentInputBlock
LLM_BLOCK_ID = "1f292d4a-41a4-4977-9684-7c8d560b9f91"  # AITextGeneratorBlock
OUTPUT_BLOCK_ID = "363ae599-353e-4804-937e-b2ee3cef3da4"  # AgentOutputBlock

NODE_INPUT = "11111111-0000-0000-0000-000000000001"
NODE_LLM = "22222222-0000-0000-0000-000000000002"
NODE_OUTPUT = "33333333-0000-0000-0000-000000000003"


def _node(
    node_id: str, block_id: str, status: ExecutionStatus, inputs: dict, outputs: dict
) -> NodeExecutionResult:
    return NodeExecutionResult(
        user_id="local-user",
        graph_id="fixture-graph",
        graph_version=1,
        graph_exec_id="fixture-exec",
        node_exec_id=f"{node_id[:8]}-exec",
        node_id=node_id,
        block_id=block_id,
        status=status,
        input_data=inputs,
        output_data=outputs,
        add_time=datetime.now(timezone.utc),
        queue_time=None,
        start_time=None,
        end_time=None,
    )


def fixture(
    failed: bool,
) -> tuple[list[NodeExecutionResult], GraphExecutionStats, list]:
    summary_text = (
        "Release 4.2 ships three changes: (1) the new scheduler retries failed "
        "runs up to three times with exponential backoff, (2) the library page "
        "now shows a correctness score per run, and (3) Slack notifications can "
        "be scoped per agent. Upgrade requires no migration."
    )
    nodes = [
        _node(
            NODE_INPUT,
            INPUT_BLOCK_ID,
            ExecutionStatus.COMPLETED,
            {"name": "release_notes", "value": "raw changelog text ..."},
            {"result": ["raw changelog text ..."]},
        ),
        _node(
            NODE_LLM,
            LLM_BLOCK_ID,
            ExecutionStatus.FAILED if failed else ExecutionStatus.COMPLETED,
            {"prompt": "Summarize these release notes for customers: ..."},
            (
                {"error": ["OpenRouter: 401 Unauthorized - invalid API key"]}
                if failed
                else {"response": [summary_text]}
            ),
        ),
    ]
    if not failed:
        nodes.append(
            _node(
                NODE_OUTPUT,
                OUTPUT_BLOCK_ID,
                ExecutionStatus.COMPLETED,
                {"name": "summary", "value": summary_text},
                {"output": [summary_text]},
            )
        )
    stats = GraphExecutionStats(
        walltime=6.2,
        node_count=len(nodes),
        node_error_count=1 if failed else 0,
        error="Node failed: invalid API key" if failed else None,
    )
    links = [
        SimpleNamespace(
            source_id=NODE_INPUT,
            sink_id=NODE_LLM,
            source_name="result",
            sink_name="prompt",
            is_static=False,
        ),
        SimpleNamespace(
            source_id=NODE_LLM,
            sink_id=NODE_OUTPUT,
            source_name="response",
            sink_name="value",
            is_static=False,
        ),
    ]
    return nodes, stats, links


async def load_real(graph_exec_id: str, user_id: str):
    db = get_database_manager_async_client()
    meta = await db.get_graph_execution_meta(
        user_id=user_id, execution_id=graph_exec_id
    )
    if meta is None:
        raise SystemExit(f"execution {graph_exec_id} not found")
    nodes = await db.get_node_executions(graph_exec_id, include_exec_data=True)
    graph_meta = await db.get_graph_metadata(meta.graph_id, meta.graph_version)
    graph = await db.get_graph(
        graph_id=meta.graph_id,
        version=meta.graph_version,
        user_id=meta.user_id,
        skip_access_check=True,
    )
    stats = meta.stats.to_db() if meta.stats else GraphExecutionStats()
    name = graph_meta.name if graph_meta else meta.graph_id
    description = graph_meta.description if graph_meta else ""
    return nodes, stats, (graph.links if graph else []), name, description, meta.status


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--failed", action="store_true", help="use the failed fixture")
    parser.add_argument("--execution", help="judge a real graph execution id")
    parser.add_argument("--user", help="owner user id of --execution")
    parser.add_argument("--mode", default="shadow", choices=["shadow", "primary"])
    args = parser.parse_args()

    settings = Settings()
    api_key = settings.secrets.typesafe_api_key
    if not api_key:
        print("TYPESAFE_API_KEY is not set in backend/.env", file=sys.stderr)
        return 2

    if args.execution:
        if not args.user:
            print("--execution requires --user", file=sys.stderr)
            return 2
        nodes, stats, links, name, description, status = await load_real(
            args.execution, args.user
        )
    else:
        nodes, stats, links = fixture(args.failed)
        name = "Release notes summarizer"
        description = (
            "Turns a raw changelog into a short customer-facing release summary."
        )
        status = ExecutionStatus.FAILED if args.failed else ExecutionStatus.COMPLETED

    evidence = _build_execution_summary(nodes, stats, name, description, links, status)
    print("=== EVIDENCE (Jev state) ===")
    print(json.dumps(evidence, indent=2))

    result: JudgeResult = await judge_execution(
        evidence,
        stats,
        status,
        api_key=api_key,
        mode=args.mode,
        timeout_seconds=settings.config.run_judge_timeout_seconds,
    )
    print("\n=== VERBATIM REQUEST ===")
    print(result.request)
    print("\n=== VERBATIM RESPONSE ===")
    print(result.response)
    print("\n=== SUMMARY ===")
    print(
        json.dumps(
            {
                "source": result.source,
                "error": result.error,
                "request_id": result.request_id,
                "latency_ms": result.latency_ms,
                "input_tokens": result.input_tokens,
                "output_tokens": result.output_tokens,
                "truncated": result.truncated,
                "derived_correctness_score": result.derived_correctness_score,
                "choices": {
                    key: (answer.get("choice", answer.get("score")))
                    for key, answer in result.answers.items()
                },
            },
            indent=2,
        )
    )
    return 0 if result.ok else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
