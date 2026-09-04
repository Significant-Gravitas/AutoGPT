"""Run the arm × task matrix and write the numbers out.

    poetry run python -m backend.copilot.splitbrain.run_experiment \
        --seeds 1 --out /tmp/splitbrain.json

Every run spends real money on the Anthropic API. Nothing here is cached, so the
token counts read as raw context pressure rather than as a billing artefact.
"""

import argparse
import json
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from typing import Any, Callable

import anthropic

from .arms import (
    DEFAULT_CALL_BUDGET,
    RunResult,
    run_baseline,
    run_prompted_delegation,
    run_split,
)
from .tasks import TASKS, TaskSpec

ARMS: dict[str, Callable[..., RunResult]] = {
    "baseline": run_baseline,
    "split-persistent": lambda c, t, b: run_split(c, t, b, "persistent"),
    "split-per_intent": lambda c, t, b: run_split(c, t, b, "per_intent"),
    "prompted-delegation": lambda c, t, b: run_prompted_delegation(c, t, b),
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--tasks", default=",".join(TASKS))
    parser.add_argument("--seeds", type=int, default=1)
    parser.add_argument("--budget", type=int, default=DEFAULT_CALL_BUDGET)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--out", default="/tmp/splitbrain.json")
    args = parser.parse_args(argv)

    logging.disable(logging.CRITICAL)
    client = anthropic.Anthropic()
    jobs = [
        (arm, TASKS[task], seed)
        for seed in range(args.seeds)
        for task in args.tasks.split(",")
        for arm in args.arms.split(",")
    ]
    print(f"{len(jobs)} runs, budget {args.budget} model calls each", file=sys.stderr)

    started = time.monotonic()
    rows: list[dict[str, Any]] = []
    lock = Lock()

    def collect(job) -> dict[str, Any]:
        row = _one_run(client, args.budget, *job)
        # Written as each run lands: a matrix costs real money, and losing an
        # hour of it to a crash at the end is not a risk worth carrying.
        with lock:
            rows.append(row)
            with open(args.out, "w") as handle:
                json.dump(rows, handle, indent=2, default=str)
        return row

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        list(pool.map(collect, jobs))
    print(f"\nwrote {args.out}  ({time.monotonic() - started:.0f}s)", file=sys.stderr)
    print(render_table(rows))
    return 0


def _one_run(
    client: anthropic.Anthropic, budget: int, arm: str, task: TaskSpec, seed: int
) -> dict[str, Any]:
    label = f"{arm}/{task.key}/s{seed}"
    print(f"  start {label}", file=sys.stderr, flush=True)
    try:
        result = ARMS[arm](client, task, budget).as_dict()
    except Exception as exc:
        print(f"  FAILED {label}: {type(exc).__name__}: {exc}", file=sys.stderr)
        return {"arm": arm, "task": task.key, "seed": seed, "error": repr(exc)}
    result["seed"] = seed
    print(
        f"  done  {label}: success={result['success']} "
        f"peak={result['peak_context']} in={result['input_tokens']} "
        f"${result['cost_usd']:.2f}",
        file=sys.stderr,
        flush=True,
    )
    return result


def render_table(rows: list[dict[str, Any]]) -> str:
    header = (
        f"{'arm':<21} {'task':<9} {'ok':<3} {'calls':>5} {'in_tok':>8} "
        f"{'PEAK':>7} {'top_peak':>8} {'$':>6} {'leak':>5}"
    )
    lines = [header, "-" * len(header)]
    for row in sorted(rows, key=lambda r: (r.get("task", ""), r.get("arm", ""))):
        if "error" in row:
            lines.append(f"{row['arm']:<21} {row['task']:<9} ERROR {row['error'][:60]}")
            continue
        lines.append(
            f"{row['arm']:<21} {row['task']:<9} "
            f"{'Y' if row['success'] else 'n':<3} {row['model_calls']:>5} "
            f"{row['input_tokens']:>8} {row['peak_context']:>7} "
            f"{row['reasoner_peak']:>8} {row['cost_usd']:>6.2f} "
            f"{len(row.get('leaked_tool_calls') or []):>5}"
        )
    total = sum(r.get("cost_usd", 0.0) for r in rows)
    lines.append("-" * len(header))
    lines.append(f"total spend ${total:.2f}")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
