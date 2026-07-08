"""Token-economics comparison for the two-phase planner/executor split.

Pure roll-up functions (no I/O) that turn a set of per-turn
:class:`TurnTokenBreakdown` samples (from flag-ON runs) plus a set of flag-OFF
total-token samples into a side-by-side report answering:

  * how many tokens the **planner** call costs,
  * how many the **executor** loop costs,
  * how many a **normal** (non-split) run costs, and
  * the split's overall **overhead** vs. that normal run.

Feed it breakdowns captured from real turns (``state.token_breakdown()`` /
the ``planner/executor token breakdown`` log line) or hand-built fixtures.
Tests pin known-answer cases.
"""

from __future__ import annotations

from typing import Sequence

from backend.copilot.planner.models import TurnTokenBreakdown

from .metrics import mean


def _pct(part: float, whole: float) -> float:
    """``part`` as a percentage of ``whole``; 0.0 when ``whole`` is 0."""
    if whole == 0:
        return 0.0
    return round(100.0 * part / whole, 2)


def summarize_breakdowns(
    breakdowns: Sequence[TurnTokenBreakdown],
) -> dict[str, float]:
    """Mean per-phase token counts across flag-ON turns."""
    return {
        "n": len(breakdowns),
        "mean_planner_tokens": mean([b.planner_tokens for b in breakdowns]),
        "mean_executor_tokens": mean([b.executor_tokens for b in breakdowns]),
        "mean_replan_tokens": mean([b.replan_tokens for b in breakdowns]),
        "mean_total_tokens": mean([b.total_tokens for b in breakdowns]),
        "mean_overhead_tokens": mean([b.overhead_tokens for b in breakdowns]),
    }


def compare_planner_executor(
    *,
    baseline_totals: Sequence[int],
    split_breakdowns: Sequence[TurnTokenBreakdown],
) -> dict[str, object]:
    """Compare split (flag-ON) vs. single-loop (flag-OFF) token usage.

    ``baseline_totals`` are total tokens per flag-OFF turn on the same task(s);
    ``split_breakdowns`` are the per-phase breakdowns of the flag-ON turns.

    Returns a report with ``baseline`` / ``split`` / ``comparison`` sections.
    Key comparison fields:
      * ``split_total_vs_baseline_pct`` — how much more (or less, if negative)
        the whole split turn costs vs. a normal run.
      * ``overhead_share_pct`` — share of the split turn spent planning +
        re-planning (the tax the split adds).
      * ``executor_vs_baseline_pct`` — did running the loop *with a plan on the
        cheaper model* make the loop itself cheaper than a normal run?
    """
    baseline_mean = mean([float(t) for t in baseline_totals])
    split = summarize_breakdowns(split_breakdowns)
    split_total = float(split["mean_total_tokens"])
    executor_mean = float(split["mean_executor_tokens"])
    overhead_mean = float(split["mean_overhead_tokens"])

    return {
        "baseline": {
            "n": len(baseline_totals),
            "mean_total_tokens": baseline_mean,
        },
        "split": split,
        "comparison": {
            "split_total_vs_baseline_pct": _pct(
                split_total - baseline_mean, baseline_mean
            ),
            "overhead_share_pct": _pct(overhead_mean, split_total),
            "executor_vs_baseline_pct": _pct(
                executor_mean - baseline_mean, baseline_mean
            ),
        },
    }
