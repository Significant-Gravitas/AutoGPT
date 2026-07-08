"""Known-answer tests for the planner/executor token-comparison suite."""

from backend.copilot.planner.models import TurnTokenBreakdown

from .planner_executor_suite import compare_planner_executor, summarize_breakdowns


def _bd(planner=(0, 0), executor=(0, 0), replan=(0, 0)) -> TurnTokenBreakdown:
    return TurnTokenBreakdown(
        planner_prompt_tokens=planner[0],
        planner_completion_tokens=planner[1],
        executor_prompt_tokens=executor[0],
        executor_completion_tokens=executor[1],
        replan_prompt_tokens=replan[0],
        replan_completion_tokens=replan[1],
    )


class TestTurnTokenBreakdown:
    def test_bucket_and_total_properties(self):
        bd = _bd(planner=(100, 20), executor=(400, 80), replan=(30, 10))
        assert bd.planner_tokens == 120
        assert bd.executor_tokens == 480
        assert bd.replan_tokens == 40
        assert bd.total_tokens == 640
        # Overhead = everything outside the executor loop.
        assert bd.overhead_tokens == 160
        assert bd.total_prompt_tokens == 530
        assert bd.total_completion_tokens == 110

    def test_cost_total_ignores_none(self):
        bd = TurnTokenBreakdown(planner_cost_usd=0.05, executor_cost_usd=0.02)
        assert bd.total_cost_usd == 0.07
        assert TurnTokenBreakdown().total_cost_usd is None


class TestSummarizeBreakdowns:
    def test_means(self):
        s = summarize_breakdowns(
            [
                _bd(planner=(100, 0), executor=(400, 0)),
                _bd(planner=(200, 0), executor=(600, 0), replan=(50, 0)),
            ]
        )
        assert s["n"] == 2
        assert s["mean_planner_tokens"] == 150
        assert s["mean_executor_tokens"] == 500
        assert s["mean_replan_tokens"] == 25
        assert s["mean_total_tokens"] == 675
        assert s["mean_overhead_tokens"] == 175

    def test_empty_is_zeroed(self):
        s = summarize_breakdowns([])
        assert s["n"] == 0
        assert s["mean_total_tokens"] == 0.0


class TestComparePlannerExecutor:
    def test_full_report(self):
        # Baseline (flag OFF): two normal runs averaging 500 tokens.
        # Split (flag ON): planner=120, executor=480, replan=0 → total 600.
        report = compare_planner_executor(
            baseline_totals=[400, 600],
            split_breakdowns=[_bd(planner=(100, 20), executor=(400, 80))],
        )
        assert report["baseline"]["mean_total_tokens"] == 500
        assert report["split"]["mean_planner_tokens"] == 120
        assert report["split"]["mean_executor_tokens"] == 480
        assert report["split"]["mean_total_tokens"] == 600
        comp = report["comparison"]
        # Split total 600 vs baseline 500 → +20%.
        assert comp["split_total_vs_baseline_pct"] == 20.0
        # Overhead 120 / 600 → 20%.
        assert comp["overhead_share_pct"] == 20.0
        # Executor 480 vs baseline 500 → -4% (loop got slightly cheaper).
        assert comp["executor_vs_baseline_pct"] == -4.0

    def test_empty_baseline_no_zero_division(self):
        report = compare_planner_executor(
            baseline_totals=[],
            split_breakdowns=[_bd(planner=(10, 0), executor=(90, 0))],
        )
        assert report["baseline"]["mean_total_tokens"] == 0.0
        assert report["comparison"]["split_total_vs_baseline_pct"] == 0.0
        # Overhead share still computes off the split total.
        assert report["comparison"]["overhead_share_pct"] == 10.0
