"""Generate benchmark reports compatible with existing format."""

import json
from datetime import datetime, timezone
from pathlib import Path

from .models import ChallengeResult


class ReportGenerator:
    """Generate benchmark reports compatible with existing agbenchmark format."""

    def __init__(self, reports_dir: Path):
        self.reports_dir = reports_dir
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        results: list[ChallengeResult],
        config_name: str,
        start_time: datetime,
        end_time: datetime,
    ) -> Path:
        """Generate a report.json file for a benchmark run.

        Args:
            results: List of challenge results for this configuration.
            config_name: Name of the configuration (e.g., "one_shot/claude").
            start_time: When the benchmark started.
            end_time: When the benchmark ended.

        Returns:
            Path to the generated report file.
        """
        # Build test results
        tests = {}
        total_cost = 0.0
        highest_difficulty = "interface"
        _difficulty_order = [  # noqa: F841
            "interface",
            "basic",
            "intermediate",
            "advanced",
            "hard",
        ]

        for result in results:
            total_cost += result.cost

            tests[result.challenge_name] = {
                "category": [],
                "difficulty": None,
                "data_path": "",
                "description": "",
                "task": "",
                "answer": "",
                "metrics": {
                    "attempted": True,
                    "is_regression": False,
                    "success_percentage": 100.0 if result.success else 0.0,
                },
                "results": [
                    {
                        "success": result.success,
                        "run_time": f"{result.run_time_seconds:.3f} seconds",
                        "fail_reason": (
                            result.error_message if not result.success else None
                        ),
                        "reached_cutoff": result.timed_out,
                        "n_steps": result.n_steps,
                        "cost": result.cost,
                        "steps": [
                            {
                                "step_num": step.step_num,
                                "tool_name": step.tool_name,
                                "tool_args": step.tool_args,
                                "result": step.result,
                                "is_error": step.is_error,
                                "cost": step.cumulative_cost,
                            }
                            for step in result.steps
                        ],
                    }
                ],
            }

        run_time = (end_time - start_time).total_seconds()

        report = {
            "command": f"direct_benchmark run --config {config_name}",
            "completion_time": end_time.astimezone(timezone.utc).isoformat(),
            "benchmark_start_time": start_time.astimezone(timezone.utc).isoformat(),
            "metrics": {
                "run_time": f"{run_time:.2f} seconds",
                "highest_difficulty": highest_difficulty,
                "total_cost": total_cost,
            },
            "config": {
                "config_name": config_name,
            },
            "tests": tests,
        }

        # Create report directory
        timestamp = start_time.strftime("%Y%m%dT%H%M%S")
        safe_config = config_name.replace("/", "_")
        report_dir = self.reports_dir / f"{timestamp}_{safe_config}"
        report_dir.mkdir(parents=True, exist_ok=True)

        report_file = report_dir / "report.json"
        with open(report_file, "w") as f:
            json.dump(report, f, indent=4)

        return report_file

    def generate_comparison_report(
        self,
        all_results: dict[str, list[ChallengeResult]],
        timestamp: datetime,
    ) -> Path:
        """Generate a comparison report across all configurations.

        Args:
            all_results: Dict mapping config_name -> list of results.
            timestamp: Timestamp for the report.

        Returns:
            Path to the comparison report file.
        """
        test_names: set[str] = set()

        comparison = {
            "timestamp": timestamp.isoformat(),
            "configurations": list(all_results.keys()),
            "results": {},
            "test_names": [],
        }

        for config_name, results in all_results.items():
            passed = sum(1 for r in results if r.success)
            total = len(results)
            total_cost = sum(r.cost for r in results)
            total_steps = sum(r.n_steps for r in results)

            comparison["results"][config_name] = {
                "tests_run": total,
                "tests_passed": passed,
                "tests_failed": total - passed,
                "success_rate": (passed / total * 100) if total > 0 else 0,
                "total_cost": total_cost,
                "avg_steps": total_steps / total if total > 0 else 0,
                "test_results": {
                    r.challenge_name: {
                        "success": r.success,
                        "n_steps": r.n_steps,
                        "cost": r.cost,
                        "error": r.error_message,
                        "timed_out": r.timed_out,
                        "steps": [
                            {
                                "step": s.step_num,
                                "tool": s.tool_name,
                                "args": s.tool_args,
                                "result": (
                                    s.result[:500] + "..."
                                    if len(s.result) > 500
                                    else s.result
                                ),
                                "error": s.is_error,
                            }
                            for s in r.steps
                        ],
                    }
                    for r in results
                },
            }
            test_names.update(r.challenge_name for r in results)

        comparison["test_names"] = sorted(test_names)

        filename = f"strategy_comparison_{timestamp.strftime('%Y%m%dT%H%M%S')}.json"
        report_path = self.reports_dir / filename

        with open(report_path, "w") as f:
            json.dump(comparison, f, indent=2)

        return report_path
