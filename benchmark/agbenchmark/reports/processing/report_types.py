from typing import Any, Dict, List

from pydantic import BaseModel, Field, constr, validator

datetime_format = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$"


class Metrics(BaseModel):
    difficulty: str | None
    success: bool | None = None
    run_time: str | None = None
    fail_reason: str | None = None
    success_percentage: float | None = Field(default=None, alias="success_%")
    attempted: bool
    cost: float | None = None

    @validator("attempted")
    def require_metrics_if_attempted(cls, v: bool, values: dict[str, Any]):
        required_fields_if_attempted = ["success", "run_time"]
        if v:
            for f in required_fields_if_attempted:
                assert (
                    values.get(f) is not None
                ), f"'{f}' must be defined if attempted is True"
        return v


class MetricsOverall(BaseModel):
    run_time: str
    highest_difficulty: str
    percentage: float | None = None
    total_cost: float | None = None


class Test(BaseModel):
    data_path: str
    is_regression: bool
    answer: str
    description: str
    metrics: Metrics
    category: List[str]
    task: str
    reached_cutoff: bool | None = None  # None if in progress
    metadata: dict[str, Any] | None = Field(default_factory=dict)


class ReportBase(BaseModel):
    command: str
    completion_time: str | None = None
    benchmark_start_time: constr(regex=datetime_format)
    metrics: MetricsOverall
    config: Dict[str, str | dict[str, str]]
    agent_git_commit_sha: str | None = None
    benchmark_git_commit_sha: str | None = None
    repo_url: str | None = None


class Report(ReportBase):
    tests: Dict[str, Test]


class ReportV2(Test, ReportBase):
    test_name: str
    run_id: str | None
    team_name: str | None
