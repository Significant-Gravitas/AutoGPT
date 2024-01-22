"""
Model definitions used internally and for reports generated during command-line runs.
"""

from typing import Any, Dict, List

from pydantic import BaseModel, Field, constr, validator

datetime_format = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$"


class TestResult(BaseModel):
    """Result details for a single run of a test/challenge."""

    success: bool | None = None
    """Whether the run was successful"""
    run_time: str | None = None
    """The (formatted) duration of the run"""
    fail_reason: str | None = None
    """If applicable, the reason why the run was not successful"""
    reached_cutoff: bool | None = None  # None if in progress
    """Whether the run had to be stopped due to reaching the timeout"""
    cost: float | None = None
    """The (known) cost incurred by the run, e.g. from using paid LLM APIs"""

    @validator("fail_reason")
    def success_xor_fail_reason(cls, v: str | None, values: dict[str, Any]):
        if v:
            success = values["success"]
            assert not success, "fail_reason must only be specified if success=False"
        else:
            assert values["success"], "fail_reason is required if success=False"
        return v


class TestMetrics(BaseModel):
    """
    Result metrics for a set of runs for a test/challenge. Should be an aggregate of all
    results for the same test/challenge within a benchmarking session.
    """

    attempted: bool
    """Whether the challenge was attempted during this session"""
    is_regression: bool
    """Whether the challenge was considered a regression test at the time of running"""
    success_percentage: float | None = Field(default=None, alias="success_%")
    """Success rate (0-100) for this challenge within the session"""


class MetricsOverall(BaseModel):
    """Global metrics concerning a benchmarking session"""

    run_time: str
    """Duration from beginning to end of the session"""
    highest_difficulty: str
    """
    Difficulty of the most difficult challenge that succeeded at least once this session
    """
    total_cost: float | None = None
    """Total known cost of the session"""


class Test(BaseModel):
    category: List[str]
    difficulty: str | None
    data_path: str
    description: str
    task: str
    answer: str
    metrics: TestMetrics
    results: list[TestResult]
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
