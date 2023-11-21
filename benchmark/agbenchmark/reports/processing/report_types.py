from typing import Any, Dict, List, Union

from pydantic import BaseModel, Field

datetime_format = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$"
from pydantic import BaseModel, constr


class ForbidOptionalMeta(type(BaseModel)):  # metaclass to forbid optional fields
    def __new__(cls, name: str, bases: tuple, dct: Dict[str, Any]) -> Any:
        for attr_name, attr_value in dct.items():
            if (
                getattr(attr_value, "__origin__", None) == Union
                and type(None) in attr_value.__args__
            ):
                raise TypeError(
                    f"Optional fields are forbidden, but found in {attr_name}"
                )

        return super().__new__(cls, name, bases, dct)


class BaseModelBenchmark(BaseModel, metaclass=ForbidOptionalMeta):
    class Config:
        extra = "forbid"


class Metrics(BaseModelBenchmark):
    difficulty: str
    success: bool
    success_percentage: float = Field(..., alias="success_%")
    run_time: str
    fail_reason: str | None
    attempted: bool
    cost: float | None


class MetricsOverall(BaseModelBenchmark):
    run_time: str
    highest_difficulty: str
    percentage: float | None
    total_cost: float | None


class Test(BaseModelBenchmark):
    data_path: str
    is_regression: bool
    answer: str
    description: str
    metrics: Metrics
    category: List[str]
    task: str
    reached_cutoff: bool
    metadata: Any


class ReportBase(BaseModelBenchmark):
    command: str
    completion_time: str | None
    benchmark_start_time: constr(regex=datetime_format)
    metrics: MetricsOverall
    config: Dict[str, str | dict[str, str]]
    agent_git_commit_sha: str | None
    benchmark_git_commit_sha: str | None
    repo_url: str | None


class Report(ReportBase):
    tests: Dict[str, Test]


class ReportV2(Test, ReportBase):
    test_name: str
    run_id: str | None
    team_name: str | None
