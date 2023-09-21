from typing import Dict, List

datetime_format = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$"
from pydantic import BaseModel, constr


class BaseModelBenchmark(BaseModel):
    class Config:
        extra = "forbid"


class TaskInfo(BaseModelBenchmark):
    data_path: str
    is_regression: bool | None
    answer: str
    description: str
    category: List[str]
    task: str


class RepositoryInfo(BaseModelBenchmark):
    repo_url: str | None
    team_name: str | None
    benchmark_git_commit_sha: str | None
    agent_git_commit_sha: str | None


class Metrics(BaseModelBenchmark):
    difficulty: str | None
    success: bool
    success_percentage: float | None
    run_time: str | None
    fail_reason: str | None
    attempted: bool
    cost: float | None


class RunDetails(BaseModelBenchmark):
    test_name: str
    run_id: str | None
    command: str
    completion_time: str | None
    benchmark_start_time: constr(regex=datetime_format)


class BenchmarkRun(BaseModelBenchmark):
    repository_info: RepositoryInfo
    run_details: RunDetails
    task_info: TaskInfo
    metrics: Metrics
    reached_cutoff: bool | None
    config: Dict[str, str | dict[str, str]]
