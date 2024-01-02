from pydantic import BaseModel, constr

datetime_format = r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}\+00:00$"


class BaseModelBenchmark(BaseModel):
    class Config:
        extra = "forbid"


class TaskInfo(BaseModelBenchmark):
    data_path: str
    is_regression: bool | None
    answer: str
    description: str
    category: list[str]
    task: str


class RepositoryInfo(BaseModelBenchmark):
    repo_url: str | None = None
    team_name: str | None = None
    agent_git_commit_sha: str | None = None
    benchmark_git_commit_sha: str | None = None


class Metrics(BaseModelBenchmark):
    cost: float | None = None
    success: bool
    attempted: bool
    difficulty: str | None = None
    run_time: str | None = None
    fail_reason: str | None = None
    success_percentage: float | None = None


class RunDetails(BaseModelBenchmark):
    test_name: str
    run_id: str | None = None
    command: str
    completion_time: str | None = None
    benchmark_start_time: constr(regex=datetime_format)


class BenchmarkRun(BaseModelBenchmark):
    repository_info: RepositoryInfo
    run_details: RunDetails
    task_info: TaskInfo
    metrics: Metrics
    reached_cutoff: bool | None = None
    config: dict[str, str | dict[str, str]]
