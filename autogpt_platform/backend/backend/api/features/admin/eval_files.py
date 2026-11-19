"""Find and read the result files the task eval and its feature eval write.

Sources, under ``backend/copilot/eval/tasks``:

- ``baseline_<name>.json``: a checked-in run's ``summary.json``; when a
  ``report_<name>.md`` sits next to it, its list of cases not passed.
- ``runs/<timestamp>/``: a task eval run made locally (``summary.json``,
  ``results.jsonl``), git-ignored.
- ``features/runs/<timestamp>/``: the same for the feature eval.

Only names matching those shapes are read, and a run is looked up by the id
the listing gave it, never by a path from the caller.
"""

import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from .eval_models import EvalSource, EvalStats, EvalSuite

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parents[3] / "copilot" / "eval" / "tasks"
MAX_FILE_BYTES = 50 * 1024 * 1024
RUN_ID_PATTERN = r"^(task|features)-[A-Za-z0-9_-]{1,80}$"

T = TypeVar("T", bound=BaseModel)

_BASELINE = re.compile(r"^baseline_([a-z0-9_]{1,60})$")
_RUN_DIR = re.compile(r"^\d{8}-\d{6}$")


class RunSource(BaseModel):
    id: str
    suite: EvalSuite
    source: EvalSource
    name: str
    summary_path: Path | None = None
    results_path: Path | None = None
    report_path: Path | None = None


class Summary(BaseModel):
    """A ``summary.json`` in one shape, whichever eval or version wrote it."""

    reviewers: list[str] = Field(default_factory=list)
    overall: dict[str, EvalStats] = Field(default_factory=dict)
    categories: dict[str, dict[str, EvalStats]] = Field(default_factory=dict)
    backend: str | None = None
    otto: str | None = None
    user: str | None = None
    drafter: str | None = None
    offline: bool = False
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    seconds_per_turn: float | None = None


class _Raw(BaseModel):
    model_config = ConfigDict(extra="ignore")


class _RawStats(_Raw):
    cases: int = 0
    score: float = 0
    pass_rate: float = 0
    partial: int = 0
    fail: int = 0


class _RawSummary(_Raw):
    models: dict[str, Any] = Field(default_factory=dict)
    setup: dict[str, Any] = Field(default_factory=dict)
    reviewers: list[str] = Field(default_factory=list)
    overall: dict[str, Any] = Field(default_factory=dict)
    categories: dict[str, Any] = Field(default_factory=dict)
    areas: dict[str, Any] = Field(default_factory=dict)
    tokens: dict[str, int] = Field(default_factory=dict)
    seconds_per_turn: float | None = None


def discover() -> list[RunSource]:
    """Every run on disk, in no particular order."""
    return [
        *_baselines(),
        *_local_runs("task", EVAL_DIR / "runs"),
        *_local_runs("features", EVAL_DIR / "features" / "runs"),
    ]


def read_text(path: Path) -> str:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise ValueError(f"{path.name} is too large to read")
    return path.read_text(encoding="utf-8")


def read_summary(source: RunSource) -> Summary | None:
    if source.summary_path is None:
        return None
    raw = _RawSummary.model_validate_json(read_text(source.summary_path))
    if "cases" in raw.overall:
        return _single_reviewer(raw)
    return _per_reviewer(raw)


def read_result_lines(path: Path) -> list[str]:
    return [line for line in read_text(path).splitlines() if line.strip()]


def started_at(name: str) -> datetime | None:
    """The UTC time a run folder (or a report heading) is named after."""
    try:
        return datetime.strptime(name, "%Y%m%d-%H%M%S").replace(tzinfo=UTC)
    except ValueError:
        return None


def load_json_list(folder: Path) -> list[dict[str, Any]]:
    """Every entry of every ``*.json`` list in a folder; a broken file is
    logged and left out."""
    entries: list[dict[str, Any]] = []
    for path in sorted(folder.glob("*.json")) if folder.is_dir() else []:
        try:
            entries += json.loads(read_text(path))
        except (OSError, ValueError) as e:
            logger.warning(
                f"Eval cases {path.name} could not be read: {type(e).__name__}"
            )
    return entries


def parse_or_none(model: type[T], line: str) -> T | None:
    try:
        return model.model_validate_json(line)
    except ValidationError:
        logger.warning("Skipping an eval result line that does not parse")
        return None


def _baselines() -> list[RunSource]:
    sources: list[RunSource] = []
    for path in sorted(EVAL_DIR.glob("baseline_*.json")):
        match = _BASELINE.match(path.stem)
        if not match or not _plain_file(path):
            continue
        report = path.with_name(f"report_{match.group(1)}.md")
        sources.append(
            RunSource(
                id=f"task-{path.stem}",
                suite="task",
                source="baseline",
                name=path.stem,
                summary_path=path,
                report_path=report if _plain_file(report) else None,
            )
        )
    return sources


def _local_runs(suite: EvalSuite, root: Path) -> list[RunSource]:
    if not root.is_dir() or root.is_symlink():
        return []
    folders = [
        d for d in sorted(root.iterdir()) if _RUN_DIR.match(d.name) and _plain_dir(d)
    ]
    sources = [
        RunSource(
            id=f"{suite}-{d.name}",
            suite=suite,
            source="local",
            name=d.name,
            summary_path=_file_or_none(d / "summary.json"),
            results_path=_file_or_none(d / "results.jsonl"),
        )
        for d in folders
    ]
    return [s for s in sources if s.summary_path or s.results_path]


def _plain_file(path: Path) -> bool:
    return path.is_file() and not path.is_symlink()


def _plain_dir(path: Path) -> bool:
    return path.is_dir() and not path.is_symlink()


def _file_or_none(path: Path) -> Path | None:
    return path if _plain_file(path) else None


def _single_reviewer(raw: _RawSummary) -> Summary:
    """The first task eval runs had one judge and flat figures."""
    name = str(raw.models.get("judge") or "reviewer")
    return Summary(
        reviewers=[name],
        overall={name: _stats(raw.overall)},
        categories={k: {name: _stats(v)} for k, v in raw.categories.items()},
        backend=raw.models.get("backend") or "openai",
        otto=raw.models.get("otto"),
        user=raw.models.get("user"),
        **_usage(raw),
    )


def _per_reviewer(raw: _RawSummary) -> Summary:
    setup = raw.setup or raw.models
    reviewers = raw.reviewers or list(setup.get("reviewers") or raw.overall)
    per_category = raw.areas or raw.categories
    return Summary(
        reviewers=reviewers,
        overall={k: _stats(v) for k, v in raw.overall.items()},
        categories={
            name: {k: _stats(v) for k, v in per.items()}
            for name, per in per_category.items()
        },
        backend=setup.get("backend"),
        otto=setup.get("otto") or setup.get("model"),
        user=setup.get("user"),
        drafter=setup.get("drafter"),
        offline=bool(setup.get("offline")),
        **_usage(raw),
    )


def _usage(raw: _RawSummary) -> dict[str, Any]:
    return {
        "prompt_tokens": raw.tokens.get("prompt"),
        "completion_tokens": raw.tokens.get("completion"),
        "seconds_per_turn": raw.seconds_per_turn,
    }


def _stats(value: Any) -> EvalStats:
    raw = _RawStats.model_validate(value)
    return EvalStats(
        cases=raw.cases,
        score=raw.score,
        pass_rate=raw.pass_rate,
        passed=round(raw.pass_rate * raw.cases / 100),
        partial=raw.partial,
        failed=raw.fail,
    )
