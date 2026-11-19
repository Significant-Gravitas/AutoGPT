"""The commands that make a run locally. The page shows them instead of a
"run" button: every task eval case and the feature eval's model tier call
paid models, and the feature eval's free ``--offline`` tier patches the task
and suggestion database accessors on their modules, which must not happen
inside the API server."""

from .eval_files import Summary
from .eval_models import EvalCommand, EvalSuite

_TASK = "poetry run python -m backend.copilot.eval.tasks"
_FEATURES = "poetry run python -m backend.copilot.eval.tasks.features"
_FROM = "cd autogpt_platform/backend && "

DEFAULT_COMMANDS = [
    EvalCommand(
        label="Feature eval, code-only cases",
        command=f"{_FROM}{_FEATURES} --offline",
        cost="Free: no model, no database.",
    ),
    EvalCommand(
        label="Feature eval, what a full run would cost",
        command=f"{_FROM}{_FEATURES} --dry-run "
        "--reviewer claude:claude-opus-5-5 --reviewer cursor:grok-4.7-medium-fast",
        cost="Free: lists the cases and the estimate, calls nothing.",
    ),
    EvalCommand(
        label="Task eval, every case",
        command=f"{_FROM}{_TASK} --backend claude-native --model claude-opus-5-5 "
        "--user-model claude-sonnet-5-5 --reviewer claude:claude-opus-5-5 "
        "--reviewer cursor:grok-4.7-medium-fast",
        cost="Paid: about 250 cases, each a model run and one review per reviewer.",
    ),
]


def rerun_commands(suite: EvalSuite, summary: Summary | None) -> list[EvalCommand]:
    """The command that repeats a run's own setup, then the free ones."""
    if summary is None:
        return DEFAULT_COMMANDS
    command = _features(summary) if suite == "features" else _task(summary)
    return [command, *[c for c in DEFAULT_COMMANDS if c.command != command.command]]


def _task(summary: Summary) -> EvalCommand:
    parts = [
        _TASK,
        f"--backend {summary.backend or 'openai'}",
        f"--model {summary.otto}" if summary.otto else "",
        f"--user-model {summary.user}" if summary.user else "",
        *_reviewer_flags(summary),
    ]
    return EvalCommand(
        label="Repeat this run",
        command=_FROM + " ".join(p for p in parts if p),
        cost="Paid: every case is a model run plus one review per reviewer.",
    )


def _features(summary: Summary) -> EvalCommand:
    if summary.offline:
        return DEFAULT_COMMANDS[0]
    parts = [
        _FEATURES,
        f"--backend {summary.backend}" if summary.backend else "",
        f"--model {summary.otto}" if summary.otto else "",
        f"--drafter {summary.drafter}" if summary.drafter else "",
        *_reviewer_flags(summary),
    ]
    return EvalCommand(
        label="Repeat this run",
        command=_FROM + " ".join(p for p in parts if p),
        cost="Paid: the model cases call Otto, the drafter and every reviewer.",
    )


def _reviewer_flags(summary: Summary) -> list[str]:
    return [f"--reviewer {r}" for r in summary.reviewers if ":" in r and r != "code"]
